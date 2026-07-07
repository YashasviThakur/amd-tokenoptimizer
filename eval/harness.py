"""Local eval harness — measure accuracy + Fireworks tokens before submitting.

Boots the mock model server, runs the agent as an all-Fireworks BASELINE and as
the HYBRID router, grades both with the local judge, and reports token savings.
`--sweep` tries several escalation thresholds so we can pick the best operating
point (highest tokens saved while still clearing the accuracy gate).

    python -m eval.harness                 # baseline vs hybrid @ default threshold
    python -m eval.harness --sweep         # threshold sweep table
    python -m eval.harness --gate 0.8      # set the assumed accuracy gate
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8900
BASE = f"http://127.0.0.1:{PORT}/v1"
_TASKS_PATH = HERE / "datasets" / "dev_tasks.json"
_EXPECTED_PATH = HERE / "datasets" / "dev_expected.json"
EXPECTED = {}


def _env():
    os.environ["LOCAL_MODEL"] = "local-small"
    os.environ["ALLOWED_MODELS"] = "fw-large-70b,fw-small-8b"
    os.environ["LOCAL_BASE_URL"] = BASE
    os.environ["FIREWORKS_BASE_URL"] = BASE
    os.environ["FIREWORKS_API_KEY"] = "mock"
    os.environ["LOCAL_API_KEY"] = "mock"
    os.environ["INPUT_PATH"] = str(_TASKS_PATH)
    os.environ["OUTPUT_PATH"] = str(HERE / "out" / "results.json")
    os.environ["MOCK_TASKS"] = str(_TASKS_PATH)
    os.environ["MOCK_EXPECTED"] = str(_EXPECTED_PATH)


def _start_server():
    env = dict(os.environ, PYTHONPATH=str(ROOT), LOCAL_MODEL="local-small")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "eval.mock_server:app",
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(150):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200:
                return proc
        except Exception:
            pass
        time.sleep(0.1)
    proc.terminate()
    raise RuntimeError("mock server did not start")


def _run(force_remote: bool, threshold: float, out_name: str):
    import agent.config as C
    import agent.main as M

    C.config.force_remote = force_remote
    C.config.escalate_threshold = threshold
    C.config.output_path = str(HERE / "out" / out_name)

    summary = M.run()
    results = json.loads(Path(C.config.output_path).read_text(encoding="utf-8"))
    meta = json.loads(Path(C.config.output_path + ".meta.json").read_text(encoding="utf-8"))
    from eval.judge import judge_all
    judged = judge_all(results, EXPECTED)
    return summary, results, meta, judged


def _per_category(meta, judged):
    passed = {p["task_id"]: p["pass"] for p in judged["per"]}
    cats = {}
    for m in meta:
        c = m["category"]
        d = cats.setdefault(c, {"n": 0, "local": 0, "remote": 0, "correct": 0, "tokens": 0})
        d["n"] += 1
        if m["route"] == "remote":
            d["remote"] += 1
        else:
            d["local"] += 1
        d["tokens"] += m.get("tokens", 0)
        d["correct"] += 1 if passed.get(m["task_id"]) else 0
    return cats


def _kept_local(meta):
    return sum(1 for m in meta if m["route"] != "remote")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--gate", type=float, default=0.80)
    ap.add_argument("--profile", choices=["generic", "strong"], default="generic",
                    help="simulated local-model quality (generic 2B vs strong code-capable ~3-4B)")
    ap.add_argument("--tasks", help="path to a tasks.json (default: dev set)")
    ap.add_argument("--expected", help="path to an expected.json (default: dev set)")
    args = ap.parse_args()

    global _TASKS_PATH, _EXPECTED_PATH, EXPECTED
    if args.tasks:
        _TASKS_PATH = Path(args.tasks)
    if args.expected:
        _EXPECTED_PATH = Path(args.expected)
    EXPECTED = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))
    os.environ["MOCK_LOCAL_PROFILE"] = args.profile
    _env()
    (HERE / "out").mkdir(exist_ok=True)
    proc = _start_server()
    try:
        base_sum, _, _, base_j = _run(force_remote=True, threshold=1.0, out_name="baseline.json")
        base_tokens = base_sum["fireworks_tokens"]

        print("\n" + "=" * 68)
        print("  AMD ACT II · Track 1 — Hybrid Token-Efficient Routing Agent")
        print(f"  Local eval harness (mock models · local profile = {args.profile})")
        print("=" * 68)
        print(f"\n  BASELINE (every task → Fireworks)")
        print(f"    accuracy         {base_j['accuracy']*100:5.1f}%  ({base_j['passed']}/{base_j['total']})")
        print(f"    Fireworks tokens {base_tokens:6d}")

        if args.sweep:
            print(f"\n  THRESHOLD SWEEP  (gate = {args.gate*100:.0f}% accuracy)")
            print(f"    {'thresh':>7}  {'acc':>6}  {'gate':>5}  {'kept_local':>10}  {'fw_tokens':>9}  {'saved':>6}")
            print("    " + "-" * 56)
            for thr in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
                s, _, meta, j = _run(False, thr, "results.json")
                kl = _kept_local(meta)
                saved = 100 * (1 - s["fireworks_tokens"] / base_tokens) if base_tokens else 0
                gate = "PASS" if j["accuracy"] >= args.gate else "fail"
                print(f"    {thr:>7.2f}  {j['accuracy']*100:5.1f}%  {gate:>5}  "
                      f"{kl:>3}/{len(meta):<6}  {s['fireworks_tokens']:>9d}  {saved:5.1f}%")
            print()
            return

        hyb_sum, _, meta, hyb_j = _run(False, args.threshold, "results.json")
        hyb_tokens = hyb_sum["fireworks_tokens"]
        saved = 100 * (1 - hyb_tokens / base_tokens) if base_tokens else 0
        gate = "PASS ✓" if hyb_j["accuracy"] >= args.gate else "FAIL ✗"

        print(f"\n  HYBRID ROUTER (threshold = {args.threshold})")
        print(f"    accuracy         {hyb_j['accuracy']*100:5.1f}%  ({hyb_j['passed']}/{hyb_j['total']})   "
              f"accuracy gate ({args.gate*100:.0f}%): {gate}")
        print(f"    Fireworks tokens {hyb_tokens:6d}   ({saved:.1f}% fewer than baseline)")
        print(f"    kept local       {_kept_local(meta)}/{len(meta)}   routes={hyb_sum['routes']}")

        print(f"\n  PER-CATEGORY")
        print(f"    {'category':<14}{'n':>3}  {'local':>5}  {'remote':>6}  {'fw_tokens':>9}  {'accuracy':>8}")
        print("    " + "-" * 54)
        for cat, d in sorted(_per_category(meta, hyb_j).items()):
            print(f"    {cat:<14}{d['n']:>3}  {d['local']:>5}  {d['remote']:>6}  {d['tokens']:>9}  "
                  f"{d['correct']/d['n']*100:>7.0f}%")
        print()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
