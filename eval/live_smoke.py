"""Live remote-path smoke test against real Fireworks (cost-capped, ~8 calls).

Validates that our minimal prompts produce correct, concise answers on a real
model across all 8 categories, and reports REAL per-category token cost. Needs
FIREWORKS_API_KEY + ALLOWED_MODELS in .env. Does NOT need a local model.

    python -m eval.live_smoke
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.backends import Model, RemoteMeter          # noqa: E402
from agent.config import config                          # noqa: E402
from agent.prompts import build_remote_messages, max_tokens_for  # noqa: E402
from eval.judge import judge_one                         # noqa: E402

HERE = Path(__file__).resolve().parent
TASKS = {t["task_id"]: t for t in json.loads((HERE / "datasets" / "dev_tasks.json").read_text(encoding="utf-8"))}
EXPECTED = json.loads((HERE / "datasets" / "dev_expected.json").read_text(encoding="utf-8"))

# one representative task per category
SAMPLE = ["f3", "m3", "s1", "u2", "n1", "d1", "l1", "g1"]


def main():
    model = config.preferred_model or (config.allowed_models[0] if config.allowed_models else "")
    if not config.fireworks_api_key or not model:
        print("Set FIREWORKS_API_KEY + ALLOWED_MODELS in .env first.")
        return 1
    meter = RemoteMeter()
    remote = Model(config.fireworks_base_url, config.fireworks_api_key, 120, meter=meter)

    print(f"\n  Live remote smoke · model={model.split('/')[-1]} · reasoning={config.reasoning_effort}\n")
    print(f"  {'cat':<14}{'ok':>3}  {'tokens':>7}  answer")
    print("  " + "-" * 60)
    passed = 0
    for tid in SAMPLE:
        task, exp = TASKS[tid], EXPECTED[tid]
        cat = exp["category"]
        before = meter.total
        try:
            out = remote.chat(model, build_remote_messages(cat, task["prompt"]),
                              max_tokens=max_tokens_for(cat), temperature=0.0,
                              reasoning_effort=config.reasoning_effort)
            ans = out[0].strip()
        except Exception as e:
            print(f"  {cat:<14}ERR  {'-':>7}  {e}")
            continue
        ok = judge_one(exp["check"], ans)
        passed += ok
        spent = meter.total - before
        shown = ans.replace("\n", " ")[:40]
        print(f"  {cat:<14}{'✓' if ok else '✗':>3}  {spent:>7}  {shown!r}")

    print("  " + "-" * 60)
    print(f"  accuracy {passed}/{len(SAMPLE)} · total remote tokens {meter.total} "
          f"(avg {meter.total // max(1, meter.calls)}/call)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
