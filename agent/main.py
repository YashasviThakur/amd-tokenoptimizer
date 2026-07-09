"""Entrypoint: /input/tasks.json -> route each task -> /output/results.json.

Robustness is a scoring requirement: malformed output scores zero, so input is
parsed defensively, every task is wrapped, answers are coerced to strings, the
results file is always valid JSON, and we always exit 0.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .backends import LocalModel, Model, RemoteMeter
from .config import config
from .router import route

_PROMPT_KEYS = ("prompt", "question", "input", "text", "query", "task")


def _extract_prompt(raw) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for k in _PROMPT_KEYS:
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""


def _normalize_tasks(data) -> list[dict]:
    """Accept a list, {"tasks": [...]}, a dict of id->task, or a single task."""
    if isinstance(data, dict):
        if "tasks" in data and isinstance(data["tasks"], list):
            data = data["tasks"]
        elif any(k in data for k in _PROMPT_KEYS):
            data = [data]  # a single task object
        else:  # mapping of id -> task
            data = [{"task_id": k, **(v if isinstance(v, dict) else {"prompt": v})}
                    for k, v in data.items()]
    if not isinstance(data, list):
        data = [data]

    tasks = []
    for i, raw in enumerate(data):
        tid = raw.get("task_id") if isinstance(raw, dict) else None
        if tid is None and isinstance(raw, dict):
            tid = raw.get("id")
        if tid is None:
            tid = f"t{i + 1}"
        # contract requires STRING task_ids; an int/float/list id would emit a
        # non-string and fail INVALID_RESULTS_SCHEMA. str() on a str is a no-op.
        tasks.append({"task_id": str(tid), "prompt": _extract_prompt(raw)})
    return tasks


def _answer_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _write_json(path: str, obj) -> None:
    """Atomic write: a crash mid-write can never leave a half-written (invalid)
    results.json — we write to a temp file and os.replace() it into place."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _build_local():
    """Load the bundled local model once (llama-cpp-python, CPU). Returns None if
    disabled or unavailable — the agent then falls back to solvers + Fireworks."""
    if not config.use_local:
        return None
    # AVX2 preflight: the llama.cpp build targets AVX2/FMA/F16C. On a CPU without
    # them the first op raises SIGILL — an UNCATCHABLE signal that kills the process
    # before any output is written (guaranteed ZERO). Detect it and degrade to a
    # valid Fireworks-only run instead.
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            if "avx2" not in f.read().lower():
                print("[agent] CPU lacks AVX2; skipping local model (Fireworks-only)", file=sys.stderr)
                return None
    except Exception:
        pass  # not Linux / no /proc (dev box) — proceed
    try:
        threads = config.local_threads or None  # 0 -> llama.cpp default
        m = LocalModel(config.local_model_path, n_ctx=config.local_n_ctx, n_threads=threads)
        print(f"[agent] local model loaded: {config.local_model_path}", file=sys.stderr)
        return m
    except Exception as e:
        print(f"[agent] local model unavailable ({e}); Fireworks-only", file=sys.stderr)
        return None


def _diagnose_env() -> None:
    """Log the resolved remote config (masked) so a total remote-call failure in
    a sandbox we can't inspect directly is diagnosable from container stderr
    instead of guessed at from the aggregate accuracy number alone."""
    key = config.fireworks_api_key
    masked = (key[:4] + "…" + key[-2:]) if len(key) > 8 else ("<empty>" if not key else "<short>")
    print(f"[agent] base_url={config.fireworks_base_url!r} api_key={masked} "
          f"allowed_models={config.allowed_models!r} preferred_model={config.preferred_model!r} "
          f"has_remote={config.has_remote()} use_local={config.use_local} "
          f"reasoning_effort={config.reasoning_effort!r}", file=sys.stderr)


def _resolve_models(remote) -> None:
    """Cross-check our model list against GET /models — but NEVER displace an
    injected ALLOWED_MODELS: that list is authoritative (the judging proxy
    matches entries verbatim; calling anything else is a MODEL_VIOLATION), and a
    proxy catalog with a different naming scheme must not evict the correct
    names. Discovery only (a) narrows to the intersection when one exists, or
    (b) supplies a list when we have NONE configured. Off by default."""
    if not config.model_discovery:
        return
    served = remote.list_models()
    if not served:
        return
    # drop obvious non-chat models (image / embedding / audio) so fallback never
    # wastes an attempt calling one for a text answer.
    _NON_CHAT = ("flux", "stable-diffusion", "sdxl", "playground", "embed", "embedding",
                 "whisper", "dall-e", "clip", "rerank", "-vision-")
    served = [m for m in served if not any(k in m.lower() for k in _NON_CHAT)] or served
    configured = config.allowed_models
    inter = [m for m in configured if m in served]
    if inter:
        config.allowed_models = inter
    elif not configured:
        config.allowed_models = served
    else:
        print(f"[agent] configured models {configured[:8]} not in /models catalog; "
              f"KEEPING them verbatim (allow-list is authoritative)", file=sys.stderr)
    print(f"[agent] resolved models: {config.allowed_models[:8]}", file=sys.stderr)


def run() -> dict:
    t0 = time.time()
    _diagnose_env()
    meter = RemoteMeter()
    remote = Model(config.fireworks_base_url, config.fireworks_api_key, config.request_timeout, meter=meter)
    if config.has_remote():
        _resolve_models(remote)
    local = _build_local()

    try:
        with open(config.input_path, encoding="utf-8") as f:
            tasks = _normalize_tasks(json.load(f))
    except Exception as e:
        print(f"[agent] FATAL: cannot read {config.input_path}: {e}", file=sys.stderr)
        _write_json(config.output_path, [])  # valid JSON, still exit 0
        return {"tasks": 0, "error": str(e)}

    # Route tasks CONCURRENTLY (config.max_workers at a time). Sequential Fireworks
    # calls made a large hidden set overrun the 10-min budget on the grader's slower
    # network -> remaining tasks emitted empty -> failed accuracy gate. Concurrency
    # cuts wall-clock ~Nx so the whole set finishes in time. Guarantees preserved:
    #  * HARD deadline: any task not finished by then is emitted empty (a slow/huge
    #    set can never blow the budget; the harness SIGKILLs at ~600s = ZERO).
    #  * results stay in input order; incremental atomic writes survive a mid-run kill.
    n = len(tasks)
    routes: dict = {}
    # pre-seed every slot with a valid empty answer so a task that never completes
    # (deadline / crash) still appears in a well-formed results.json.
    results = [{"task_id": str(t.get("task_id")), "answer": ""} for t in tasks]
    meta = [{"task_id": t.get("task_id"), "route": "deadline-skip", "tokens": 0} for t in tasks]
    hard_deadline = config.run_deadline_s + 60.0

    def _work(task):
        try:
            return route(task, local, remote, prefer_remote=False)
        except Exception as e:  # never let one task sink the batch
            return {"task_id": task.get("task_id"), "answer": "", "route": "error",
                    "category": "?", "tokens": 0, "error": str(e)}

    done_count = 0
    with ThreadPoolExecutor(max_workers=max(1, config.max_workers)) as ex:
        fut_to_idx = {ex.submit(_work, task): i for i, task in enumerate(tasks)}
        pending = set(fut_to_idx)
        while pending:
            remaining = hard_deadline - (time.time() - t0)
            if remaining <= 0:  # out of time — leave the rest as their empty pre-seed
                skipped = len(pending)
                print(f"[agent] hard deadline {hard_deadline:.0f}s hit; "
                      f"emitting {skipped} unfinished tasks as empty", file=sys.stderr)
                for f in pending:
                    f.cancel()
                routes["deadline-skip"] = routes.get("deadline-skip", 0) + skipped
                break
            just_done, pending = wait(pending, timeout=min(remaining, 5.0),
                                      return_when=FIRST_COMPLETED)
            for f in just_done:
                i = fut_to_idx[f]
                r = f.result()
                routes[r.get("route", "?")] = routes.get(r.get("route", "?"), 0) + 1
                if r.get("error"):
                    print(f"[agent] task {r.get('task_id')} ({r.get('category')}) failed: {r['error']}",
                          file=sys.stderr)
                results[i] = {"task_id": str(r.get("task_id")), "answer": _answer_str(r.get("answer"))}
                meta[i] = {"task_id": r.get("task_id"), "route": r.get("route"),
                           "category": r.get("category"), "confidence": r.get("confidence"),
                           "tokens": r.get("tokens") or 0, "error": r.get("error")}
                done_count += 1
                if done_count % 8 == 0:  # periodic atomic flush -> a kill leaves a valid partial file
                    try:
                        _write_json(config.output_path, results)
                    except Exception:
                        pass

    try:
        _write_json(config.output_path, results)
    except Exception as e:  # last resort — never crash; write SOMETHING valid
        print(f"[agent] primary write failed: {e}", file=sys.stderr)
        try:
            with open(config.output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, default=str)
        except Exception:
            pass
    try:  # diagnostics sidecar for the eval harness (ignored by the judging harness)
        _write_json(config.output_path + ".meta.json", meta)
    except Exception:
        pass

    summary = {
        "tasks": len(results), "seconds": round(time.time() - t0, 2), "routes": routes,
        "fireworks_calls": meter.calls, "fireworks_tokens": meter.total,
        "prompt_tokens": meter.prompt_tokens, "completion_tokens": meter.completion_tokens,
    }
    print(f"[agent] {summary}")
    return summary


def selftest() -> int:
    """Offline container health check: run solver-answerable tasks with remote
    disabled and validate the output contract. Prints PASS/FAIL, exits accordingly.
    """
    import tempfile

    sample = [
        {"task_id": "st1", "prompt": "What is 12 * 12?"},
        {"task_id": "st2", "prompt": "Alice is taller than Bob. Bob is taller than Carol. Who is the shortest?"},
        {"task_id": "st3", "prompt": "If all A are B and all B are C, are all A C? Answer yes or no."},
        # discount solver must anchor the price (not grab the first number = 20%)
        {"task_id": "st4", "prompt": "A shirt is discounted by 20%. The original price is $50. Sale price in dollars?"},
        # compound expressions must DEFER (offline => empty answer); a wrong-number
        # regression in the operand-count / percent guards would fail these.
        {"task_id": "st5", "prompt": "What is 2 to the power of 3 plus 1?"},
        {"task_id": "st6", "prompt": "What is 20% of 50 plus 5?"},
        # race ordering ("ahead of" + "won") and profit% solvers
        {"task_id": "st7", "prompt": "In a race, Maya finished ahead of Leo, and Leo finished ahead of Nina. Who won?"},
        {"task_id": "st8", "prompt": "A shopkeeper buys an item for $80 and sells it for $100. What is the profit percentage?"},
        # red-teamed misfire regressions: a year RANGE must not be eval'd as
        # subtraction, and a REVERSE discount must defer (forward formula = wrong)
        {"task_id": "st9", "prompt": "How many years did World War I last, from 1914-1918?"},
        {"task_id": "st10", "prompt": "After a 20% discount, a shirt costs $40. What was the original price?"},
    ]
    d = tempfile.mkdtemp()
    inp, outp = os.path.join(d, "tasks.json"), os.path.join(d, "results.json")
    _write_json(inp, sample)

    config.input_path, config.output_path = inp, outp
    config.allowed_models = []      # force offline: no remote model list
    config.fireworks_api_key = ""   # ...and no key, so has_remote() is False offline
    config.use_local = False        # solver + contract check only (no model load)
    run()

    try:
        out = json.loads(open(outp, encoding="utf-8").read())
        by = {o["task_id"]: o["answer"] for o in out}
        ok = (isinstance(out, list) and len(out) == 10
              and all(isinstance(o.get("task_id"), str) and isinstance(o.get("answer"), str) for o in out)
              and "144" in by["st1"]
              and "carol" in by["st2"].lower()
              and "yes" in by["st3"].lower()
              and "40" in by["st4"]                       # discount anchored correctly
              and by["st5"].strip() == ""                 # power+add: solver deferred
              and by["st6"].strip() == ""                 # percent+add: solver deferred
              and "maya" in by["st7"].lower()             # race ordering (ahead of / won)
              and "25" in by["st8"]                       # profit percentage
              and by["st9"].strip() == ""                 # year range: no -4 misfire
              and by["st10"].strip() == "")               # reverse discount: deferred
    except Exception as e:
        print(f"[selftest] FAIL: {e}")
        return 1
    print("[selftest] PASS" if ok else f"[selftest] FAIL: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    try:
        run()
    except Exception as e:  # a non-zero exit or missing results.json = ZERO score
        print(f"[agent] FATAL: {e}", file=sys.stderr)
        try:  # guarantee *some* valid results.json exists if run() died early
            if not os.path.exists(config.output_path):
                _write_json(config.output_path, [])
        except Exception:
            pass
    sys.exit(0)
