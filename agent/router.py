"""The routing brain (rules-compliant): plain code, else Fireworks.

Per the organizers: only calls through FIREWORKS_BASE_URL are scored; there is no
local-LLM tier. "Routing intelligence" = decide when a task can be answered with
plain deterministic CODE (zero tokens) vs. when it needs an LLM call (the cheapest
ALLOWED_MODELS Fireworks model, with the fewest possible tokens).

Per task:
  1. Try free deterministic solvers (arithmetic, ordering, syllogism, …) — 0 tokens.
  2. Otherwise call the cheapest capable Fireworks model with a minimal prompt.
"""
from __future__ import annotations

import re

from .classifier import classify
from .config import config
from .prompts import build_batch_messages, build_messages, build_remote_messages, max_tokens_for
from .solvers import free_solve

# short-answer categories safe to batch into one Fireworks call
BATCHABLE = {"sentiment", "factual", "math", "logic"}

_BATCH_LINE = re.compile(r"(?m)^\s*(\d+)\s*[\).:\-]\s*(.+?)\s*$")


def parse_batch(text: str, n: int) -> list[str] | None:
    """Extract exactly n 'N) <answer>' lines, in order; None if it doesn't line up."""
    found: dict[int, str] = {}
    for m in _BATCH_LINE.finditer(text or ""):
        i = int(m.group(1))
        if 1 <= i <= n and i not in found:
            found[i] = m.group(2).strip()
    if len(found) == n:
        return [found[i] for i in range(1, n + 1)]
    return None


def _pick_remote_model(category: str) -> str:
    """Choose a model from the harness-injected ALLOWED_MODELS.

    Score is by token count; live testing showed reasoning models are far more
    verbose. Prefer a compact non-reasoning instruct model (Gemma) for most tasks
    and a code-specialized model for code — matched by substring so it works for
    whatever exact IDs the harness injects. A configured preferred model wins if
    allowed; otherwise the first allowed model.
    """
    models = config.allowed_models
    if not models:
        return ""
    if config.preferred_model and config.preferred_model in models:
        return config.preferred_model

    def find(sub: str):
        return next((m for m in models if sub in m.lower()), None)

    if category in ("code_gen", "code_debug"):
        return find("code") or find("gemma") or models[0]
    return find("gemma") or models[0]


def _fireworks(task_id, category, prompt, remote, *, full_prompt=False):
    """One minimal Fireworks call; returns the result dict with token count."""
    model = _pick_remote_model(category)
    builder = build_messages if full_prompt else build_remote_messages
    before = remote.meter.total
    try:
        out = remote.chat(model, builder(category, prompt),
                          max_tokens=max_tokens_for(category), temperature=0.0, n=1,
                          reasoning_effort=config.reasoning_effort)
        return {"task_id": task_id, "answer": out[0].strip(), "route": "remote",
                "category": category, "tokens": remote.meter.total - before, "model": model}
    except Exception as e:
        return {"task_id": task_id, "answer": "", "route": "error",
                "category": category, "tokens": remote.meter.total - before, "error": str(e)}


def fireworks_batch(category: str, tasks: list[dict], remote) -> list[dict]:
    """One Fireworks call for a group of same-category short-answer tasks.

    Falls back to individual calls if the batched response can't be parsed into
    exactly one answer per task — so accuracy is never sacrificed (only tokens).
    """
    if len(tasks) == 1:
        t = tasks[0]
        return [_fireworks(t.get("task_id"), category, t.get("prompt", ""), remote)]

    model = _pick_remote_model(category)
    prompts = [t.get("prompt", "") for t in tasks]
    before = remote.meter.total
    try:
        out = remote.chat(model, build_batch_messages(category, prompts),
                          max_tokens=max_tokens_for(category) * len(tasks) + 40,
                          temperature=0.0, n=1, reasoning_effort=config.reasoning_effort)
        answers = parse_batch(out[0], len(tasks))
    except Exception:
        answers = None
    spent = remote.meter.total - before

    if answers is None:  # parse/format failure -> individual calls (accuracy-safe)
        return [_fireworks(t.get("task_id"), category, t.get("prompt", ""), remote) for t in tasks]

    return [{"task_id": t.get("task_id"), "answer": a.strip(), "route": "remote-batch",
             "category": category, "tokens": (spent if i == 0 else 0), "model": model}
            for i, (t, a) in enumerate(zip(tasks, answers))]


def route(task: dict, remote) -> dict:
    task_id = task.get("task_id")
    prompt = task.get("prompt", "")
    category = classify(prompt)

    # baseline mode (eval only): send everything to Fireworks with full prompts
    if config.force_remote and config.has_remote():
        return _fireworks(task_id, category, prompt, remote, full_prompt=True)

    # 1) plain-code deterministic solvers — 0 tokens (the only free path)
    solved = free_solve(category, prompt)
    if solved is not None:
        return {"task_id": task_id, "answer": solved, "route": "code",
                "category": category, "tokens": 0, "confidence": 1.0}

    # 2) everything else -> cheapest capable Fireworks model, minimal prompt
    if config.has_remote():
        return _fireworks(task_id, category, prompt, remote)

    return {"task_id": task_id, "answer": "", "route": "no-remote", "category": category, "tokens": 0}
