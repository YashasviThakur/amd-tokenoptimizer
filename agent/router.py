"""The routing brain: answer locally when we can trust it, escalate when we can't.

Per task:
  1. math short-circuit — a pure-arithmetic question is solved exactly for free.
  2. local answer (with self-consistency sampling on hard categories).
  3. confidence = category prior + free verifier signals + sample agreement.
  4. confident? keep the local answer (0 tokens). Otherwise escalate to the
     smallest-but-capable Fireworks model with a tiny prompt.
"""
from __future__ import annotations

import re

from . import verifiers as V
from .classifier import HARD, classify
from .config import config
from .prompts import build_messages, build_remote_messages, max_tokens_for
from .solvers import free_solve

# Base trust per category (how often a small local model is right, roughly).
PRIOR = {
    "sentiment": 0.80, "ner": 0.72, "summarization": 0.74, "factual": 0.68,
    "math": 0.40, "logic": 0.38, "code_debug": 0.45, "code_gen": 0.45,
}

# Categories with no cheap correctness verifier — use self-consistency sampling
# (two local draws; disagreement is a free signal that the small model is unsure).
SELF_CONSISTENCY = HARD | {"factual"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _confidence(category: str, prompt: str, samples: list[str]) -> float:
    ans = samples[0] if samples else ""
    c = PRIOR.get(category, 0.6)

    if len(samples) > 1:
        c += 0.20 if _norm(samples[0]) == _norm(samples[1]) else -0.30

    if category == "math":
        c += 0.15 if V.is_number(ans) else -0.45
    elif category == "sentiment":
        c += 0.20 if V.label_ok(ans) else -0.50
    elif category == "ner":
        c += 0.20 if V.valid_json(ans) else -0.45
    elif category in ("code_gen", "code_debug"):
        c += 0.15 if V.code_compiles(ans) else -0.35
    elif category == "summarization":
        c += 0.10 if V.length_ok(prompt, ans) else -0.15

    return max(0.0, min(1.0, c))


def _pick_remote_model(category: str) -> str:
    """Choose a model from the harness-injected ALLOWED_MODELS.

    Token score is by count (not price), and live testing showed reasoning models
    are far more verbose. So we prefer a compact non-reasoning instruct model
    (Gemma) for most tasks and a code-specialized model for code — matched by
    substring so it works whatever exact IDs the harness injects. A configured
    preferred model wins if allowed; otherwise fall back to the first allowed.
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


def route(task: dict, local, remote) -> dict:
    """Return {task_id, answer, route, category, tokens, confidence}."""
    task_id = task.get("task_id")
    prompt = task.get("prompt", "")
    category = classify(prompt)

    # baseline mode (eval only): send everything straight to Fireworks (naive)
    if config.force_remote and config.has_remote():
        model = _pick_remote_model(category)
        before = remote.meter.total
        out = remote.chat(model, build_messages(category, prompt),
                         max_tokens=max_tokens_for(category), temperature=0.0, n=1,
                         reasoning_effort=config.reasoning_effort)
        return {"task_id": task_id, "answer": out[0].strip(), "route": "remote",
                "category": category, "tokens": remote.meter.total - before,
                "confidence": 0.0, "model": model}

    # 1) free deterministic solvers (arithmetic, ordering, syllogism) — 0 tokens
    solved = free_solve(category, prompt)
    if solved is not None:
        return {"task_id": task_id, "answer": solved, "route": "local-solver",
                "category": category, "tokens": 0, "confidence": 1.0}

    # 2) local answer(s)
    messages = build_messages(category, prompt)
    n = config.local_samples_hard if category in SELF_CONSISTENCY else 1
    try:
        samples = local.chat(config.local_model, messages,
                             max_tokens=max_tokens_for(category), temperature=0.0 if n == 1 else 0.4, n=n)
    except Exception:
        samples = []

    # 3) confidence
    conf = _confidence(category, prompt, samples) if samples else 0.0

    # 4) keep local, or escalate
    if samples and conf >= config.escalate_threshold:
        return {"task_id": task_id, "answer": samples[0].strip(), "route": "local",
                "category": category, "tokens": 0, "confidence": round(conf, 3)}

    if config.has_remote():
        model = _pick_remote_model(category)
        try:
            before = remote.meter.total
            out = remote.chat(model, build_remote_messages(category, prompt),
                             max_tokens=max_tokens_for(category), temperature=0.0, n=1,
                             reasoning_effort=config.reasoning_effort)
            return {"task_id": task_id, "answer": out[0].strip(), "route": "remote",
                    "category": category, "tokens": remote.meter.total - before,
                    "confidence": round(conf, 3), "model": model}
        except Exception:
            pass

    # last resort: best local answer we have (never fail the task)
    return {"task_id": task_id, "answer": (samples[0].strip() if samples else ""),
            "route": "local-fallback", "category": category, "tokens": 0,
            "confidence": round(conf, 3)}
