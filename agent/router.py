"""The routing brain: answer free when we can trust it, escalate when we can't.

Local model inference is FREE (0 Fireworks tokens = best possible score), so we
answer as much as possible locally — but a small (2-3B) model is only reliable on
some categories. Per task:

  1. free deterministic solver (arithmetic, ordering, syllogism, …) — 0 tokens.
  2. HARD categories (math / logic / code) the small model gets wrong AND is slow
     on -> straight to Fireworks (skip the local attempt).
  3. otherwise answer locally (0 tokens); score confidence from a category prior,
     free verifier signals, and self-consistency; keep it if confident.
  4. low confidence (or near the wall-clock deadline) -> escalate to the cheapest
     Fireworks model with a tiny prompt.
"""
from __future__ import annotations

import re

from . import verifiers as V
from .classifier import HARD, classify
from .config import config
from .prompts import build_messages, build_remote_messages, build_retry_messages, max_tokens_for
from .solvers import free_solve

# Categories the local model handles reliably (short outputs, fast on CPU).
LOCAL_OK = {"sentiment", "summarization", "ner", "factual"}
# No cheap correctness verifier -> take two local draws; disagreement = unsure.
SELF_CONSISTENCY = {"factual"}
RETRY_CATEGORIES = {"ner", "summarization", "sentiment"}

# Base trust per category for a ~3B local model (measured on the practice set:
# reliable on sentiment/summary/ner, decent on factual, poor on math/logic/code).
PRIOR = {
    "sentiment": 0.82, "summarization": 0.72, "ner": 0.74, "factual": 0.56,
    "math": 0.28, "logic": 0.28, "code_debug": 0.33, "code_gen": 0.38,
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _confidence(category: str, prompt: str, samples: list[str]) -> float:
    ans = samples[0] if samples else ""
    c = PRIOR.get(category, 0.6)
    if len(samples) > 1:
        c += 0.20 if _norm(samples[0]) == _norm(samples[1]) else -0.30
    if category == "sentiment":
        c += 0.20 if V.label_ok(ans) else -0.50
    elif category == "ner":
        c += 0.15 if (V.valid_json(ans) or _looks_labeled(ans)) else -0.40
    elif category == "summarization":
        c += 0.10 if V.length_ok(prompt, ans) else -0.20
    elif category == "factual":
        c += -0.40 if not ans.strip() else 0.0
    return max(0.0, min(1.0, c))


def _looks_labeled(ans: str) -> bool:
    """NER output that labels entities either as JSON or as 'Name (Type)' pairs."""
    return bool(re.search(r"\(\s*(person|org|organization|location|date|time)\b", ans, re.I))


def _pick_remote_model(category: str) -> str:
    """Choose a model from the harness-injected ALLOWED_MODELS. Prefer a compact
    non-reasoning instruct model (fewest tokens) and a code model for code."""
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


def _fireworks(task_id, category, prompt, remote, *, full_prompt=False, conf=0.0) -> dict:
    """One minimal Fireworks call; returns the result dict with token count."""
    model = _pick_remote_model(category)
    builder = build_messages if full_prompt else build_remote_messages
    before = remote.meter.total
    try:
        out = remote.chat(model, builder(category, prompt),
                          max_tokens=max_tokens_for(category), temperature=0.0, n=1,
                          reasoning_effort=config.reasoning_effort)
        return {"task_id": task_id, "answer": out[0].strip(), "route": "remote",
                "category": category, "tokens": remote.meter.total - before,
                "confidence": round(conf, 3), "model": model}
    except Exception as e:
        return {"task_id": task_id, "answer": "", "route": "error", "category": category,
                "tokens": remote.meter.total - before, "confidence": round(conf, 3), "error": str(e)}


def route(task: dict, local, remote, prefer_remote: bool = False) -> dict:
    """Return {task_id, answer, route, category, tokens, confidence}.

    prefer_remote=True (set by main.py near the wall-clock deadline) skips slow
    local inference and escalates directly, so the run always finishes in time.
    """
    task_id = task.get("task_id")
    prompt = task.get("prompt", "")
    category = classify(prompt)

    # baseline mode (eval only): everything straight to Fireworks with full prompts
    if config.force_remote and config.has_remote():
        return _fireworks(task_id, category, prompt, remote, full_prompt=True)

    # 1) free deterministic solvers — 0 tokens, exact
    solved = free_solve(category, prompt)
    if solved is not None:
        return {"task_id": task_id, "answer": solved, "route": "local-solver",
                "category": category, "tokens": 0, "confidence": 1.0}

    have_local = bool(local) and config.use_local
    # 2) hard categories / near-deadline / no local -> Fireworks (skip slow local)
    if config.has_remote() and (prefer_remote or not have_local or category in HARD):
        return _fireworks(task_id, category, prompt, remote)

    # 3) local answer for the categories a small model handles well
    if have_local:
        messages = build_messages(category, prompt)
        n = config.local_samples_hard if category in SELF_CONSISTENCY else 1
        try:
            samples = local.chat(config.local_model_path, messages,
                                 max_tokens=max_tokens_for(category),
                                 temperature=0.0 if n == 1 else 0.4, n=n)
        except Exception:
            samples = []
        conf = _confidence(category, prompt, samples) if samples else 0.0

        if samples and conf >= config.escalate_threshold:
            return {"task_id": task_id, "answer": samples[0].strip(), "route": "local",
                    "category": category, "tokens": 0, "confidence": round(conf, 3)}

        # 3b) one free strict local retry before spending tokens (opt-in)
        if samples and config.local_retry and category in RETRY_CATEGORIES:
            try:
                retry = local.chat(config.local_model_path, build_retry_messages(category, prompt),
                                   max_tokens=max_tokens_for(category), temperature=0.0, n=1)
                if _confidence(category, prompt, retry) >= config.escalate_threshold:
                    return {"task_id": task_id, "answer": retry[0].strip(), "route": "local-retry",
                            "category": category, "tokens": 0,
                            "confidence": round(_confidence(category, prompt, retry), 3)}
            except Exception:
                pass

        # 4) escalate low-confidence local answer to Fireworks
        if config.has_remote():
            return _fireworks(task_id, category, prompt, remote, conf=conf)

        # 5) offline last resort: best local answer (never fail the task)
        return {"task_id": task_id, "answer": (samples[0].strip() if samples else ""),
                "route": "local-fallback", "category": category, "tokens": 0,
                "confidence": round(conf, 3)}

    # no local and no remote (shouldn't happen) -> empty, still valid
    return {"task_id": task_id, "answer": "", "route": "none", "category": category,
            "tokens": 0, "confidence": 0.0}
