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

import json
import re
import time as _time

from . import verifiers as V
from .backends import extract_final
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
# reliable on sentiment/summary/ner, poor on math/logic/code).
# factual sits at 0.55: a LONE draw stays below the 0.60 threshold (still
# escalates — same gate-safe behavior as before), but two AGREEING draws
# (self-consistency, LOCAL_SAMPLES_HARD=2) reach 0.75 and are kept for 0 tokens.
# Agreement is the hallucination guard the category otherwise lacks (no cheap
# correctness verifier); a disagreeing pair lands at 0.25 and escalates. Measured
# 100% local on dev factual — the biggest single token reclaim after the solvers.
PRIOR = {
    "sentiment": 0.82, "summarization": 0.72, "ner": 0.74, "factual": 0.55,
    "math": 0.28, "logic": 0.28, "code_debug": 0.33, "code_gen": 0.38,
}


def _norm(s: str) -> str:
    # trailing punctuation is stripped so "Paris" and "Paris." count as AGREEING
    # self-consistency draws — a false disagreement needlessly escalates (tokens).
    t = re.sub(r"\s+", " ", (s or "").strip().lower())
    return re.sub(r"[\s.!?]+$", "", t)


def _confidence(category: str, prompt: str, samples: list[str]) -> float:
    ans = samples[0] if samples else ""
    c = PRIOR.get(category, 0.6)
    if len(samples) > 1:
        c += 0.20 if _norm(samples[0]) == _norm(samples[1]) else -0.30
    if category == "sentiment":
        c += 0.20 if V.label_ok(ans) else -0.50
    elif category == "ner":
        c += 0.15 if (V.valid_ner_json(ans) or _looks_labeled(ans)) else -0.40
    elif category == "summarization":
        c += 0.10 if V.length_ok(prompt, ans) else -0.20
    elif category == "factual":
        c += -0.40 if not ans.strip() else 0.0
    return max(0.0, min(1.0, c))


def _looks_labeled(ans: str) -> bool:
    """NER output that labels entities either as JSON or as 'Name (Type)' pairs."""
    return bool(re.search(r"\(\s*(person|org|organization|location|date|time)\b", ans, re.I))


# Families ranked by MEASURED end-to-end accuracy on our 96-task stress set.
# minimax-m3 leads: 94.8% through this router (the only Track-1 allowed model we
# could validate — gemma-4 404s on the real API, so it has never been tested and
# must not outrank a measured model). kimi deprioritized (dumps reasoning into
# content; the extractor handles it, but clean-content models come first).
# NOTHING is excluded: a homogeneous grader list is still attempted.
_FAMILY_PREF = ("minimax", "gpt-oss", "gemma", "glm", "deepseek", "qwen", "llama", "mixtral", "phi")
_DEPRIORITIZE = ("kimi",)
_SHORT_CATEGORIES = {"sentiment", "math", "factual", "logic"}


def _salvage_strong(category: str, s: str) -> bool:
    """A reasoning-trace extraction confident enough to RETURN immediately (a model
    that empties `content` behaves the same on every candidate, so failing over just
    burns tokens for the same result). Only the deterministic categories qualify;
    code/summarization stay a FLOOR so we still try a possibly-clean-content model."""
    s = (s or "").strip()
    if not s:
        return False
    if category == "sentiment":
        return s.lower() in ("positive", "negative", "neutral")
    if category == "math":
        return bool(re.fullmatch(r"-?\d[\d,]*\.?\d*", s))
    if category == "ner":
        try:
            json.loads(s)
            return True
        except Exception:
            return False
    if category in ("factual", "logic"):
        return 0 < len(s) <= 120
    return False


def _candidate_models(category: str) -> list[str]:
    """Ordered, de-duplicated list of models to TRY (best first). The router fails
    over down this list when a model errors or truncates — the observed 26% was one
    bad model with no fallback. Capped so token/latency cost stays bounded."""
    models = list(dict.fromkeys(config.allowed_models))  # de-dup, keep order
    if not models:
        # has_remote() can be true off the API key alone (no model list injected):
        # fall back to the VERBATIM launch-day allow-list (config.fallback_models),
        # never a prefixed/invented id — one 404ing name = every escalation dead.
        models = list(dict.fromkeys(
            ([config.preferred_model] if config.preferred_model else [])
            + config.fallback_models))
    if not models:
        return []

    def rank(m: str):
        lm = m.lower()
        depr = any(d in lm for d in _DEPRIORITIZE)
        pref = next((i for i, f in enumerate(_FAMILY_PREF) if f in lm), len(_FAMILY_PREF))
        code_bias = 0 if (category in ("code_gen", "code_debug") and "code" in lm) else 1
        return (code_bias, 1 if depr else 0, pref, models.index(m))

    ordered = sorted(models, key=rank)
    # honor an explicit preference first, if it's actually allowed — except for
    # code tasks, where a code-specialist model (rank puts it first) wins.
    if (config.preferred_model and config.preferred_model in ordered
            and not (category in ("code_gen", "code_debug")
                     and "code" in ordered[0].lower())):
        ordered.remove(config.preferred_model)
        ordered.insert(0, config.preferred_model)
    # cap fan-out at 3, but force FAMILY DIVERSITY into the last slot: with an
    # allow-list of three gemma variants, a systemic gemma failure (template,
    # rate limit) would otherwise kill every candidate for the task.
    top = ordered[:2]
    fam = ordered[0].split("-")[0].lower() if ordered else ""
    diverse = next((m for m in ordered[2:] if m.split("-")[0].lower() != fam), None)
    if diverse:
        top.append(diverse)
    elif len(ordered) > 2:
        top.append(ordered[2])
    # Send each pick EXACTLY as the harness injected it — nothing else. The judging
    # proxy matches ALLOWED_MODELS entries VERBATIM; ANY other model string (a bare/
    # prefixed re-spelling of an allowed id, or an always-on model not on the list)
    # makes the WHOLE submission a MODEL_VIOLATION and unscoreable. That is strictly
    # worse than a task whose model 404s and falls back. So: no id-format toggling,
    # no serverless safety net — only verbatim allow-list entries. De-dup only.
    return list(dict.fromkeys(m for m in top if m))


def _fireworks(task_id, category, prompt, remote, *, full_prompt=False, conf=0.0,
               deadline: float | None = None, local_fallback: str = "") -> dict:
    """Escalate to Fireworks, failing over across candidate models until one returns
    a usable answer. A model that errors (5xx/error-body), times out, or truncates a
    short answer (finish_reason=length) is abandoned for the next candidate. Bounded
    by a per-task wall-clock `deadline` so fallback can't blow the <30s/task limit.

    `local_fallback` is the answer the local model already produced for this task (if
    any). If EVERY remote candidate fails/returns empty — the exact symptom when the
    grader's Fireworks access is dead (no credits / blocked / all 4xx) — we return the
    local answer instead of an empty string. An empty answer is 0 credit (definitely
    wrong); the local answer is sometimes right. Never discard it for an empty remote."""
    builder = build_messages if full_prompt else build_remote_messages
    messages = builder(category, prompt)  # per-model variant built inside the loop
    max_tok = max_tokens_for(category)
    candidates = _candidate_models(category)
    before = remote.meter.total
    last = {"answer": "", "model": "", "error": "no candidates"}

    def _time_left():
        if deadline is None:
            return None
        return deadline - _time.time()

    for model in candidates:
        if not model:
            continue
        # Per model: try at the normal ceiling; on truncation (finish=length — a
        # reasoning-style model burned the budget on its trace before the answer),
        # retry the SAME model ONCE at a higher ceiling before failing over. This
        # now applies to EVERY category: gateway simulation showed the old
        # short-categories-only rule accepting trace-truncated garbage for
        # summarization/ner/code. Ceiling capped so a retry stays <30s/request.
        for mt in (max_tok, min(max_tok * 3, 1536)):
            rem = _time_left()
            if rem is not None and rem <= 4.0:  # not enough time for another attempt
                break
            call_timeout = min(rem, config.request_timeout) if rem else None
            try:
                # prompt is MODEL-AWARE: a no-reasoning-channel family (gemma…) gets
                # a CoT prompt for math/logic — "final answer only" forbids the
                # thinking those tasks need on a plain instruct model.
                msgs = messages if full_prompt else build_remote_messages(category, prompt, model)
                out = remote.chat(model, msgs, max_tokens=mt, temperature=0.0, n=1,
                                  reasoning_effort=config.reasoning_effort, timeout=call_timeout)
                ans = (out[0].get("text") or "").strip()
                if ans and category in ("math", "logic") and "FINAL:" in ans:
                    # CoT answer: keep only the marked final line (the reasoning
                    # stays out of the submitted answer)
                    tail = ans.rsplit("FINAL:", 1)[-1].strip()
                    if tail:
                        ans = tail.splitlines()[0].strip()
                finish = out[0].get("finish")
                reasoning = out[0].get("reasoning") or ""
                truncated = finish == "length"
                if ans and not truncated:  # good clean answer — done
                    return {"task_id": task_id, "answer": ans, "route": "remote",
                            "category": category, "tokens": remote.meter.total - before,
                            "confidence": round(conf, 3), "model": model}
                # Empty/weak content: the answer is in the reasoning trace (reasoning
                # model). Extract it PER-CATEGORY (a category-blind grab was the
                # sentiment/summarization/code gate failure). BUT a trace cut off by
                # finish=length is MID-COMPUTATION — extraction grabs a plausible-
                # looking fragment ("...then take 8% of" -> '8'), so on a first-try
                # truncation we fall through to the higher-ceiling retry (written for
                # exactly this case) instead of returning the fragment immediately;
                # the fragment is kept below only as a better-than-empty floor.
                salvage = (extract_final(category, reasoning).strip() if reasoning
                           else (out[0].get("salvage") or "").strip())
                if (not ans and salvage and _salvage_strong(category, salvage)
                        and not (truncated and mt == max_tok)):
                    return {"task_id": task_id, "answer": salvage, "route": "remote-reasoning",
                            "category": category, "tokens": remote.meter.total - before,
                            "confidence": round(conf, 3), "model": model}
                if ans and not last["answer"]:  # truncated partial: floor only
                    last = {"answer": ans, "model": model, "error": f"weak({finish})"}
                elif salvage and not last["answer"]:  # trace-extracted floor
                    last = {"answer": salvage, "model": model, "error": "salvaged"}
                if truncated and mt == max_tok:
                    continue  # retry SAME model at the higher ceiling
                break  # empty, or already retried high -> fail over to next model
            except Exception as e:
                last = {"answer": last["answer"], "model": model, "error": str(e)[:140]}
                break  # transport/model error -> next candidate model

    # Every candidate failed/weak. Prefer any partial remote answer; else fall back to
    # the local answer we already had (escalation must NEVER discard a non-empty local
    # answer — if the grader's Fireworks is down, an empty here scores 0, strictly
    # worse than the local model's answer). Only a genuinely empty result -> error.
    answer = last["answer"] or local_fallback
    route = "remote" if last["answer"] else ("local-fallback" if local_fallback else "error")
    return {"task_id": task_id, "answer": answer,
            "route": route, "category": category,
            "tokens": remote.meter.total - before, "confidence": round(conf, 3),
            "model": last["model"], "error": None if answer else last["error"]}


def _local_rescue(task_id, category, prompt, local, deadline) -> dict | None:
    """Last-resort LOCAL answer after every remote candidate failed (dead gateway).

    Only runs when enough per-task budget remains — a dead gateway fails fast
    (1-3s of the 28s budget), leaving room; a slow-timeout failure doesn't, and
    we skip rather than blow the <30s/task limit. Output is capped small for CPU
    speed. A short answer that's sometimes right strictly beats the empty answer
    (always wrong) we'd otherwise emit. Free: the failed calls metered ~nothing."""
    remaining = deadline - _time.time()
    if remaining < 10.0:
        return None
    try:
        samples = local.chat(config.local_model_path, build_messages(category, prompt),
                             max_tokens=min(max_tokens_for(category), 256),
                             temperature=0.0, n=1)
        ans = (samples[0] or "").strip() if samples else ""
    except Exception:
        return None
    if not ans:
        return None
    return {"task_id": task_id, "answer": ans, "route": "local-rescue",
            "category": category, "tokens": 0,
            "confidence": round(PRIOR.get(category, 0.5), 3)}


def route(task: dict, local, remote, prefer_remote: bool = False) -> dict:
    """Return {task_id, answer, route, category, tokens, confidence}.

    prefer_remote=True (set by main.py near the wall-clock deadline) skips slow
    local inference and escalates directly, so the run always finishes in time.
    """
    task_id = task.get("task_id")
    prompt = task.get("prompt", "")
    category = classify(prompt)
    # per-task wall-clock budget for the (possibly multi-model) Fireworks fallback,
    # so trying alternate models can never blow the <30s/task limit.
    deadline = _time.time() + config.per_task_budget_s

    # baseline mode (eval only): everything straight to Fireworks with full prompts
    if config.force_remote and config.has_remote():
        return _fireworks(task_id, category, prompt, remote, full_prompt=True, deadline=deadline)

    # 1) free deterministic solvers — 0 tokens, exact.
    # DIAGNOSTIC: config.disable_solvers forces EVERY task through the model (only
    # when a real API key is present, so the offline self-test still passes). Used
    # to isolate whether remote calls work at all in the grader: score ~0% => every
    # model call is failing; high score => the model path works. Flip back off after.
    skip_solvers = config.disable_solvers and bool(config.fireworks_api_key)
    solved = None if skip_solvers else free_solve(category, prompt)
    if solved is not None:
        return {"task_id": task_id, "answer": solved, "route": "local-solver",
                "category": category, "tokens": 0, "confidence": 1.0}

    have_local = bool(local) and config.use_local
    # LOCAL_ONLY: the zero-token mode — never touch Fireworks; solvers + the local
    # model answer everything (0 tokens is the unbeatable floor of an ascending-
    # token leaderboard). Only honored when the local model actually loaded, so a
    # bad flag can never strand every task with no answerer at all.
    remote_ok = config.has_remote() and not (config.local_only and have_local)
    # 2) REMOTE-FIRST (default) or hard/near-deadline/no-local/long -> Fireworks.
    # Remote-first: every non-solver task goes to the gateway model — the profile
    # all four gate-passing leaderboard agents run. The local tier's format-only
    # confidence gates kept wrong-but-well-formed answers (sentiment at conf 1.0
    # with the WRONG label), which is what failed the gate at 26.3%; local is now
    # a dead-remote rescue only. A long prompt also skips local (slow CPU prefill
    # on 2 vCPU risks the <30s/task limit).
    too_long = len(prompt) > config.local_max_prompt_chars
    if remote_ok and (config.remote_first or prefer_remote or not have_local
                      or category in HARD or too_long):
        r = _fireworks(task_id, category, prompt, remote, deadline=deadline)
        # Dead-remote rescue: every candidate failed with nothing to show (the
        # grader's Fireworks access being down does exactly this) -> a local answer
        # strictly beats the empty one we'd otherwise emit, and costs 0 tokens.
        if r["route"] == "error" and have_local:
            rescue = _local_rescue(task_id, category, prompt, local, deadline)
            if rescue is not None:
                return rescue
        return r

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

        # 4) escalate low-confidence local answer to Fireworks — but pass the local
        # answer as a fallback so a dead-remote grader can't turn a usable local
        # answer into an empty (0-credit) one. Skipped entirely in LOCAL_ONLY mode.
        if remote_ok:
            return _fireworks(task_id, category, prompt, remote, conf=conf, deadline=deadline,
                              local_fallback=(samples[0].strip() if samples else ""))

        # 5) offline last resort: best local answer (never fail the task)
        return {"task_id": task_id, "answer": (samples[0].strip() if samples else ""),
                "route": "local-fallback", "category": category, "tokens": 0,
                "confidence": round(conf, 3)}

    # no local and no remote (shouldn't happen) -> empty, still valid
    return {"task_id": task_id, "answer": "", "route": "none", "category": category,
            "tokens": 0, "confidence": 0.0}
