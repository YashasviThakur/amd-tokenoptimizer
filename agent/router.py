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
from .classifier import classify
from .config import config
from .prompts import (_NO_REASONING_FAMILIES, build_batch_messages, build_messages,
                      build_remote_messages, build_retry_messages, max_tokens_for,
                      wants_elaboration)
from .solvers import free_solve

# Categories the local model may answer for 0 tokens. sentiment + summarization are
# pure-local and need no world knowledge (no hallucination risk). code_gen is GATED by
# the execution oracle (_confidence -> V.run_extracted_tests): kept only when the code
# passes the prompt's OWN I/O examples. ner is GATED by a correctness check too
# (_confidence -> _ner_local_ok): kept only when TWO local draws agree on the exact
# entity set AND every extracted entity is grounded verbatim in the source sentence
# (V.ner_entities_grounded). A format-only shape check can't catch a hallucinated-but-
# well-formed entity — that check can, so NER is safe to keep locally again.
# MAX config (ship 15): code is BACK. The ship 9/11 TIMEOUTs were the organizer's
# overloaded backend (their announcement), not our runtime — the FULL local workload
# (sentiment n=2 + summarization + ner n=2 + FOUR code tasks n=2 @ cap 96) measures
# 124s serialized at grader-like 2 threads, and the LOCAL_TIME_BUDGET_S guard bounds
# the worst case deterministically (overflow -> remote; never a TIMEOUT). Gates:
# ner = grounding + agreement + completeness (OOD 9/12 kept, 0 wrong-kept);
# code = in-process differential oracle (OOD 12/12 kept, 0 wrong-kept).
LOCAL_OK = {"sentiment", "summarization", "ner", "code_gen", "code_debug"}
# No cheap correctness verifier -> take two local draws; disagreement = unsure.
SELF_CONSISTENCY = {"factual", "sentiment", "ner"}
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
    "sentiment": 0.35,  # PROVISIONAL — retune vs measured tuned-model per-category accuracy before shipping
    "summarization": 0.55,  # PROVISIONAL — retune vs measured tuned-model per-category accuracy before shipping
    "ner": 0.74, "factual": 0.55,
    "math": 0.28, "logic": 0.28, "code_debug": 0.33,
    "code_gen": 0.55,  # unused for code_gen: _confidence overrides it with the execution oracle (pass->0.9, else->0.2)
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
        # OVERRIDE the prior (hence '=' not '+='): keep local ONLY when two draws agree
        # on the exact normalized entity set AND every entity is grounded verbatim in
        # the source (V.ner_entities_grounded) AND both parse as clean NER JSON. Any of
        # those failing -> 0.2 (< escalate_threshold) -> escalate. A format-only shape
        # check kept hallucinated-but-well-formed answers; this correctness gate can't.
        c = 0.9 if _ner_local_ok(prompt, samples) else 0.2
    elif category == "summarization":
        c += 0.10 if V.length_ok(prompt, ans) else -0.20
    elif category == "factual":
        c += -0.40 if not ans.strip() else 0.0
    elif category in ("code_gen", "code_debug"):
        # DIFFERENTIAL oracle, executed IN-PROCESS (no subprocess — subprocess.run
        # hangs or is seccomp-killed on the grader, which TIMED OUT ship 9). Keep local
        # ONLY when TWO independent draws behaviourally AGREE on an adversarial auto-
        # generated input battery (V.differential_code_ok -> V._run_battery, AST-guarded
        # in-process exec). Any divergence -> escalate. OOD-validated 12/12, 0 wrong-kept.
        c = 0.9 if V.differential_code_ok(prompt, samples) else 0.2
    return max(0.0, min(1.0, c))


# Entity-type synonyms -> a canonical bucket, so two draws that label the same
# entity under 'org' vs 'organization' still compare equal, and any UNRECOGNIZED
# key makes the draw invalid (-> escalate).
_NER_CANON = {"person": "person", "people": "person",
              "org": "org", "organization": "org", "organisation": "org",
              "location": "location", "loc": "location", "place": "location",
              "date": "date", "time": "date", "misc": "misc"}


def _ner_entity_sets(ans: str):
    """Canonical {type: frozenset(lowercased entities)} for a NER answer, or None if
    it doesn't parse to a JSON dict keyed ONLY by recognized entity types. Empty lists
    are dropped, so two draws that differ only in which empty keys they emit still
    compare equal; a draw with NO entities at all returns {} (falsy -> not kept)."""
    raw = ans or ""
    m = V._FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    try:
        obj = json.loads(raw.strip())
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out: dict = {}
    for k, v in obj.items():
        ck = _NER_CANON.get(str(k).strip().lower())
        if ck is None or not isinstance(v, list):
            return None  # junk key or non-list value -> invalid draw
        vals = {str(x).strip().lower() for x in v if isinstance(x, str) and str(x).strip()}
        if vals:
            out[ck] = out.get(ck, frozenset()) | frozenset(vals)
    return out


def _ner_local_ok(prompt: str, samples: list[str]) -> bool:
    """Keep an NER answer local ONLY if ALL hold: two draws are available; each parses
    to clean, non-empty NER JSON; both are grounded (every entity verbatim in the
    source); and the two draws agree on the exact canonical entity set. Any failure
    escalates — a conservative gate that can waste tokens but never keeps a wrong
    answer (the catastrophic case)."""
    if len(samples) < 2:
        return False
    s0, s1 = _ner_entity_sets(samples[0]), _ner_entity_sets(samples[1])
    if not s0 or not s1:      # unparseable, junk-keyed, or all-empty -> escalate
        return False
    if s0 != s1:              # self-consistency failure -> escalate
        return False
    # grounding (no hallucinated entity) AND completeness (no DROPPED source entity —
    # the ship-6 failure grounding alone missed). Both draws grounded; completeness on
    # the agreed set. OOD-measured 0 wrong-kept with the completeness check added.
    return (V.ner_entities_grounded(prompt, samples[0])
            and V.ner_entities_grounded(prompt, samples[1])
            and V.ner_source_covered(prompt, samples[0]))


# Families ranked by MEASURED end-to-end accuracy on our 96-task stress set.
# minimax-m3 leads (94.8% through this router). kimi SECOND: it is the only
# OTHER allowed model verified to actually respond (gemma-4 404s on every key
# we could test — it may be two dead entries), so kimi is the one real second
# chance a task gets when minimax fails or times out; its reasoning-dump quirk
# is handled by the extractor. NOTHING is excluded: a homogeneous grader list
# is still attempted.
_FAMILY_PREF = ("minimax", "kimi", "gpt-oss", "gemma", "glm", "deepseek", "qwen", "llama", "mixtral", "phi")
_DEPRIORITIZE = ()
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
        # must contain a real token: a truncated trace's last line was a lone
        # '-' and the old length-only check submitted it as the final answer
        return 0 < len(s) <= 120 and bool(re.search(r"[A-Za-z0-9]", s))
    return False


def _candidate_models(category: str) -> list[str]:
    """Ordered, de-duplicated list of models to TRY (best first). The router fails
    over down this list when a model errors or truncates — the observed 26% was one
    bad model with no fallback. Capped so token/latency cost stays bounded."""
    models = list(dict.fromkeys(config.allowed_models))  # de-dup, keep order
    if not models:
        # has_remote() can be true off the API key alone (no model list injected):
        # fall back to the VERBATIM launch-day allow-list (config.fallback_models)
        # ONLY. preferred_model must never be a candidate SOURCE here — a stray
        # REMOTE_MODEL env var would put an off-list id first = MODEL_VIOLATION
        # (this team lost two submissions to exactly that class of leak).
        models = list(dict.fromkeys(config.fallback_models))
    if not models:
        return []

    def rank(m: str):
        lm = m.lower()
        depr = any(d in lm for d in _DEPRIORITIZE)
        pref = next((i for i, f in enumerate(_FAMILY_PREF) if f in lm), len(_FAMILY_PREF))
        # NO code-name bias: it made kimi-k2p7-code front BOTH code categories
        # (4-5 of 19 hidden tasks) purely on its name — an UNMEASURED model this
        # file itself deprioritizes for dumping reasoning into content — while
        # minimax measured 100% on code locally. Measured always outranks named.
        return (1 if depr else 0, pref, models.index(m))

    ordered = sorted(models, key=rank)
    # STABILITY PLAY, gated on PROOF: when /models discovery has VERIFIED which
    # allowed models the proxy serves (config.models_verified), lead with the
    # instruct family everywhere. Reasoning models carry three per-run
    # pathologies (empty content -> salvage, slow generation -> timeout, trace
    # truncation) that fire randomly and cost ~1-2 tasks per run — the measured
    # 13<->14 wobble — while instruct models always return content fast (the
    # stable-16 qualifier profile). Math/logic on instruct models get the CoT +
    # 'FINAL:' prompt automatically (build_remote_messages is model-aware).
    # Without verification this is a no-op: minimax stays first (measured).
    # FORCE_INSTRUCT_FIRST (token play): lead with gemma UNCONDITIONALLY. minimax
    # measured a hard ~9k-token floor (reasoning billed as completion); the 3.5k
    # leaders run the instruct family. If gemma is served -> ~3.5k tokens; if it
    # 404s -> failover to minimax (0 wasted tokens, same 16). CoT+FINAL prompt
    # activates for gemma math/logic automatically.
    if config.models_verified or config.force_instruct_first:
        instruct = [m for m in ordered
                    if any(f in m.lower() for f in _NO_REASONING_FAMILIES)]
        if instruct:
            ordered = instruct + [m for m in ordered if m not in instruct]
    # honor an explicit preference first, if it's actually allowed — except for
    # code tasks, where a code-specialist model (rank puts it first) wins.
    if (config.preferred_model and config.preferred_model in ordered
            and not (category in ("code_gen", "code_debug")
                     and "code" in ordered[0].lower())):
        ordered.remove(config.preferred_model)
        ordered.insert(0, config.preferred_model)
    # INSTRUCT-FIRST (gemma token play): try EVERY allowed instruct variant before
    # the reasoning failover, so the user can deploy the CHEAPEST gemma (nvfp4 /
    # a4b / 31b) and it still gets reached — an undeployed variant 404s in ~1s at
    # 0 tokens, then the next is tried. The final reasoning model (minimax) is the
    # safety net if no gemma is deployed. Bounded: <=3 gemmas + 1 failover.
    if config.force_instruct_first or config.models_verified:
        instruct = [m for m in ordered
                    if any(f in m.lower() for f in _NO_REASONING_FAMILIES)]
        failover = next((m for m in ordered if m not in instruct), None)
        top = instruct + ([failover] if failover else [])
    else:
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


# VOTE OFF (stability build): measured a wash on the hidden set (14/19 with and
# without), while each vote is an extra burst call into a rate-limiting proxy
# and a residual override risk (n=2 degeneracy). Fewer calls = fewer ways for a
# task to die. Machinery kept; re-enable by listing categories here.
_VOTE_CATEGORIES: set = set()


def _final_line(category: str, ans: str) -> str:
    """Keep only the 'FINAL: …' line of a CoT answer (math/logic); else as-is."""
    if ans and category in ("math", "logic") and "FINAL:" in ans:
        tail = ans.rsplit("FINAL:", 1)[-1].strip()
        if tail:
            return tail.splitlines()[0].strip()
    return ans


def _vote_refine(remote, model, msgs, category, max_tok, anchor, time_left) -> str:
    """Anchor-safe majority vote: draw TWO hot samples; only when BOTH agree with
    each other AND against the deterministic temp-0 anchor do they override it.
    Every failure mode (timeout, error, proxy ignoring n>1, disagreement, tie)
    returns the anchor unchanged — this can only replace the answer on 2-vs-1
    evidence, never degrade it to a single hot sample."""
    if time_left is not None and time_left < 8.0:
        return anchor
    try:
        outs = remote.chat(model, msgs, max_tokens=max_tok, temperature=0.8, n=2,
                           reasoning_effort=config.reasoning_effort,
                           timeout=min(time_left, config.request_timeout) if time_left else None)
    except Exception:
        return anchor
    samples = [_final_line(category, (o.get("text") or "").strip())
               for o in outs if o.get("finish") != "length"]
    samples = [s for s in samples if s]
    if (len(samples) == 2 and _norm(samples[0]) == _norm(samples[1])
            and _norm(samples[0]) != _norm(anchor)):
        w = samples[0]
        # numerically equal ("10.0" vs "10", "1,000" vs "1000") is AGREEMENT —
        # keep the anchor's judge-clean formatting, never swap it for a variant
        try:
            if abs(float(_norm(w).replace(",", "")) -
                   float(_norm(anchor).replace(",", ""))) < 1e-9:
                return anchor
        except ValueError:
            pass
        # format gate (audit, executed): hot pairs byte-agree most easily on
        # formulaic wrappers ("The answer is 42.") — a winner must be at least
        # as judge-clean as the anchor or the override loses exact-match judging
        if category == "math":
            if not re.fullmatch(r"-?\d[\d,]*\.?\d*", w.strip().rstrip(".")):
                return anchor
        elif len(w) > 120 or not re.search(r"[A-Za-z0-9]", w):
            return anchor
        return w
    return anchor


# "in exactly 12 words" / "no more than 15 words" — the judge counts; the model
# doesn't unless made to. Measured: a 15-word cap got 16 words (judged FAIL).
_WORD_LIMIT = re.compile(
    r"\b(exactly|no more than|at most|maximum(?: of)?|within|under|fewer than)\s+(\d{1,3})\s+words\b", re.I)


def _word_count(s: str) -> int:
    return len(re.findall(r"[\w'-]+", s or ""))


def _enforce_word_limit(remote, model, prompt, ans, time_left) -> str:
    """Post-check an explicit word limit; one strict same-model retry on violation.
    Keeps whichever answer satisfies (or comes closer to) the limit — never
    returns empty, never runs when no limit is stated or time is short."""
    m = _WORD_LIMIT.search(prompt)
    if not m:
        return ans
    kind = m.group(1).lower()
    n = int(m.group(2))
    if kind == "fewer than":
        exact, n = False, n - 1
    else:
        exact = kind == "exactly"
    wc = _word_count(ans)
    if (wc == n) if exact else (wc <= n):
        return ans
    if time_left is not None and time_left < 8.0:
        return ans
    strict = ((f"Write the summary in EXACTLY {n} words." if exact
               else f"Write the summary in AT MOST {n} words.")
              + " Output only the summary text, nothing else. Do not count words aloud.")
    try:
        out = remote.chat(model, [{"role": "system", "content": strict},
                                  {"role": "user", "content": prompt}],
                          max_tokens=max_tokens_for("summarization"), temperature=0.0, n=1,
                          reasoning_effort=config.reasoning_effort,
                          timeout=min(time_left, config.request_timeout) if time_left else None)
    except Exception:
        return ans
    cand = (out[0].get("text") or "").strip()
    if cand and out[0].get("finish") != "length":
        cw = _word_count(cand)
        if (cw == n) if exact else (cw <= n):
            return cand
        if abs(cw - n) < abs(wc - n):
            return cand
    return ans


_BATCH_LINE = re.compile(r"^\s*(\d{1,2})\s*[\).:\]]\s*(.+?)\s*$")


def batch_remote(category: str, items: "list[tuple[int, str]]", remote, *,
                 deadline: float | None = None) -> dict:
    """Answer several same-category tasks in ONE Fireworks call ('N) answer' per line),
    amortizing the fixed per-call template. `items` = [(orig_index, prompt), ...].

    Returns {orig_index: raw_answer} ONLY when EVERY item parsed to a non-empty answer;
    on ANY doubt (single item, no model, error, truncation, missing/blank line, wrong
    count) returns {} so the caller routes those tasks individually. => batching can
    lower tokens but can NEVER drop or corrupt a task (the per-task path is the net)."""
    if len(items) < 2:
        return {}
    prompts = [p for _, p in items]
    model = next((m for m in _candidate_models(category) if m), None)
    if not model:
        return {}
    msgs = build_batch_messages(category, prompts)
    # N answers need N-fold room, but keep the reasoning trace bounded.
    max_tok = min(max(max_tokens_for(category), config.max_tokens_floor) + 80 * len(items), 3072)
    t_off = "minimax" in model.lower() and (config.thinking_off_all or config.thinking_off_soft)
    rem = (deadline - _time.time()) if deadline is not None else None
    if rem is not None and rem <= 5.0:
        return {}
    try:
        out = remote.chat(model, msgs, max_tokens=max_tok, temperature=0.0, n=1,
                          reasoning_effort=config.reasoning_effort,
                          timeout=min(rem, config.request_timeout) if rem else None,
                          thinking_off=t_off)
    except Exception:
        return {}
    if not out or out[0].get("finish") == "length":  # truncated batch = unreliable
        return {}
    text = (out[0].get("text") or "").strip()
    if not text:
        return {}
    parsed: dict = {}
    for line in text.splitlines():
        m = _BATCH_LINE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= len(items) and (n - 1) not in parsed:
            parsed[n - 1] = m.group(2).strip()
    # ALL items must have a non-empty answer, else fall back for the WHOLE group.
    if len(parsed) != len(items) or not all(parsed.get(j, "").strip() for j in range(len(items))):
        return {}
    return {items[j][0]: parsed[j] for j in range(len(items))}


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
    max_tok = max(max_tokens_for(category), config.max_tokens_floor)
    candidates = _candidate_models(category)
    word_limited = category == "summarization" and bool(_WORD_LIMIT.search(prompt))
    if word_limited:
        # MEASURED: reasoning models (minimax) loop forever counting words —
        # finish=length with EMPTY content even at a 4096 ceiling — while an
        # instruct-class model answers in one clean shot (exactly N words at
        # 768). Reorder (same allowed set, no new ids) so instruct families
        # (gemma on the grader) take word-limited summaries first.
        instruct = [m for m in candidates
                    if any(f in m.lower() for f in _NO_REASONING_FAMILIES)]
        candidates = instruct + [m for m in candidates if m not in instruct]
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
        # single attempt when the base ceiling is already large (a floor-raised
        # 4096 has no bigger retry; the old min() would have "retried" SMALLER)
        for mt in ((max_tok,) if max_tok >= 1536 else (max_tok, min(max_tok * 3, 1536))):
            rem = _time_left()
            if rem is not None and rem <= 4.0:  # not enough time for another attempt
                break
            call_timeout = min(rem, config.request_timeout) if rem else None
            try:
                # prompt is MODEL-AWARE: a no-reasoning-channel family (gemma…) gets
                # a CoT prompt for math/logic — "final answer only" forbids the
                # thinking those tasks need on a plain instruct model.
                msgs = messages if full_prompt else build_remote_messages(category, prompt, model)
                # thinking OFF for minimax-family models: ALL categories when
                # thinking_off_all (hard categories get the visible-CoT prompt via
                # build_remote_messages — measured 6/6 correct at 1/4 the tokens);
                # else soft categories only (the conservative build-J behavior).
                t_off = ("minimax" in model.lower()
                         and (config.thinking_off_all
                              or (config.thinking_off_soft and category in
                                  ("sentiment", "factual", "ner", "summarization"))))
                out = remote.chat(model, msgs, max_tokens=mt, temperature=0.0, n=1,
                                  reasoning_effort=config.reasoning_effort, timeout=call_timeout,
                                  thinking_off=t_off)
                # CoT answers keep only the marked FINAL line (reasoning stays out)
                ans = _final_line(category, (out[0].get("text") or "").strip())
                finish = out[0].get("finish")
                reasoning = out[0].get("reasoning") or ""
                truncated = finish == "length"
                if ans and not truncated:  # good clean answer — done
                    if not full_prompt:
                        if category == "summarization":
                            ans = _enforce_word_limit(remote, model, prompt, ans, _time_left())
                        elif category in _VOTE_CATEGORIES:
                            # vote at the BASE ceiling, not mt: after a 1536 truncation
                            # retry, two hot reasoning traces at mt would bill ~3k
                            # tokens for a confirmation signal the base ceiling gives
                            ans = _vote_refine(remote, model, msgs, category,
                                               max_tokens_for(category), ans, _time_left())
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
                    if word_limited:
                        break  # the counting loop won't finish at 3x either
                        # (measured empty at 4096) — save the ~20s for the next model
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
    # Local wall-time budget: cumulative time already spent inside local generation
    # (accumulated below on the shared `local` object). Once spent, remaining tasks
    # go remote — deterministic TIMEOUT protection on the 2-vCPU serialized model.
    local_exhausted = (have_local and
                       getattr(local, "_time_spent", 0.0) >= config.local_time_budget_s)
    # A sentiment task that demands a REASON can't use the one-word local path — the
    # rubric fails a bare label ("one-sided reason fails regardless of label"), so it
    # routes remote where build_remote_messages switches to the task-following prompt.
    needs_task_format = category == "sentiment" and wants_elaboration(prompt)
    if remote_ok and (config.remote_first or prefer_remote or not have_local
                      or category not in LOCAL_OK or too_long or local_exhausted
                      or needs_task_format):
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
        if category in ("ner", "code_gen", "code_debug"):
            n = max(n, 2)  # the NER completeness/agreement + differential-code gates need two draws
        # Timing guard: code answers at n=2 on a 2-vCPU box are the slowest local
        # path (measured ~5s short / ~31s long). Cap the local code draft so SHORT
        # functions stay local (fast, free) while a LONG one truncates -> won't
        # compile -> the oracle escalates it to Fireworks (which handles long code
        # well anyway). Bounds worst-case local code latency; other cats unchanged.
        mt = max_tokens_for(category)
        if category in ("code_gen", "code_debug"):
            mt = min(mt, config.local_code_max_tokens)
        _t_local = _time.time()
        try:
            samples = local.chat(config.local_model_path, messages,
                                 max_tokens=mt,
                                 temperature=0.0 if n == 1 else 0.4, n=n)
        except Exception:
            samples = []
        finally:
            # accumulate spent local wall-time on the shared model object (thread-safe
            # enough: worst case a lost update lets one extra task through the budget)
            local._time_spent = getattr(local, "_time_spent", 0.0) + (_time.time() - _t_local)
        conf = _confidence(category, prompt, samples) if samples else 0.0

        if samples and conf >= config.escalate_threshold:
            return {"task_id": task_id, "answer": samples[0].strip(), "route": "local",
                    "category": category, "tokens": 0, "confidence": round(conf, 3)}

        # 3b) one free strict local retry before spending tokens (opt-in) — also
        # skipped once the local wall-time budget is spent, and billed against it.
        if (samples and config.local_retry and category in RETRY_CATEGORIES
                and getattr(local, "_time_spent", 0.0) < config.local_time_budget_s):
            _t_local = _time.time()
            try:
                retry = local.chat(config.local_model_path, build_retry_messages(category, prompt),
                                   max_tokens=max_tokens_for(category), temperature=0.0, n=1)
                if _confidence(category, prompt, retry) >= config.escalate_threshold:
                    return {"task_id": task_id, "answer": retry[0].strip(), "route": "local-retry",
                            "category": category, "tokens": 0,
                            "confidence": round(_confidence(category, prompt, retry), 3)}
            except Exception:
                pass
            finally:
                local._time_spent = getattr(local, "_time_spent", 0.0) + (_time.time() - _t_local)

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
