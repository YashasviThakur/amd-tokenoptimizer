"""Token-minimal prompts, per category.

Every remote token counts, so system prompts are terse, outputs are constrained
to exactly what the judge needs, and max_tokens is capped hard per category.
"""
from __future__ import annotations

# system prompt + max_tokens CEILING per category. These are truncation guards,
# not targets — a model stops when it's done, so a high ceiling costs nothing on
# a short answer but prevents a *reasoning* model from having its answer starved
# (the reasoning trace consumes budget before the final answer is emitted).
# max_tokens is a CEILING (truncation guard), not a target — a model stops when
# done, so a high ceiling costs nothing on a short answer. But the models the
# harness allows are *reasoning* models: they emit a reasoning trace (billed as
# completion tokens) BEFORE the final answer. At 128 the trace fills the whole
# budget and the answer is truncated away (finish_reason=length) -> judged wrong,
# which is exactly what failed the accuracy gate at 15.8%. These ceilings give the
# reasoning room to finish; cheap models (gpt-oss) still stop in ~25-70 tokens.
# reasoning_effort now defaults OFF (see config.py) — without it trimming the
# trace, reasoning length is less predictable, so short-answer categories need
# real headroom too (sentiment was measured truncating at 256). 512 uniformly
# for short answers; only a live-measured 84% NovaAI-level competitor needs
# these numbers cut further, and that's a token-optimization pass, not a gate one.
POLICY = {
    "factual":       ("Answer correctly and concisely. Output only the answer, no preamble.", 512),
    "math":          ("Solve the problem. Output only the final numeric answer, nothing else.", 512),
    "sentiment":     ("Classify the sentiment. Reply with exactly one word: positive, negative, or neutral.", 512),
    "summarization": ("Summarize as instructed, honoring any length constraint. Output only the summary.", 512),
    "ner":           ('Extract named entities. Output ONLY minified JSON with keys '
                      '"person","org","location","date" (each a list of strings).', 512),
    "code_debug":    ("Fix the bug. Output only the corrected code, no explanation.", 896),
    "logic":         ("Solve the puzzle. Reason internally, then output only the final answer.", 512),
    "code_gen":      ("Write the function to spec. Output only the code, no explanation.", 896),
}
DEFAULT = ("Answer correctly and concisely. Output only the answer.", 512)


def system_for(category: str) -> str:
    return POLICY.get(category, DEFAULT)[0]


def max_tokens_for(category: str) -> int:
    return POLICY.get(category, DEFAULT)[1]


def build_messages(category: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_for(category)},
        {"role": "user", "content": prompt},
    ]


# Short system prompts for *remote* calls. These are reasoning models, so the
# instruction must be explicit ("give ONLY the final answer") — terse cues like
# "Number only." made the model dump its reasoning into the answer field instead
# of a separate channel, which then truncated. "final answer" wording keeps the
# reasoning out of the content and the answer clean. Every prompt token is score,
# so they stay as short as possible while still forcing clean output.
REMOTE_SYSTEM = {
    "factual": "Give ONLY the final answer, no explanation.",
    "math": "Give ONLY the final numeric answer, nothing else.",
    "sentiment": "Reply with ONLY one word: positive, negative, or neutral.",
    "summarization": "Output ONLY the summary, no preamble.",
    "ner": 'Output ONLY minified JSON: {"person":[],"org":[],"location":[],"date":[]}.',
    "code_debug": "Output ONLY the corrected code, no explanation.",
    "code_gen": "Output ONLY the code, no explanation.",
    "logic": "Give ONLY the final answer, no explanation.",
}


def _compress(text: str) -> str:
    """Trim trailing whitespace and collapse blank lines — saves prompt tokens
    without touching code indentation or meaning."""
    out, blank = [], False
    for ln in (text or "").split("\n"):
        ln = ln.rstrip()
        if not ln:
            if blank:
                continue
            blank = True
        else:
            blank = False
        out.append(ln)
    return "\n".join(out).strip()


def build_remote_messages(category: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": REMOTE_SYSTEM.get(category, "Answer only.")},
        {"role": "user", "content": _compress(prompt)},
    ]


def build_batch_messages(category: str, prompts: list[str]) -> list[dict]:
    """Pack several same-category short-answer tasks into ONE Fireworks call.

    Amortizes the fixed per-call overhead (system prompt + reasoning-model
    scaffold) across many tasks — the biggest token lever now that most tasks
    hit Fireworks. Only used for short-answer categories; the model must return
    one 'N) <answer>' line per item.
    """
    sys = REMOTE_SYSTEM.get(category, "Answer only.") + (
        " Answer each numbered item on its own line as 'N) answer'.")
    body = "\n".join(f"{i + 1}) {_compress(p)}" for i, p in enumerate(prompts))
    return [{"role": "system", "content": sys}, {"role": "user", "content": body}]


def build_retry_messages(category: str, prompt: str) -> list[dict]:
    """A stricter local re-attempt: emphasize exact output format so a malformed
    first answer becomes verifiable — recovering the task locally for 0 tokens."""
    strict = system_for(category) + (
        " Be precise and output ONLY the answer in the exact required format — "
        "no explanation, no preamble, no code fences unless the answer is code.")
    return [
        {"role": "system", "content": strict},
        {"role": "user", "content": prompt},
    ]
