"""Token-minimal prompts, per category.

Every remote token counts, so system prompts are terse, outputs are constrained
to exactly what the judge needs, and max_tokens is capped hard per category.
Local prompts are the same shape (local is free, but short answers are faster).
"""
from __future__ import annotations

# system prompt + max_tokens CEILING per category. These are truncation guards,
# not targets — a model stops when it's done, so a high ceiling costs nothing on
# a short answer but prevents a *reasoning* model from having its answer starved
# (the reasoning trace consumes budget before the final answer is emitted).
POLICY = {
    "factual":       ("Answer correctly and concisely. Output only the answer, no preamble.", 128),
    "math":          ("Solve the problem. Output only the final numeric answer, nothing else.", 128),
    "sentiment":     ("Classify the sentiment. Reply with exactly one word: positive, negative, or neutral.", 64),
    "summarization": ("Summarize as instructed, honoring any length constraint. Output only the summary.", 256),
    "ner":           ('Extract named entities. Output ONLY minified JSON with keys '
                      '"person","org","location","date" (each a list of strings).', 256),
    "code_debug":    ("Fix the bug. Output only the corrected code, no explanation.", 512),
    "logic":         ("Solve the puzzle. Reason internally, then output only the final answer.", 128),
    "code_gen":      ("Write the function to spec. Output only the code, no explanation.", 512),
}
DEFAULT = ("Answer correctly and concisely. Output only the answer.", 128)


def system_for(category: str) -> str:
    return POLICY.get(category, DEFAULT)[0]


def max_tokens_for(category: str) -> int:
    return POLICY.get(category, DEFAULT)[1]


def build_messages(category: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_for(category)},
        {"role": "user", "content": prompt},
    ]


# Ultra-short system prompts for *remote* calls — the task text already describes
# itself, so we only nudge the output format. Every saved prompt token is score.
REMOTE_SYSTEM = {
    "factual": "Answer only.",
    "math": "Number only.",
    "sentiment": "One word: positive, negative, or neutral.",
    "summarization": "Summary only.",
    "ner": "Minified JSON only.",
    "code_debug": "Corrected code only.",
    "code_gen": "Code only.",
    "logic": "Answer only.",
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
