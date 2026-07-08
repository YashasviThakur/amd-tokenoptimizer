"""Cheap, LLM-free task classifier over the eight Track-1 capability categories.

Categories: factual, math, sentiment, summarization, ner, code_debug,
logic, code_gen. Keeping this heuristic (not a model call) keeps it free and
instant — the router uses the category to pick the free solver and the
token-minimal Fireworks prompt/caps.
"""
from __future__ import annotations

import re

CATEGORIES = (
    "factual", "math", "sentiment", "summarization",
    "ner", "code_debug", "logic", "code_gen",
)

# Categories where a small (2-3B) local model is unreliable — multi-step
# reasoning and code. The router samples these twice (self-consistency) and is
# quick to escalate them to Fireworks.
HARD = {"math", "logic", "code_debug", "code_gen"}

_CODE_HINT = re.compile(r"```|def |class |function |\bcode\b|python|javascript|java\b|c\+\+")
_CODE_BLOCK = re.compile(r"\bdef\s+\w+\s*\(|```|\bclass\s+\w+\s*[:(]")
_WANTS_CODE = re.compile(
    r"write (?:a |me a |the )?(?:python )?function|write python|write code|"
    r"implement (?:a |the )?function|create (?:a |the )?function|generate (?:a )?function"
)
_COMPARATIVE = re.compile(
    r"\b(older|younger|taller|shorter|faster|slower|bigger|smaller|heavier|lighter"
    r"|higher|lower|earlier|later|richer|poorer|stronger|weaker|longer|ahead|behind)\b"
    r"[^.?!]*\bthan\b"
    r"|\b(ahead of|behind|in front of)\b"
    r"|finish(?:es|ed)?\s+(?:before|after|ahead|behind|last|first|a race)"
    r"|who finishes"
)


def classify(prompt: str) -> str:
    p = (prompt or "").lower()

    if any(k in p for k in ("sentiment", "positive or negative", "how does the author feel",
                            "classify the tone", "is this review")):
        return "sentiment"
    if any(k in p for k in ("named entit", "extract entit", "person, org", "identify the names",
                            "extract the entities", "list the entities")) \
            or ("entit" in p and ("extract" in p or "list the" in p)):
        return "ner"
    if any(k in p for k in ("summarise", "summarize", "summary", "tl;dr", "in one sentence",
                            "condense", "shorten this")):
        return "summarization"
    # real code present, or an explicit request to write code — decide before math
    has_code_block = bool(_CODE_BLOCK.search(p))
    wants_code = bool(_WANTS_CODE.search(p))
    if has_code_block or wants_code:
        if wants_code and not has_code_block:
            return "code_gen"
        if any(k in p for k in ("write", "implement", "complete the function", "finish the")):
            return "code_gen"
        return "code_debug"
    if any(k in p for k in ("fix the bug", "debug", "what's wrong", "whats wrong",
                            "correct the", "error in this code", "why does this code")):
        return "code_debug"
    if any(k in p for k in ("calculate", "what is", "how much", "how many", "percent", "%",
                            "average", "sum of", "product of", "speed", "total cost", "profit")) \
            and re.search(r"\d", p):
        return "math"
    if re.search(r"[0-9]+\s*[\+\-\*/x]\s*[0-9]+", p):
        return "math"
    # constraint/assignment puzzle: a "who owns/sits/…?" question with constraint
    # language ("each a different", "does not", "neither"). These are deductive
    # logic (not factual) — a small model gets them wrong, so route them to escalate.
    puzzle = bool(re.search(r"\bwho (?:owns|has|holds|sits|sit|drinks|lives|plays|drives|wears|"
                            r"got|likes|is (?:in|next|the|first|last|second))\b", p)
                  and re.search(r"\b(each|a different|different (?:pet|color|colour|job|house|drink|"
                                r"sport)|does not|doesn't|do not|don't|neither|only one|not the|"
                                r"no one|no two)\b", p))
    if any(k in p for k in ("who is the shortest", "who is the tallest", "who is the oldest",
                            "who is the youngest", "if all", "puzzle", "seating", "ranking",
                            "ranked by", "ranked from", "each of", "deduce", "in what order",
                            "which of the following", "who sits", "constraints", "answer yes or no",
                            "no ties", "finished ahead", "scored higher", "scored lower")) \
            or _COMPARATIVE.search(p) or puzzle:
        return "logic"
    if _CODE_HINT.search(p):
        return "code_gen" if ("write" in p or "implement" in p) else "code_debug"
    return "factual"
