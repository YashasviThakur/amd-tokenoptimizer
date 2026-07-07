"""Complexity router.

A fast, LLM-free heuristic scores each query 0..1. Below the threshold the
query is cheap and gets answered on-device (AMD GPU); at or above it, the query
is escalated to the frontier model. Keeping the router itself model-free is a
deliberate cost choice — the router must never become another expensive call.

Scoring blends: length, punctuation, code/math signals, and two tiers of
keywords. STRONG keywords (prove, algorithm, derive, …) are decisive; MEDIUM
keywords nudge. SIMPLE openers (what is, define, convert, …) pull back down so
factual one-liners stay local.
"""
from __future__ import annotations

import re

STRONG_KEYWORDS = (
    "prove", "proof", "derive", "derivation", "theorem", "algorithm", "complexity",
    "implement", "design", "architect", "analyze", "analysis", "refactor", "optimize",
    "sql", "regex", "gradient", "integral", "differentiate", "benchmark", "debug",
)
MEDIUM_KEYWORDS = (
    "step by step", "compare", "trade-off", "tradeoff", "reasoning", "strategy",
    "function", "translate", "evaluate", "explain why", "summarize the", "multi-step",
    "window function", "consensus", "distributed", "closure", "recursion", "justify",
    "explain the", "pros and cons",
)
CODE_RE = re.compile(
    r"[{}();=<>]|def |class |import |```|\bfor\b|\bwhile\b|\bselect\b"
    r"|\bpython\b|\bjavascript\b|\btypescript\b|\bjava\b|\bc\+\+\b|\brust\b|\bgolang\b"
)
MATH_RE = re.compile(
    r"\d+\s*[\+\-\*/\^]\s*\d+|integral|derivative|gradient|matrix|vector"
    r"|equation|theorem|probability|softmax"
)
SIMPLE_STARTS = (
    "hi", "hello", "hey", "thanks", "thank you", "what is", "whats", "what's",
    "who is", "who was", "who wrote", "when is", "when was", "where is", "define",
    "capital of", "how are you", "spell", "convert", "how many", "what time",
    "translate to",
)
WORD_RE = re.compile(r"\w+")


class ComplexityRouter:
    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    def score(self, query: str) -> float:
        q = (query or "").lower().strip()
        if not q:
            return 0.0
        s = 0.0
        n = len(WORD_RE.findall(q))
        s += min(0.30, n / 140.0)                 # longer prompts trend complex
        s += min(0.10, q.count("?") * 0.05)       # multiple questions
        s += min(0.08, q.count(".") * 0.02)       # multiple sentences
        if CODE_RE.search(q):
            s += 0.25
        if MATH_RE.search(q):
            s += 0.22

        strong = sum(0.22 for kw in STRONG_KEYWORDS if kw in q)
        s += min(0.50, strong)
        medium = sum(0.10 for kw in MEDIUM_KEYWORDS if kw in q)
        s += min(0.30, medium)

        for kw in SIMPLE_STARTS:
            if q.startswith(kw) or q == kw:
                s -= 0.20
                break
        return max(0.0, min(1.0, s))

    def decide(self, query: str) -> dict:
        sc = self.score(query)
        route = "remote" if sc >= self.threshold else "local"
        if route == "local":
            reason = f"low complexity ({sc:.2f}) → answered on-device (AMD GPU)"
        else:
            reason = f"high complexity ({sc:.2f}) → escalated to frontier model"
        return {"route": route, "complexity": round(sc, 3), "reason": reason}
