"""Free deterministic solvers — correct answers with zero Fireworks tokens.

These are *general* (pattern-based, not answer-hardcoded) and, above all,
*conservative*: each returns None unless it can fully prove the answer, so a
miss escalates to the model rather than emitting a wrong answer. Emitting a
wrong answer silently costs the accuracy gate — far worse than spending a few
tokens — so every ambiguity resolves to None.
"""
from __future__ import annotations

import collections
import graphlib
import re

from .verifiers import try_arithmetic  # pure expression + "x% of y"

_NAME = r"[A-Z][a-z]+"
_GT = {"taller", "older", "faster", "bigger", "larger", "heavier", "stronger",
       "richer", "wealthier", "longer"}
_LT = {"shorter", "younger", "slower", "smaller", "lighter", "weaker", "poorer"}

# Each ordering dimension: the superlatives that select it and the comparators
# that belong to it. Edges from a *different* dimension are rejected (mixed →
# defer), and intent is read only from the question clause (so a name like
# "First" elsewhere can't flip it).
_ATTRS = [
    {"max": {"tallest"}, "min": {"shortest"}, "gt": {"taller"}, "lt": {"shorter"}},
    {"max": {"oldest"}, "min": {"youngest"}, "gt": {"older"}, "lt": {"younger"}},
    {"max": {"fastest"}, "min": {"slowest"}, "gt": {"faster"}, "lt": {"slower"}},
    {"max": {"biggest", "largest"}, "min": {"smallest"}, "gt": {"bigger", "larger"}, "lt": {"smaller"}},
    {"max": {"heaviest"}, "min": {"lightest"}, "gt": {"heavier"}, "lt": {"lighter"}},
    {"max": {"strongest"}, "min": {"weakest"}, "gt": {"stronger"}, "lt": {"weaker"}},
    {"max": {"richest", "wealthiest"}, "min": {"poorest"}, "gt": {"richer", "wealthier"}, "lt": {"poorer"}},
    {"max": {"longest"}, "min": {"shortest"}, "gt": {"longer"}, "lt": {"shorter"}},
    {"race": True, "max": {"first"}, "min": {"last"}, "gt": set(), "lt": set()},
]


def _fmt(x: float) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(round(x, 4))


def _reach(adj, n):
    seen, stack = set(), [n]
    while stack:
        for y in adj[stack.pop()]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return seen


def solve_ordering(prompt: str) -> str | None:
    """Comparative chains → who is the X-est. Proves a unique total order or None."""
    # 1) intent from the question clause only (whole words), pick the dimension
    qm = re.search(r"who\b([^?]*)\?", prompt.lower())
    if not qm:
        return None
    qwords = set(re.findall(r"[a-z]+", qm.group(1)))
    attr = want_max = want_min = None
    for a in _ATTRS:
        hi, lo = bool(qwords & a["max"]), bool(qwords & a["min"])
        if hi or lo:
            if hi and lo:
                return None  # ambiguous
            attr, want_max, want_min = a, hi, lo
            break
    if attr is None:
        return None

    # 2) edges — only this dimension; any other-dimension comparator → defer
    edges = []
    for a, comp, b in re.findall(rf"({_NAME})\s+is\s+(\w+)\s+than\s+({_NAME})", prompt):
        c = comp.lower()
        if c in attr["gt"]:
            edges.append((a, b))
        elif c in attr["lt"]:
            edges.append((b, a))
        elif c in _GT or c in _LT:
            return None  # mixed dimension
    if attr.get("race"):
        for a, rel, b in re.findall(rf"({_NAME})\s+finishes\s+(before|after)\s+({_NAME})", prompt):
            edges.append((a, b) if rel == "before" else (b, a))
    if len(edges) < 2:
        return None

    # 3) reject contradictions and cycles
    edge_set = set(edges)
    if any((lo, hi) in edge_set for hi, lo in edges):
        return None
    adj = collections.defaultdict(set)
    names = set()
    for hi, lo in edges:
        adj[hi].add(lo)
        names.update((hi, lo))
    try:
        graphlib.TopologicalSorter(adj).static_order()
    except graphlib.CycleError:
        return None

    # 4) require a fully-determined total order (scores are a 0..n-1 permutation)
    score = {n: len(_reach(adj, n)) for n in names}
    if sorted(score.values()) != list(range(len(names))):
        return None
    return max(score, key=score.get) if want_max else min(score, key=score.get)


def solve_syllogism(prompt: str) -> str | None:
    """'all A are B, all B are C … are all A C?' → Yes/No, or None if unprovable."""
    q = prompt.lower()
    if re.search(r"\bsome\b", q):
        return None  # existentials: defer to the model
    m = re.search(r"are\s+all\s+(\w+)\s+(?:definitely\s+|necessarily\s+)?(\w+)\s*\?", q)
    if not m:
        return None
    x, y = m.group(1), m.group(2)

    # a direct negative premise on the queried pair proves "No"
    for a, b in re.findall(r"no\s+(\w+)\s+are\s+(\w+)", q):
        if a == x and b == y:
            return "No"

    pos = re.findall(r"all\s+(\w+)\s+are\s+(\w+)", q)
    if not pos:
        return None
    adj = collections.defaultdict(set)
    for a, b in pos:
        adj[a].add(b)
    seen, stack = set(), [x]
    while stack:
        for t in adj[stack.pop()]:
            if t == y:
                return "Yes"
            if t not in seen:
                seen.add(t)
                stack.append(t)
    return None  # cannot prove Yes and no direct negative → escalate


def solve_math_word(prompt: str) -> str | None:
    """Guarded word-problem arithmetic: averages/sums, discount, speed = dist/time.

    Each branch uses a tight capture so it only fires on a clean, unambiguous
    list/pattern and otherwise defers (returns None) rather than risk a wrong
    answer on a multi-step problem.
    """
    pl = prompt.lower()

    # average / mean of an explicit list: "average of 12, 18, and 30"
    am = re.search(r"(?:average|mean)\s+of\s+([\d\.\s,and]+?)\s*[.?]", pl)
    if am and "speed" not in pl and "rate" not in pl and "per " not in pl:
        nums = re.findall(r"\d+(?:\.\d+)?", am.group(1))
        if len(nums) >= 2:
            vals = [float(x) for x in nums]
            return _fmt(sum(vals) / len(vals))

    # sum / total of an explicit list: "the sum of 3, 5, and 7"
    sm = re.search(r"(?:sum|total)\s+of\s+([\d\.\s,and]+?)\s*[.?]", pl)
    if sm:
        nums = re.findall(r"\d+(?:\.\d+)?", sm.group(1))
        if len(nums) >= 2:
            return _fmt(sum(float(x) for x in nums))

    p = prompt.lower().replace(",", "")

    if any(k in p for k in ("discount", "reduced by", "% off")):
        price = re.search(r"\$?\s*(\d+(?:\.\d+)?)", p)
        disc = re.search(r"(\d+(?:\.\d+)?)\s*%", p)
        # single, unambiguous discount only (one percentage in the text)
        if price and disc and len(re.findall(r"\d+(?:\.\d+)?\s*%", p)) == 1:
            return _fmt(float(price.group(1)) * (1 - float(disc.group(1)) / 100.0))

    if "speed" in p:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilometers|miles|meters|m)\b.*?\bin\s+"
                      r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", p)
        if m and float(m.group(2)) != 0:
            return _fmt(float(m.group(1)) / float(m.group(2)))

    return None


def free_solve(category: str, prompt: str) -> str | None:
    """Dispatch to a free solver for the category, or None to use a model."""
    if category == "math":
        return try_arithmetic(prompt) or solve_math_word(prompt)
    if category == "logic":
        return solve_ordering(prompt) or solve_syllogism(prompt)
    return None
