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
import math
import re

from .verifiers import try_arithmetic  # pure expression + "x% of y"

# A name is a capitalized word, optionally with a single standalone label letter
# ("Box A", "Team B"). The \b after the letter stops it swallowing the next word
# (so "Bob And Carol" never captures "Bob A").
_NAME = r"[A-Z][a-z]+(?:\s[A-Z]\b)?"
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
    {"race": True, "max": {"first", "won", "win", "wins", "winner"},
     "min": {"last", "lost"}, "gt": set(), "lt": set()},
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
    # 1) intent from the question clause only (whole words), pick the dimension.
    # Accepts "who ..." and "which ...", ending at ? . or ! (many puzzles use
    # statement form: "Tell me who came in last."). Misfires are still fenced off
    # by the >=2-edge + unique-total-order requirement below.
    qm = re.search(r"\b(?:who|which)\b([^?.!\n]*)", prompt.lower())
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
        # "X finishes before Y", "X finished ahead of Y", "X came behind Y", …
        for a, rel, b in re.findall(
                rf"({_NAME})\s+(?:is|was|finish\w*|came|placed|ended up|ends up)\s+"
                rf"(before|after|ahead of|behind)\s+({_NAME})", prompt):
            higher = rel in ("before", "ahead of")  # earlier finish = higher rank
            edges.append((a, b) if higher else (b, a))
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


_PROP_VERB = (r"(?:need|needs|require|requires|have|has|can|absorb|absorbs|"
              r"contain|contains|produce|produces|breathe|breathes|grow|grows)")


def solve_syllogism(prompt: str) -> str | None:
    """Chained universal syllogism → Yes/No, or None if unprovable.

    Handles 'all A are B, all B are C, are all A C?' AND a multi-word predicate
    'all A are B, all B need water, do all A need water?'. Proves only a fully
    transitive chain; anything with 'some' or an unproven target defers.
    """
    q = prompt.lower()
    if re.search(r"\bsome\b", q):
        return None  # existentials: defer to the model

    def norm(w):
        return w.strip().rstrip("s")

    # question: "are all X Y?" or "do all X <predicate phrase>?"
    m = (re.search(r"are\s+all\s+(\w+?)s?\s+(?:definitely\s+|necessarily\s+)?([\w ]+?)\s*\?", q)
         or re.search(r"do\s+all\s+(\w+?)s?\s+([\w ]+?)\s*\?", q))
    if not m:
        return None
    x, target = norm(m.group(1)), m.group(2).strip()
    tnorm = norm(target)

    # premises = everything EXCEPT the question clause (so the question can't be
    # mistaken for a premise and prove itself).
    prem = q[:m.start()] + " " + q[m.end():]

    # a direct negative premise on the queried pair proves "No"
    for a, b in re.findall(r"no\s+(\w+?)s?\s+are\s+(\w+?)s?\b", prem):
        if norm(a) == x and norm(b) == tnorm:
            return "No"

    # implication graph: "all A are B" (subtype) + "all A <predicate>" (property)
    adj = collections.defaultdict(set)
    for a, b in re.findall(r"all\s+(\w+?)s?\s+are\s+(\w+?)s?\b", prem):
        adj[norm(a)].add(norm(b))
    for a, prop in re.findall(rf"all\s+(\w+?)s?\s+({_PROP_VERB}[\w ]*?)\s*[.?]", prem):
        adj[norm(a)].add(prop.strip())
    if not adj:
        return None

    seen, stack = set(), [x]
    while stack:
        for t in adj[stack.pop()]:
            if t == tnorm or norm(t) == tnorm or t == target:
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
        disc = re.search(r"(\d+(?:\.\d+)?)\s*%", p)
        # price must be anchored to a currency/price signal — NOT just "the first
        # number", which is the discount % in phrasings that state it first.
        price = (re.search(r"\$\s*(\d+(?:\.\d+)?)", p)
                 or re.search(r"(\d+(?:\.\d+)?)\s*dollars", p)
                 or re.search(r"(?:costs?|priced at|price(?:d| is| was)?|originally|was)\s+\$?\s*(\d+(?:\.\d+)?)", p))
        # single unambiguous discount, and the price is not the discount number
        if (price and disc
                and len(re.findall(r"\d+(?:\.\d+)?\s*%", p)) == 1
                and price.group(1) != disc.group(1)):
            return _fmt(float(price.group(1)) * (1 - float(disc.group(1)) / 100.0))

    if "speed" in p:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilometers|miles|meters|m)\b.*?\bin\s+"
                      r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", p)
        if m and float(m.group(2)) != 0:
            return _fmt(float(m.group(1)) / float(m.group(2)))

    # profit / loss percentage: "buys for $80 ... sells for $100 ... profit percentage"
    if "percentage" in p and ("profit" in p or "loss" in p):
        amts = re.findall(r"\$\s*(\d+(?:\.\d+)?)", p) or re.findall(r"(\d+(?:\.\d+)?)\s*dollars", p)
        buys = any(k in p for k in ("buys", "bought", "cost", "purchase"))
        sells = any(k in p for k in ("sells", "sold"))
        if buys and sells and len(amts) == 2:
            cost, sell = float(amts[0]), float(amts[1])
            if cost and "profit" in p and sell >= cost:
                return _fmt((sell - cost) / cost * 100.0)
            if cost and "loss" in p and cost >= sell:
                return _fmt((cost - sell) / cost * 100.0)

    # reverse percent change: "increased by 40% becomes 70" -> 70 / 1.40 = 50
    if len(re.findall(r"\d+(?:\.\d+)?", p)) == 2:
        m = re.search(r"(?:increased|grew|rose|raised|went up)\s+by\s+(\d+(?:\.\d+)?)\s*%"
                      r".*?(?:becomes?|is now|equals?|is|to)\s+(\d+(?:\.\d+)?)", p)
        if m:
            return _fmt(float(m.group(2)) / (1 + float(m.group(1)) / 100.0))
        m = re.search(r"(?:decreased|reduced|fell|dropped|lowered|went down)\s+by\s+(\d+(?:\.\d+)?)\s*%"
                      r".*?(?:becomes?|is now|equals?|is|to)\s+(\d+(?:\.\d+)?)", p)
        if m and float(m.group(1)) != 100:
            return _fmt(float(m.group(2)) / (1 - float(m.group(1)) / 100.0))

    return None


def solve_math_extra(prompt: str) -> str | None:
    """Word-form math that isn't a bare expression: powers, roots, factorial,
    gcd/lcm. Every branch requires an exact keyword and the exact operand count,
    otherwise defers — so it never guesses on a multi-number problem."""
    p = prompt.lower().replace(",", "")
    # Operand-count gate: a lone operation must contain EXACTLY the numbers it
    # consumes. This is what stops a compound problem ("2 to the power of 3 plus
    # 1", "5 squared plus 1", "square root of 16 plus 9") from being silently
    # answered by the first operand only — any extra number forces a defer.
    nums = re.findall(r"-?\d+(?:\.\d+)?", p)

    # factorial: "5 factorial", "factorial of 5", "5!" (exactly one integer)
    if ("factorial" in p or re.search(r"\b\d+\s*!", p)) and len(nums) == 1:
        v = float(nums[0])
        if v.is_integer() and 0 <= v <= 20:
            return str(math.factorial(int(v)))  # exact int, no float rounding

    # N to the power of M / N raised to (the power of) M (exactly two numbers)
    m = re.search(r"(\d+(?:\.\d+)?)\s+(?:to the power(?: of)?|raised to(?: the power(?: of)?)?)\s+"
                  r"(-?\d+(?:\.\d+)?)", p)
    if m and len(nums) == 2:
        return _fmt(float(m.group(1)) ** float(m.group(2)))

    # single-operand ops: squared / cubed / square root / cube root
    if len(nums) == 1:
        m = re.search(r"\b(\d+(?:\.\d+)?)\s+squared\b", p) or re.search(r"\bsquare of\s+(\d+(?:\.\d+)?)", p)
        if m:
            return _fmt(float(m.group(1)) ** 2)
        m = re.search(r"\b(\d+(?:\.\d+)?)\s+cubed\b", p) or re.search(r"\bcube of\s+(\d+(?:\.\d+)?)", p)
        if m:
            return _fmt(float(m.group(1)) ** 3)
        m = re.search(r"square root of\s+(\d+(?:\.\d+)?)", p)
        if m:
            return _fmt(math.sqrt(float(m.group(1))))
        m = re.search(r"cube root of\s+(\d+(?:\.\d+)?)", p)
        if m:
            return _fmt(round(float(m.group(1)) ** (1.0 / 3.0), 6))

    # gcd / lcm of exactly two integers
    if len(nums) == 2 and re.search(r"\b(?:gcd|greatest common (?:divisor|factor))\b", p):
        return str(math.gcd(int(float(nums[0])), int(float(nums[1]))))
    if len(nums) == 2 and re.search(r"\b(?:lcm|least common multiple)\b", p):
        a, b = int(float(nums[0])), int(float(nums[1]))
        if a and b:
            return str(a * b // math.gcd(a, b))

    return None


def free_solve(category: str, prompt: str) -> str | None:
    """Dispatch to a free solver for the category, or None to use a model."""
    if category == "math":
        return try_arithmetic(prompt) or solve_math_word(prompt) or solve_math_extra(prompt)
    if category == "logic":
        return solve_ordering(prompt) or solve_syllogism(prompt)
    return None
