"""Free, model-free checks + an exact arithmetic calculator (zero tokens).

None of these call a model. They turn "is this answer well-formed?" into cheap
signals used by the solvers and the eval judge: a valid number for math,
parseable JSON for NER, compilable code for code tasks, a real label for
sentiment, a length-constrained summary. Plus a deterministic calculator that
answers pure-arithmetic math for free.
"""
from __future__ import annotations

import ast
import json
import operator as _op
import re

_LABELS = {"positive", "negative", "neutral"}
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_FENCE_RE = re.compile(r"```(?:json|python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


# ── arithmetic calculator (free, exact) ──────────────────────────────────────
_OPS = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
        ast.Div: _op.truediv, ast.Pow: _op.pow, ast.Mod: _op.mod, ast.USub: _op.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsafe expression")


def _fmt(x: float) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(round(x, 4)) if isinstance(x, float) else str(x)


def try_arithmetic(prompt: str) -> str | None:
    """Return an exact answer for a clearly arithmetic question, else None."""
    p = (prompt or "").lower().replace("×", "*").replace("÷", "/").replace("^", "**")

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%)\s*of\s*(\d+(?:\.\d+)?)", p)
    if m:
        # only trust "X% of Y" when it IS the whole computation: exactly two
        # numbers and no trailing operator (else "20% of 50 plus 5" would wrongly
        # return 10), and no WORD modifier wrapping it ("HALF of 20% of 50",
        # "TWICE 10% of 40" — verified misfires: the digit-count guard alone let
        # them through). A compound percent expression defers to the model.
        if re.search(r"\b(?:half|twice|double|doubled|triple|thrice|quarter|third)\b", p):
            return None
        two_nums = len(re.findall(r"\d+(?:\.\d+)?", p)) == 2
        if two_nums and not re.search(r"[\+\-\*/]\s*\d", p[m.end():]):
            return _fmt(float(m.group(1)) / 100.0 * float(m.group(2)))
        return None  # compound percent expression -> escalate, don't crude-eval

    m = re.search(r"([0-9][0-9\.\s\+\-\*/\(\)]*[0-9\)])", p)
    if m and re.search(r"[\+\-\*/]", m.group(1)):
        # Only trust the captured expression when it IS the whole question — a
        # bare compute request like "What is 47 * 23?". Red-teaming showed the
        # old unguarded eval turning year ranges ("from 1914-1918?" -> -4),
        # scores ("3-11" -> -8), and fractions ("1/2 of 30" -> 0.5) into
        # confident wrong answers. If ANY word besides a compute lead-in
        # remains outside the expression, defer to the model.
        rest = p[:m.start()] + " " + p[m.end():]
        rest = re.sub(r"\b(what is|what's|whats|calculate|compute|evaluate|"
                      r"the value of|the result of|result of|value of|equals|"
                      r"equal to|please|find)\b", " ", rest)
        # a minus/slash OUTSIDE the captured expression means the capture is a
        # fragment of something bigger (a negative operand, a fraction) — defer.
        if "-" in rest or "/" in rest or re.sub(r"[^a-z0-9]", "", rest):
            return None
        try:
            return _fmt(_eval(ast.parse(m.group(1).strip(), mode="eval").body))
        except Exception:
            return None
    return None


# ── validity checks ──────────────────────────────────────────────────────────
def is_number(text: str) -> bool:
    """True if the answer is essentially a number (allows $, commas, a unit word)."""
    t = (text or "").strip()
    if not t or len(t) > 24:
        return False
    core = t.replace(",", "").replace("$", "").rstrip(".").strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", core):
        return True
    # e.g. "75 km/h", "60 dollars" — starts with a number, short
    return bool(re.match(r"^-?\d+(?:\.\d+)?\b", core))


def label_ok(text: str) -> bool:
    return (text or "").strip().strip(".").lower() in _LABELS


def strip_code(text: str) -> str:
    m = _FENCE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def code_compiles(text: str) -> bool:
    src = strip_code(text)
    if not src:
        return False
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False


_NER_KEYS = {"person", "people", "org", "organization", "organisation",
             "location", "loc", "place", "date", "time", "misc"}


def valid_ner_json(text: str) -> bool:
    """Stricter than valid_json for NER confidence: require a JSON OBJECT keyed by
    entity types (the requested schema), not just any parseable JSON. A bare list
    like ["Obama"] or a wrong-shape blob no longer counts as 'well-formed'."""
    raw = text or ""
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    try:
        obj = json.loads(raw.strip())
    except Exception:
        return False
    return isinstance(obj, dict) and any(str(k).lower() in _NER_KEYS for k in obj)


def valid_json(text: str) -> bool:
    raw = text or ""
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    raw = raw.strip()
    try:
        json.loads(raw)
        return True
    except Exception:
        return False


def length_ok(prompt: str, answer: str) -> bool:
    p = (prompt or "").lower()
    words = len(re.findall(r"\w+", answer or ""))
    sentences = len(re.findall(r"[.!?]+", answer or "")) or 1
    if "one sentence" in p or "single sentence" in p:
        return sentences <= 1 and words > 0
    m = re.search(r"(\d+)\s*words?", p)
    if m:
        return words <= int(m.group(1)) * 1.3
    return words > 0
