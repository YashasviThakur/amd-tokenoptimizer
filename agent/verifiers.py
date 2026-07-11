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
import os
import re
import subprocess
import sys
import tempfile

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

    # word-operator arithmetic: "47 plus 23", "12 times 4", "100 divided by 5",
    # "20 minus 5". EXACTLY two numeric operands in natural reading order, and
    # nothing else numeric outside the operator phrase — so a multi-step word
    # problem ("20 minus 5 apples, then double it") can't be crudely reduced.
    # Reversal words ("subtracted from", "less than") are excluded: their operand
    # order is inverted in prose and easy to compute backwards.
    mw = re.search(r"(-?\d+(?:\.\d+)?)\s+(plus|added to|minus|times|multiplied by|"
                   r"divided by)\s+(-?\d+(?:\.\d+)?)", p)
    if mw and len(re.findall(r"-?\d+(?:\.\d+)?", p)) == 2:
        rest = p[:mw.start()] + " " + p[mw.end():]
        rest = re.sub(r"\b(what is|what's|whats|calculate|compute|evaluate|"
                      r"the value of|the result of|result of|value of|equals|"
                      r"equal to|please|find|is|of)\b", " ", rest)
        # any leftover word/number means the phrase is a fragment of something
        # bigger -> defer to the model rather than answer the fragment.
        if not re.sub(r"[^a-z0-9]", "", rest):
            a, op, b = float(mw.group(1)), mw.group(2), float(mw.group(3))
            if op in ("plus", "added to"):
                return _fmt(a + b)
            if op == "minus":
                return _fmt(a - b)
            if op in ("times", "multiplied by"):
                return _fmt(a * b)
            if op == "divided by" and b != 0:
                return _fmt(a / b)

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


def _ner_source(prompt: str) -> str:
    """The text an extracted entity must be grounded in. When the prompt delimits
    the sentence with double/curly quotes (the common grader format) use ONLY the
    quoted span(s) — a stricter haystack. Otherwise fall back to the whole prompt:
    a genuinely-grounded entity is always a substring of the prompt, so the fallback
    can only over-escalate (safe, costs tokens), never keep a fabricated entity."""
    p = prompt or ""
    spans = re.findall(r'"([^"]{3,})"', p) + re.findall(r"“([^”]{3,})”", p)
    joined = " ".join(spans).strip()
    return joined if joined else p


def ner_entities_grounded(prompt: str, answer: str) -> bool:
    """True ONLY if `answer` parses to a JSON NER dict in which EVERY entity is a
    non-empty string that appears VERBATIM (case-insensitive) in the source text,
    AND at least one entity is present. This is the correctness gate that lets NER
    stay local: a hallucinated entity (not in the sentence), a malformed value, or
    an all-empty extraction returns False, so the router escalates to Fireworks.
    Conservative by construction — it can reject a good answer (wasting tokens) but
    can never keep an ungrounded one."""
    raw = answer or ""
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1)
    try:
        obj = json.loads(raw.strip())
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    hay = _ner_source(prompt).lower()
    any_ent = False
    for v in obj.values():
        if not isinstance(v, list):
            return False
        for e in v:
            if not isinstance(e, str):
                return False
            es = e.strip().lower()
            if not es:
                continue
            any_ent = True
            if es not in hay:
                return False  # hallucinated entity -> escalate
    return any_ent


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


# ── code_gen execution oracle (SECURITY-SENSITIVE: RUNS model-generated code) ──
# A code_gen answer is kept LOCAL only if the generated code actually PASSES the
# concrete (input -> expected) examples embedded in THIS prompt. Correctness is
# re-derived per prompt from the prompt's own examples (no stored answer key), so
# it cannot overfit and can never keep a wrong answer. Execution is locked down: a
# 2s wall timeout, CPU + address-space rlimits (POSIX), stdin closed, and a blanket
# except -> any hang/crash/OOM/error becomes 'fail' (=> the router escalates).

# One Python literal, shallow nesting — used only to FIND candidate operands in
# free-text example clauses; every hit is still re-validated with literal_eval.
_LIT = (r"(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""
        r"|\[[^\[\]]*\]|\{[^{}]*\}|\([^()]*\)"
        r"|-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+(?:[eE][+-]?\d+)?|True|False|None)")

# Separator between an example's call/input and its expected result. The four the
# task requires (==, returns, ->, gives) plus safe natural variants; word forms are
# \b-bounded so "gives" can't fire on "given".
_SEP_RE = re.compile(
    r"\s*(?:==|=>|->|→"
    r"|(?:returns?|should\s+return|gives?(?:\s+back)?|yields?|outputs?|produces?"
    r"|results?\s+in|evaluates?\s+to|maps?\s+to|equals?|is\s+equal\s+to|becomes?)\b"
    r")\s*[:=]?\s*", re.I)

# Free-text example clause with NO function name: "given <lit> ... returns <lit>".
# Both operands must be Python literals — a strong filter against false positives.
_NL_RE = re.compile(
    r"(?:given|for|input|when|with|on|if)\s+"
    r"(?:the\s+)?(?:input|list|string|value|arg(?:ument)?s?|number)?\s*(?:of|is|=|:)?\s*"
    r"(?P<inp>" + _LIT + r")"
    r"[^.\n]{0,40}?"
    r"(?:returns?|gives?(?:\s+back)?|outputs?|yields?|produces?|becomes?|maps?\s+to"
    r"|->|→|==|should\s+(?:return|be|give|output|produce)"
    r"|the\s+(?:result|output|answer)\s+(?:is|should\s+be))\s+"
    r"(?P<out>" + _LIT + r")",
    re.I)


def _safe_eval(text: str):
    """ast.literal_eval that never raises. Returns (value, ok)."""
    text = (text or "").strip()
    if not text:
        return None, False
    try:
        return ast.literal_eval(text), True
    except Exception:
        return None, False


def _end_of_quote(s: str) -> int:
    """s[0] is a quote char -> index just past the matching close quote (0 if none)."""
    q = s[0]
    i = 1
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == q:
            return i + 1
        i += 1
    return 0


def _balanced_span(s: str) -> int:
    """s[0] opens a ([{ -> index just past the matching close (0 if unbalanced).
    Quote- and escape-aware so brackets inside strings don't miscount."""
    depth = 0
    instr = None
    i = 0
    while i < len(s):
        ch = s[i]
        if instr:
            if ch == "\\":
                i += 2
                continue
            if ch == instr:
                instr = None
        elif ch in "'\"":
            instr = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return 0


def _balanced_call(s: str, start: int):
    """start = index just AFTER the opening '(' of name(...). Return (inner, rest):
    inner = content up to the matching ')', rest = text after it; (None, '') if
    unbalanced. Quote/escape-aware."""
    depth = 1
    instr = None
    i = start
    while i < len(s):
        ch = s[i]
        if instr:
            if ch == "\\":
                i += 2
                continue
            if ch == instr:
                instr = None
        elif ch in "'\"":
            instr = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return s[start:i], s[i + 1:]
        i += 1
    return None, ""


def _leading_literal(s: str):
    """Extract + eval the Python literal at the very start of s. Returns (value, ok)."""
    s = s.lstrip()
    if not s:
        return None, False
    c = s[0]
    if c in "'\"":
        j = _end_of_quote(s)
        cand = s[:j] if j else ""
    elif c in "([{":
        j = _balanced_span(s)
        cand = s[:j] if j else ""
    else:
        # int or float (opt. sci), stopping at a non-word char so a trailing
        # sentence period ("16.") or bracket ("5)") is left out of the literal.
        m = re.match(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?!\w)|(?:True|False|None)\b", s)
        cand = m.group(0) if m else ""
    return _safe_eval(cand)


def _parse_call_args(args_str: str):
    """The inside of name(...) -> a tuple of argument values (single or multiple).
    Returns (tuple, ok). A trailing comma forces a tuple even for one argument, so
    *args later expands correctly for both cases."""
    args_str = args_str.strip().rstrip(",").strip()
    if not args_str:
        return (), True
    val, ok = _safe_eval("(" + args_str + ",)")
    if ok and isinstance(val, tuple):
        return val, True
    return (), False


def _func_name(prompt: str, code: str):
    """The function under test: the FIRST `def <name>(` in the code, else a name in
    the prompt's write/implement instruction (backtick, quoted, or 'named X')."""
    m = re.search(r"^[ \t]*def[ \t]+([A-Za-z_]\w*)[ \t]*\(", code, re.M)
    if m:
        return m.group(1)
    m = re.search(r"(?:function|def)\s+(?:named\s+|called\s+)?[`'\"]?([A-Za-z_]\w*)[`'\"]?\s*\(",
                  prompt)
    if m:
        return m.group(1)
    m = re.search(r"[`'\"]([A-Za-z_]\w*)\s*\(", prompt)
    if m:
        return m.group(1)
    m = re.search(r"(?:function|def)\s+(?:named|called)\s+([A-Za-z_]\w*)", prompt)
    if m:
        return m.group(1)
    return None


def _extract_io_pairs(prompt: str, name: str):
    """Concrete (args_tuple, expected) examples parsed from the prompt. CONSERVATIVE:
    every operand is validated by ast.literal_eval; anything uncertain is skipped."""
    pairs = []
    seen = set()

    def _add(args, expected):
        key = repr((args, expected))
        if key not in seen:
            seen.add(key)
            pairs.append((args, expected))

    # 1) explicit calls: <name>(<args>) <sep> <result>
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", prompt):
        inner, rest = _balanced_call(prompt, m.end())
        if inner is None:
            continue
        sep = _SEP_RE.match(rest)
        if not sep:
            continue
        expected, ok_e = _leading_literal(rest[sep.end():])
        if not ok_e:
            continue
        args, ok_a = _parse_call_args(inner)
        if not ok_a:
            continue
        _add(args, expected)

    # 2) free-text example clauses (no function name)
    for m in _NL_RE.finditer(prompt):
        inp, ok1 = _safe_eval(m.group("inp"))
        out, ok2 = _safe_eval(m.group("out"))
        if ok1 and ok2:
            args = inp if isinstance(inp, tuple) else (inp,)
            _add(args, out)

    return pairs


def _rlimit_preexec():
    """A preexec_fn that caps the child's CPU seconds and address space, so a runaway
    (infinite loop / huge allocation) is killed by the OS. Returns None where the
    `resource` module is absent (Windows) — the wall timeout is then the sole guard."""
    try:
        import resource
    except Exception:
        return None

    def _apply():
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))

    return _apply


def run_extracted_tests(prompt: str, code: str) -> str:
    """'pass' | 'fail' | 'no_tests'. A code_gen answer is kept LOCAL only on 'pass'.

    Parse concrete (input -> expected) examples from `prompt`, run the model's `code`
    against them as asserts in a locked-down subprocess, and report whether they all
    pass. 'no_tests' when the prompt has no parseable example (=> escalate). SECURITY:
    any hang/crash/OOM/error -> 'fail' (never raises, never hangs, never keeps a wrong
    answer); the 2s timeout + rlimits + closed stdin + blanket except are the net."""
    tmpfile = None
    try:
        src = strip_code(code)
        if not src.strip():
            return "fail"
        name = _func_name(prompt, src)
        if not name:
            return "no_tests"
        pairs = _extract_io_pairs(prompt, name)
        if not pairs:
            return "no_tests"

        body = [src, "", "# --- assertions extracted from the prompt ---"]
        for args, expected in pairs:
            body.append(f"assert {name}(*{args!r}) == {expected!r}")
        script = "\n".join(body) + "\n"

        tmpdir = tempfile.gettempdir()
        fd, tmpfile = tempfile.mkstemp(suffix=".py", dir=tmpdir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(script)

        kwargs = dict(timeout=2, capture_output=True, cwd=tmpdir, stdin=subprocess.DEVNULL)
        preexec = _rlimit_preexec()
        if preexec is not None:
            kwargs["preexec_fn"] = preexec
        proc = subprocess.run([sys.executable, tmpfile], **kwargs)
        # 'pass' ONLY if every assert held (exit 0) AND >=1 test was extracted.
        return "pass" if proc.returncode == 0 else "fail"
    except Exception:
        # timeout / OS-kill (rlimit) / write error / anything unexpected -> escalate.
        return "fail"
    finally:
        if tmpfile:
            try:
                os.remove(tmpfile)
            except Exception:
                pass
