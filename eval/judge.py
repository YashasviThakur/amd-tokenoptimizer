"""A local accuracy judge — a stand-in for the harness's LLM-judge.

Grades each answer with a category-appropriate check so we can measure our
accuracy-gate pass rate and token spend before submitting.
"""
from __future__ import annotations

import ast
import json
import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_FENCE = re.compile(r"```(?:json|python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _first_number(s: str):
    m = _NUM.search((s or "").replace(",", ""))
    return float(m.group()) if m else None


def _strip_fence(s: str) -> str:
    m = _FENCE.search(s or "")
    return (m.group(1) if m else (s or "")).strip()


def judge_one(check: dict, answer: str) -> bool:
    t = check["type"]
    v = check["value"]
    a = answer or ""

    if t == "number":
        n = _first_number(a)
        return n is not None and abs(n - float(v)) < 1e-6
    if t in ("exact", "label"):
        return _norm(a).strip(".") == _norm(v)
    if t == "contains":
        vals = v if isinstance(v, list) else [v]
        return any(_norm(x) in _norm(a) for x in vals)
    if t == "keywords":
        na = _norm(a)
        ok_all = all(_norm(k) in na for k in v.get("all", []))
        anyk = v.get("any", [])
        ok_any = (not anyk) or any(_norm(k) in na for k in anyk)
        return ok_all and ok_any
    if t == "json_entities":
        na = _norm(a)
        try:
            parsed = json.loads(_strip_fence(a))
        except Exception:
            parsed = None
        total = hit = 0
        for key, ents in v.items():
            for e in ents:
                total += 1
                found = False
                if isinstance(parsed, dict) and key in parsed:
                    if _norm(e) in _norm(json.dumps(parsed[key], ensure_ascii=False)):
                        found = True
                if not found and _norm(e) in na:
                    found = True
                hit += 1 if found else 0
        return total > 0 and hit / total >= 0.7
    if t == "code":
        src = _strip_fence(a)
        ok = all(m in src for m in v.get("must_contain", []))
        if v.get("compiles"):
            try:
                ast.parse(src)
            except Exception:
                ok = False
        return ok
    return False


def judge_all(results: list[dict], expected: dict) -> dict:
    by_id = {r["task_id"]: r.get("answer", "") for r in results}
    per = []
    for tid, exp in expected.items():
        per.append({
            "task_id": tid,
            "category": exp["category"],
            "pass": judge_one(exp["check"], by_id.get(tid, "")),
        })
    passed = sum(1 for p in per if p["pass"])
    return {"passed": passed, "total": len(per),
            "accuracy": passed / len(per) if per else 0.0, "per": per}
