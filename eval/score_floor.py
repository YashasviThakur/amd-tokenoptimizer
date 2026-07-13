"""Measure the 0-token heuristic floor: run the agent offline (no remote, no model)
on dev_tasks and score against dev_expected's check rules. Prints per-category and
total accuracy — the free accuracy we get before spending a single token."""
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

TASKS = HERE / "datasets" / "dev_tasks.json"
EXPECT = json.loads((HERE / "datasets" / "dev_expected.json").read_text(encoding="utf-8"))
OUT = Path(os.environ.get("TEMP", "/tmp")) / "floor_out.json"


def check(ans: str, spec: dict) -> bool:
    a = (ans or "").strip()
    al = a.lower()
    t = spec["type"]
    v = spec["value"]
    if t == "number":
        nums = re.findall(r"-?\d+(?:\.\d+)?", a.replace(",", ""))
        return any(abs(float(n) - float(v)) < 0.01 for n in nums) if nums else False
    if t == "exact":
        return al == str(v).lower()
    if t == "contains":
        return str(v).lower() in al
    if t == "label":
        return str(v).lower() in al                       # answer mentions the label
    if t == "keywords":
        allk = all(k.lower() in al for k in v.get("all", []))
        anyk = (not v.get("any")) or any(k.lower() in al for k in v["any"])
        return allk and anyk
    if t == "json_entities":
        try:
            obj = json.loads(a)
            flat = " ".join(str(x).lower() for lst in obj.values()
                            for x in (lst if isinstance(lst, list) else [lst]))
        except Exception:
            flat = al
        # need most expected entities present somewhere in the answer
        wanted = [e.lower() for lst in v.values() for e in lst]
        hit = sum(1 for e in wanted if e in flat)
        return hit >= max(1, len(wanted) - 1)             # allow one miss
    if t == "code":
        must = v.get("must_contain", [])
        return all(m.lower() in al for m in must)
    return False


env = dict(os.environ, STRICT_NO_REMOTE="1", USE_LOCAL="0", DISABLE_SOLVERS="0",
           ALLOWED_MODELS="", FIREWORKS_API_KEY="", MODEL_DISCOVERY="0",
           INPUT_PATH=str(TASKS), OUTPUT_PATH=str(OUT), PYTHONPATH=str(ROOT))
subprocess.run([sys.executable, "-m", "agent.main"], cwd=str(ROOT), env=env,
               capture_output=True, text=True, timeout=120)

res = {r["task_id"]: r["answer"] for r in json.load(open(OUT, encoding="utf-8"))}
by_cat = defaultdict(lambda: [0, 0])
correct = 0
for tid, spec in EXPECT.items():
    if "check" not in spec:
        continue
    ok = check(res.get(tid, ""), spec["check"])
    by_cat[spec["category"]][0] += int(ok)
    by_cat[spec["category"]][1] += 1
    correct += int(ok)
    mark = "OK " if ok else "  x"
    print(f"  {mark} {tid:5} {spec['category']:14} -> {repr(str(res.get(tid,''))[:50])}  (want {spec.get('answer')!r})")

total = sum(v[1] for v in by_cat.values())
print("\n--- per category (0-token heuristics only) ---")
for cat, (c, n) in sorted(by_cat.items()):
    print(f"  {cat:14} {c}/{n}")
print(f"\n0-TOKEN FLOOR: {correct}/{total} = {100*correct/total:.1f}%")
