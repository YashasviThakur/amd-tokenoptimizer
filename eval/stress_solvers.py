"""Solver-correctness + classifier check on a labeled dataset.

The one thing a free solver must NEVER do is fire and return a wrong answer —
that silently costs accuracy. This asserts: whenever free_solve() returns an
answer, that answer passes the task's check. Also reports classifier accuracy.

    python -m eval.stress_solvers                    # dev set
    python -m eval.stress_solvers --tasks X --expected Y
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.classifier import classify        # noqa: E402
from agent.solvers import free_solve          # noqa: E402
from eval.judge import judge_one              # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(HERE / "datasets" / "dev_tasks.json"))
    ap.add_argument("--expected", default=str(HERE / "datasets" / "dev_expected.json"))
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))

    clf_correct = fired = fired_correct = 0
    misfires, misclass = [], []
    for t in tasks:
        tid, prompt = t["task_id"], t.get("prompt", "")
        exp = expected.get(tid)
        cat = classify(prompt)
        if exp and cat == exp["category"]:
            clf_correct += 1
        elif exp:
            misclass.append((tid, exp["category"], cat, prompt[:55]))

        ans = free_solve(cat, prompt)
        if ans is not None:
            fired += 1
            if exp and judge_one(exp["check"], ans):
                fired_correct += 1
            else:
                misfires.append((tid, cat, prompt[:55], ans))

    n = len(tasks)
    print(f"\n  tasks: {n}")
    print(f"  classifier correct:       {clf_correct}/{n}")
    print(f"  free solver fired:        {fired}/{n}")
    print(f"  correct WHEN it fired:    {fired_correct}/{fired}  "
          f"{'✓ no misfires' if fired == fired_correct else '✗ MISFIRES (gate risk!)'}")
    if misfires:
        print("\n  MISFIRES (solver emitted a WRONG answer — must guard):")
        for tid, cat, p, ans in misfires:
            print(f"    [{tid}] ({cat}) {p!r} -> {ans!r}")
    if misclass:
        print(f"\n  misclassifications ({len(misclass)}):")
        for tid, want, got, p in misclass[:20]:
            print(f"    [{tid}] want {want:<14} got {got:<14} | {p!r}")
    print()
    return 0 if fired == fired_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
