"""Fire a labeled query set at a running TokenOptimizer and print the savings.

Usage:
    python run.py                      # in one terminal
    python scripts/benchmark.py        # in another

    python scripts/benchmark.py --set demo --url http://localhost:4321
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

# Windows consoles default to cp1252 and choke on the bar/box glyphs below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def load_queries(which: str):
    with open(os.path.join(HERE, "curated_queries.json"), encoding="utf-8") as f:
        data = json.load(f)
    return data[which]


def bar(pct: float, width: int = 28) -> str:
    fill = int(round(pct / 100 * width))
    return "█" * fill + "·" * (width - fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:4321")
    ap.add_argument("--set", dest="which", default="benchmark", choices=["demo", "benchmark"])
    args = ap.parse_args()

    queries = load_queries(args.which)
    print(f"\n  TokenOptimizer benchmark · {len(queries)} queries · {args.url}\n")
    print(f"  {'route':<8}{'lat':>7}  {'saved':>11}  query")
    print("  " + "─" * 66)

    hits = {"cache": 0, "local": 0, "remote": 0}
    correct = 0
    with httpx.Client(timeout=180) as client:
        for q in queries:
            t0 = time.perf_counter()
            r = client.post(f"{args.url}/v1/chat/completions",
                            json={"messages": [{"role": "user", "content": q["query"]}]})
            r.raise_for_status()
            x = r.json().get("x_tokenoptimizer", {})
            route = x.get("route", "?")
            hits[route] = hits.get(route, 0) + 1
            if route == q.get("expect"):
                correct += 1
            lat = (time.perf_counter() - t0) * 1000
            qs = q["query"][:44] + ("…" if len(q["query"]) > 44 else "")
            print(f"  {route:<8}{lat:>6.0f}m  ${x.get('saved_usd', 0):>10.6f}  {qs}")

        stats = client.get(f"{args.url}/api/stats").json()

    print("  " + "─" * 66)
    print(f"\n  Routing:  cache {hits['cache']}   local·AMD {hits['local']}   remote·Fireworks {hits['remote']}")
    print(f"  Router accuracy vs labels: {correct}/{len(queries)}")
    print()
    print(f"  Baseline (all-frontier):  ${stats['baseline_usd']:.6f}")
    print(f"  Actual (TokenOptimizer):  ${stats['spent_usd']:.6f}")
    print(f"  Saved:                    ${stats['saved_usd']:.6f}")
    print(f"  Remote tokens avoided:    {stats['remote_tokens_avoided']:,}")
    print()
    print(f"  Cost reduction  {stats['saved_pct']:>5.1f}%  [{bar(stats['saved_pct'])}]")
    print(f"  Offloaded       {stats['local_pct']:>5.1f}%  [{bar(stats['local_pct'])}]")
    print()


if __name__ == "__main__":
    main()
