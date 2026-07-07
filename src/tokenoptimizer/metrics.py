"""Savings accounting — the numbers the cockpit shows and the demo lives on.

Every request is priced two ways: what it *actually* cost (0 for cache/local,
real Fireworks price for remote) and what it *would* have cost if the whole
workload had gone to the frontier model (the baseline). The gap is the money
TokenOptimizer saved.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from .pricing import cost_usd


class MetricsStore:
    def __init__(self, remote_model: str, maxrecent: int = 60):
        self.remote_model = remote_model
        self._lock = threading.Lock()
        self.records: deque = deque(maxlen=maxrecent)
        self.total_requests = 0
        self.route_counts = {"cache": 0, "local": 0, "remote": 0}
        self.spent_usd = 0.0
        self.baseline_usd = 0.0
        self.tokens_processed = 0
        self.remote_tokens_avoided = 0
        self.total_latency_ms = 0.0

    def record(self, *, route, query, prompt_tokens, completion_tokens, model,
               latency_ms, complexity=None, reason="", cache_score=None) -> dict:
        with self._lock:
            self.total_requests += 1
            self.route_counts[route] = self.route_counts.get(route, 0) + 1
            actual = 0.0 if route in ("cache", "local") else cost_usd(model, prompt_tokens, completion_tokens)
            baseline = cost_usd(self.remote_model, prompt_tokens, completion_tokens)
            self.spent_usd += actual
            self.baseline_usd += baseline
            self.tokens_processed += prompt_tokens + completion_tokens
            if route in ("cache", "local"):
                self.remote_tokens_avoided += prompt_tokens + completion_tokens
            self.total_latency_ms += latency_ms
            rec = {
                "ts": time.time(),
                "route": route,
                "query": (query or "")[:120],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": model,
                "latency_ms": round(latency_ms, 1),
                "cost_usd": round(actual, 6),
                "baseline_usd": round(baseline, 6),
                "saved_usd": round(baseline - actual, 6),
                "complexity": complexity,
                "reason": reason,
                "cache_score": cache_score,
            }
            self.records.appendleft(rec)
            return rec

    def snapshot(self) -> dict:
        with self._lock:
            saved = self.baseline_usd - self.spent_usd
            pct = (saved / self.baseline_usd * 100.0) if self.baseline_usd > 0 else 0.0
            offloaded = self.route_counts["local"] + self.route_counts["cache"]
            local_pct = (offloaded / self.total_requests * 100.0) if self.total_requests else 0.0
            avg_lat = (self.total_latency_ms / self.total_requests) if self.total_requests else 0.0
            return {
                "total_requests": self.total_requests,
                "route_counts": dict(self.route_counts),
                "spent_usd": round(self.spent_usd, 6),
                "baseline_usd": round(self.baseline_usd, 6),
                "saved_usd": round(saved, 6),
                "saved_pct": round(pct, 1),
                "tokens_processed": self.tokens_processed,
                "remote_tokens_avoided": self.remote_tokens_avoided,
                "local_pct": round(local_pct, 1),
                "avg_latency_ms": round(avg_lat, 1),
            }

    def recent(self) -> list:
        with self._lock:
            return list(self.records)
