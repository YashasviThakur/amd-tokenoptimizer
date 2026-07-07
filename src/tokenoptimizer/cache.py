"""Semantic response cache — a hit returns a prior answer for zero new tokens."""
from __future__ import annotations

import threading
import time

import numpy as np


class SemanticCache:
    def __init__(self, threshold: float = 0.90, max_size: int = 2000):
        self.threshold = threshold
        self.max_size = max_size
        self._emb: list[np.ndarray] = []
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def lookup(self, embedding: np.ndarray):
        with self._lock:
            if not self._emb:
                self.misses += 1
                return None
            mat = np.vstack(self._emb)
            denom = np.linalg.norm(mat, axis=1) * (np.linalg.norm(embedding) + 1e-9) + 1e-9
            sims = (mat @ embedding) / denom
            i = int(np.argmax(sims))
            score = float(sims[i])
            if score >= self.threshold:
                self.hits += 1
                entry = self._entries[i]
                entry["last_hit"] = time.time()
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                return {"score": score, **entry}
            self.misses += 1
            return None

    def add(self, embedding, query, response_text, prompt_tokens, completion_tokens, model) -> None:
        with self._lock:
            if len(self._entries) >= self.max_size:
                self._emb.pop(0)
                self._entries.pop(0)
            self._emb.append(np.asarray(embedding, dtype=np.float32))
            self._entries.append({
                "query": query,
                "response_text": response_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": model,
                "created": time.time(),
                "hit_count": 0,
            })

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
            }
