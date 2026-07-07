"""Inference backends: a real OpenAI-compatible client and a zero-dep mock.

`OpenAICompatBackend` talks to any OpenAI-style /chat/completions endpoint —
that's vLLM or llama.cpp serving Gemma on AMD ROCm for `local`, and Fireworks AI
for `remote`. `MockBackend` fabricates realistic answers, latencies and token
counts so the product runs (and the demo survives) with no models or keys.
"""
from __future__ import annotations

import asyncio
import random
import time

import httpx


def _content(message) -> str:
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", "") or ""


def last_user_text(messages) -> str:
    for m in reversed(messages or []):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "user":
            return _content(m)
    return _content(messages[-1]) if messages else ""


def _short(text: str, n: int = 90) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[:n] + "…"


class OpenAICompatBackend:
    def __init__(self, name, base_url, api_key, model, timeout=120.0):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def complete(self, messages, temperature=0.7, max_tokens=512, model=None) -> dict:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = data.get("usage", {}) or {}
        return {
            "text": data["choices"][0]["message"]["content"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "model": data.get("model", self.model),
            "latency_ms": latency_ms,
        }


class MockBackend:
    def __init__(self, name, model, kind, min_ms, max_ms):
        self.name = name
        self.model = model
        self.kind = kind  # "local" | "remote"
        self.min_ms = min_ms
        self.max_ms = max_ms

    async def complete(self, messages, temperature=0.7, max_tokens=512, model=None) -> dict:
        from .tokens import count_message_tokens, count_tokens

        user = last_user_text(messages)
        span = self.min_ms + (self.max_ms - self.min_ms) * min(1.0, len(user) / 300.0)
        delay = span * random.uniform(0.85, 1.15)
        await asyncio.sleep(delay / 1000.0)

        if self.kind == "local":
            text = (
                f"[on-device · {self.model} · AMD ROCm] "
                f"Concise answer to “{_short(user)}”. "
                "Handled locally on the AMD GPU — zero tokens sent to the cloud."
            )
        else:
            text = (
                f"[frontier · {self.model} · Fireworks AI] "
                f"In-depth response to “{_short(user)}”. "
                "This query scored complex enough to escalate to the large remote model, "
                "which returns the multi-step reasoning and detail the task requires."
            )
        return {
            "text": text,
            "prompt_tokens": count_message_tokens(messages),
            "completion_tokens": count_tokens(text),
            "model": self.model,
            "latency_ms": delay,
        }
