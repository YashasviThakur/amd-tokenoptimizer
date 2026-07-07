"""Token pricing table (USD per 1M tokens).

Numbers are approximate 2026 Fireworks AI serverless rates and are only used to
turn token counts into a *relative* dollar figure for the savings cockpit.
Local models running on your own AMD GPU have ~0 marginal token cost.
"""
from __future__ import annotations

PRICING: dict[str, dict[str, float]] = {
    # Remote frontier models (Fireworks AI) — approximate serverless $/1M tokens
    "accounts/fireworks/models/llama-v3p1-70b-instruct": {"prompt": 0.90, "completion": 0.90},
    "accounts/fireworks/models/llama-v3p1-405b-instruct": {"prompt": 3.00, "completion": 3.00},
    "accounts/fireworks/models/llama-v3p3-70b-instruct": {"prompt": 0.90, "completion": 0.90},
    "accounts/fireworks/models/qwen2p5-72b-instruct": {"prompt": 0.90, "completion": 0.90},
    "accounts/fireworks/models/deepseek-v3": {"prompt": 0.90, "completion": 0.90},
    # Local models on your AMD GPU — ~0 marginal token cost
    "google/gemma-3-4b-it": {"prompt": 0.0, "completion": 0.0},
    "google/gemma-3-1b-it": {"prompt": 0.0, "completion": 0.0},
    "google/gemma-2-2b-it": {"prompt": 0.0, "completion": 0.0},
}

DEFAULT_REMOTE = {"prompt": 0.90, "completion": 0.90}


def price_for(model: str) -> dict[str, float]:
    return PRICING.get(model, DEFAULT_REMOTE)


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = price_for(model)
    return (prompt_tokens * p["prompt"] + completion_tokens * p["completion"]) / 1_000_000
