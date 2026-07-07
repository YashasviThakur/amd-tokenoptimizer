"""Runtime configuration, read once from the environment (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # optional: load a local .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _b(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


@dataclass
class Settings:
    # mock = runs anywhere with zero deps/keys (for dev + demo fallback)
    # live = talks to a real local model (AMD/ROCm) + Fireworks AI
    mode: str = os.getenv("TOKENOPT_MODE", "mock").strip().lower()
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _i("PORT", 4321)

    # Local, on-AMD OpenAI-compatible server (vLLM / llama.cpp / ollama on ROCm)
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    local_api_key: str = os.getenv("LOCAL_API_KEY", "EMPTY")
    local_model: str = os.getenv("LOCAL_MODEL", "google/gemma-3-4b-it")

    # Remote frontier model (Fireworks AI)
    fireworks_base_url: str = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    fireworks_api_key: str = os.getenv("FIREWORKS_API_KEY", "")
    remote_model: str = os.getenv("REMOTE_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct")

    # Embeddings + semantic cache
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    cache_threshold: float = _f("CACHE_THRESHOLD", 0.90)
    cache_max_size: int = _i("CACHE_MAX_SIZE", 2000)

    # Router: complexity >= threshold escalates to the frontier model
    router_complexity_threshold: float = _f("ROUTER_THRESHOLD", 0.40)

    request_timeout: float = _f("REQUEST_TIMEOUT", 120.0)

    # Cosmetic: label shown on the GPU cockpit in mock/fallback mode
    gpu_label: str = os.getenv("GPU_LABEL", "AMD Instinct MI300X")


settings = Settings()
