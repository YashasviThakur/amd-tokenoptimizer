"""Configuration read purely from the environment (the harness injects these)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _split(name: str, default: str = "") -> list[str]:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


@dataclass
class Config:
    input_path: str = os.getenv("INPUT_PATH", "/input/tasks.json")
    output_path: str = os.getenv("OUTPUT_PATH", "/output/results.json")

    # Fireworks (remote) — injected by the harness. ALL remote calls go here.
    fireworks_api_key: str = os.getenv("FIREWORKS_API_KEY", "")
    fireworks_base_url: str = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models: list[str] = field(default_factory=lambda: _split("ALLOWED_MODELS"))

    # Preferred remote model (used if present in ALLOWED_MODELS), else first allowed.
    preferred_model: str = os.getenv("REMOTE_MODEL", "")
    # Frontier models are reasoning models — 'low' trims reasoning tokens (the score)
    # while keeping accuracy on the hard tasks we escalate. Ignored if unsupported.
    reasoning_effort: str = os.getenv("REASONING_EFFORT", "low")

    # Local model — free tokens. Any OpenAI-compatible server (llama.cpp/ollama/vLLM).
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    local_api_key: str = os.getenv("LOCAL_API_KEY", "EMPTY")
    local_model: str = os.getenv("LOCAL_MODEL", "local-small")

    # Routing knobs — tuned via the eval harness.
    escalate_threshold: float = float(os.getenv("ESCALATE_THRESHOLD", "0.60"))
    local_samples_hard: int = int(os.getenv("LOCAL_SAMPLES_HARD", "2"))
    # one free local re-attempt (strict prompt) before spending Fireworks tokens.
    # OFF by default: verifiers check format, not correctness, so a retry can keep
    # a valid-but-wrong answer (accuracy/gate risk). Enable + A/B only with data.
    local_retry: bool = os.getenv("LOCAL_RETRY", "0").strip().lower() in ("1", "true", "yes")
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "28"))
    # Baseline switch used by the eval harness: force every task to Fireworks.
    force_remote: bool = os.getenv("FORCE_REMOTE", "0").strip().lower() in ("1", "true", "yes")

    def has_remote(self) -> bool:
        return bool(self.allowed_models)


config = Config()
