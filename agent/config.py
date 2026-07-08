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

    # Local model (bundled in the image; llama-cpp-python, CPU). Local answers cost
    # 0 Fireworks tokens — the winning lever. Disabled → code+Fireworks only.
    use_local: bool = os.getenv("USE_LOCAL", "1").strip().lower() in ("1", "true", "yes")
    local_model_path: str = os.getenv("LOCAL_MODEL_PATH", "/models/model.gguf")
    local_n_ctx: int = int(os.getenv("LOCAL_N_CTX", "4096"))
    # 0 = let llama.cpp pick (all cores). Grading box has 2 vCPU.
    local_threads: int = int(os.getenv("LOCAL_THREADS", "0"))
    # self-consistency draws for hard categories (agreement = free confidence
    # signal). Costs CPU time; 1 disables it.
    local_samples_hard: int = int(os.getenv("LOCAL_SAMPLES_HARD", "2"))
    local_retry: bool = os.getenv("LOCAL_RETRY", "0").strip().lower() in ("1", "true", "yes")

    # Keep a local answer when confidence >= this; else escalate to Fireworks.
    escalate_threshold: float = float(os.getenv("ESCALATE_THRESHOLD", "0.60"))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "28"))
    # Global wall-clock budget (grading kills at 10min). When elapsed exceeds this,
    # remaining tasks skip slow local inference and go straight to Fireworks so the
    # run always finishes and writes output. Trades a few tokens to avoid TIMEOUT=0.
    run_deadline_s: float = float(os.getenv("RUN_DEADLINE_S", "540"))
    # Baseline switch used by the eval harness: force every task to Fireworks.
    force_remote: bool = os.getenv("FORCE_REMOTE", "0").strip().lower() in ("1", "true", "yes")

    def has_remote(self) -> bool:
        return bool(self.allowed_models)


config = Config()
