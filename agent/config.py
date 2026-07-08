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
    # self-consistency draws (agreement = a free confidence signal). 1 disables it;
    # kept at 1 since factual now always escalates, and a 2nd draw doubles CPU time.
    local_samples_hard: int = int(os.getenv("LOCAL_SAMPLES_HARD", "1"))
    local_retry: bool = os.getenv("LOCAL_RETRY", "0").strip().lower() in ("1", "true", "yes")
    # Prompts longer than this skip the local model (slow CPU prefill on 2 vCPU
    # risks the <30s/task limit) and escalate to Fireworks instead.
    local_max_prompt_chars: int = int(os.getenv("LOCAL_MAX_PROMPT_CHARS", "2000"))

    # Keep a local answer when confidence >= this; else escalate to Fireworks.
    escalate_threshold: float = float(os.getenv("ESCALATE_THRESHOLD", "0.60"))
    # httpx read timeout. 14s so a single Fireworks call + one retry (14+0.5+14)
    # stays under the <30s/task limit even against a slow-but-alive endpoint.
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "14"))
    # Soft wall-clock budget: past this, remaining tasks skip local and go to
    # Fireworks (fast). main.py adds a HARD stop (+60s) that ends the loop and emits
    # empties, so a large/slow hidden set can never blow the 10-min budget (=ZERO).
    run_deadline_s: float = float(os.getenv("RUN_DEADLINE_S", "480"))
    # Baseline switch used by the eval harness: force every task to Fireworks.
    force_remote: bool = os.getenv("FORCE_REMOTE", "0").strip().lower() in ("1", "true", "yes")

    def has_remote(self) -> bool:
        return bool(self.allowed_models)


config = Config()
