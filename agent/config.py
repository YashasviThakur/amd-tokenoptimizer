"""Configuration read purely from the environment (the harness injects these)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _split(name: str, default: str = "") -> list[str]:
    return [m.strip() for m in os.getenv(name, default).split(",") if m.strip()]


# The harness may name the allowed-model list differently. Read every plausible
# env var so the Fireworks path is never silently disabled by a naming mismatch
# (an empty allow-list made has_remote() False -> zero API calls -> gate failure).
# Allow-list SOURCES only. REMOTE_MODEL is deliberately NOT here: it is a PREFERENCE
# hint (read into preferred_model below), not a statement that the model is allowed.
# Listing it merged our own REMOTE_MODEL default into config.allowed_models, so the
# router called an id the grader never allowed -> MODEL_VIOLATION (the whole
# submission unscoreable). A preference must only REORDER the injected list, never
# extend it.
_MODEL_ENV_PRIORITY = ("ALLOWED_MODELS", "MODELS", "FIREWORKS_MODELS", "MODEL_NAME",
                       "MODEL", "FIREWORKS_MODEL", "LLM_MODEL")
# our OWN config vars that contain "MODEL" but never hold a harness allow-list id
# (REMOTE_MODEL included: it is a preference, so the generic sweep must skip it too).
_MODEL_ENV_OWN = ("MODEL_DISCOVERY", "LOCAL_MODEL_PATH", "FALLBACK_MODELS", "REMOTE_MODEL")
# a model id: no spaces, path-ish charset, and at least one letter + 3 chars —
# rejects flag values like "0"/"1"/"true" leaking in from boolean *MODEL* vars.
_MODEL_ID = re.compile(r"^(?=.{3,})(?=.*[A-Za-z])[\w./:-]+$")
_MODEL_NON_ID = {"true", "false", "yes", "no", "none", "null", "auto", "default"}


def _discover_models() -> list[str]:
    """Model list from the environment: the known names first; ONLY if none of
    them is set, a generic sweep of ANY *MODEL* env var (the harness's exact
    name is unconfirmed — a missed injection strands escalation on the fallback
    list). The sweep never runs alongside a real ALLOWED_MODELS, so a helper
    var like MODEL_PROVIDER=fireworks can never displace or pollute the
    authoritative list, and URL/path-shaped values are rejected outright."""
    out: list[str] = []
    for name in _MODEL_ENV_PRIORITY:
        for m in _split(name):
            if m not in out:
                out.append(m)
    if out:
        return out
    for name, val in os.environ.items():
        up = name.upper()
        if "MODEL" not in up or up in _MODEL_ENV_PRIORITY or up in _MODEL_ENV_OWN:
            continue
        if up.startswith("LOCAL_") or "PATH" in up or "DIR" in up:
            continue  # LOCAL_MODEL_PATH etc. are ours, not the harness's
        for m in (v.strip() for v in val.split(",")):
            if ("://" in m or m.startswith("/") or not _MODEL_ID.match(m or " ")
                    or m.lower() in _MODEL_NON_ID or m in out):
                continue
            out.append(m)
    return out


@dataclass
class Config:
    input_path: str = os.getenv("INPUT_PATH", "/input/tasks.json")
    output_path: str = os.getenv("OUTPUT_PATH", "/output/results.json")

    # Fireworks (remote) — injected by the harness. ALL remote calls go here.
    fireworks_api_key: str = os.getenv("FIREWORKS_API_KEY", "")
    fireworks_base_url: str = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    allowed_models: list[str] = field(default_factory=_discover_models)

    # Preferred remote model (used if present in ALLOWED_MODELS), else first allowed.
    preferred_model: str = os.getenv("REMOTE_MODEL", "")
    # REMOTE-FIRST (gate-pass mode): every non-solver task goes to Fireworks; the
    # local model is ONLY a dead-remote rescue. All four qualifying leaderboard
    # agents sit at exactly 84.2% (16/19) with all-remote profiles, while every
    # local-first image failed the gate (10.5% -> 26.3%): the local 3B's format-only
    # confidence gates keep wrong-but-well-formed answers, and classifier fall-
    # through routes hard word problems to it. Remote-first buys the qualifying
    # accuracy; solvers still take their tasks for 0 tokens first.
    remote_first: bool = os.getenv("REMOTE_FIRST", "1").strip().lower() in ("1", "true", "yes")
    # Used ONLY if the harness injects no model list at all: the VERBATIM Track-1
    # launch-day ALLOWED_MODELS (short names, community-confirmed). Never prefix
    # ids with accounts/fireworks/models/ — the judging proxy matches the
    # allow-list entry verbatim, and a non-listed string is a MODEL_VIOLATION.
    # SERVERLESS-FIRST: gpt-oss-120b is always-on (no deployment, bills only on use)
    # and answered 8/8 on the real API in our live smoke. gemma/minimax/kimi are
    # ON-DEMAND — a 404 there means "not deployed", not "banned" (organizer note),
    # so leading with them made every remote call die when nothing was deployed.
    # The router sends every id VERBATIM (no bare/prefixed toggling) — an off-list
    # spelling is a MODEL_VIOLATION — so this fallback fires only when the harness
    # injects NO list at all, and its entries must already match the proxy's naming.
    fallback_models: list[str] = field(default_factory=lambda: _split(
        "FALLBACK_MODELS",
        "gpt-oss-120b,gpt-oss-20b,gemma-4-31b-it,"
        "minimax-m3,kimi-k2p7-code"))
    # Default OFF for the judging proxy: the field is nonstandard, the allowed
    # gemma-4 models don't use it, and every rejected call costs a second POST
    # (and possibly double-billed prompt tokens gateway-side). Set REASONING_EFFORT
    # explicitly for local testing against real Fireworks reasoning models.
    reasoning_effort: str = os.getenv("REASONING_EFFORT", "")

    # Local model (bundled in the image; llama-cpp-python, CPU). Local answers cost
    # 0 Fireworks tokens — but a 3B-Q4 model is unreliable on the broad hidden set
    # and its wrong answers were failing the ACCURACY gate (=excluded, tokens moot).
    # Default OFF: pass the gate first (exact solvers + Fireworks), re-enable per-
    # category only once the local model is proven on the real distribution.
    use_local: bool = os.getenv("USE_LOCAL", "0").strip().lower() in ("1", "true", "yes")
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

    # ZERO-TOKEN mode: never call Fireworks — solvers + the bundled local model
    # answer everything. 0 tokens is the unbeatable floor of an ascending-token
    # leaderboard; flip on ONLY once the leaderboard confirms local-only accuracy
    # clears the gate. Ignored if the local model failed to load (a bad flag must
    # never strand every task with no answerer).
    local_only: bool = os.getenv("LOCAL_ONLY", "0").strip().lower() in ("1", "true", "yes")
    # Keep a local answer when confidence >= this; else escalate to Fireworks.
    escalate_threshold: float = float(os.getenv("ESCALATE_THRESHOLD", "0.60"))
    # httpx read timeout. 26s (was 14): a reasoning model's trace can legitimately
    # take >14s, and the OLD value timed those calls out -> empty answer -> wrong.
    # Read timeouts are no longer retried, so a single 26s call stays under the
    # <30s/task limit while giving slow generations room to finish.
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "26"))
    # Concurrency: route this many tasks through Fireworks at once. 3 (was 8):
    # competitor postmortems report the judging proxy rate-limits bursts (429s
    # with no backoff became fallback answers); 3 workers over ~19 tasks still
    # finishes in well under a minute while never stampeding the proxy.
    max_workers: int = int(os.getenv("MAX_WORKERS", "3"))
    # Per-task wall-clock budget for the model-fallback loop. A task may try up to 3
    # candidate models; this cap keeps the total per task under the <30s/task limit
    # (a model that fails fast — 5xx / error body — leaves plenty of room to try the
    # next; a slow-but-working model just answers on the first attempt).
    per_task_budget_s: float = float(os.getenv("PER_TASK_BUDGET_S", "28"))
    # Query {FIREWORKS_BASE_URL}/models at startup. Default OFF: no participant
    # reference uses /models on the judging proxy, ALLOWED_MODELS is authoritative
    # (verbatim entries; anything else risks MODEL_VIOLATION), and a proxy catalog
    # with different naming could displace the correct injected list. Even when ON,
    # discovery now only reorders/keeps the injected list — see main._resolve_models.
    model_discovery: bool = os.getenv("MODEL_DISCOVERY", "0").strip().lower() in ("1", "true", "yes")
    # DIAGNOSTIC ONLY: skip the free code-solvers so EVERY task goes to the model
    # (gated on a real key being present, so the offline self-test/CI smoke still
    # answer via solvers). Lets us tell "remote is fully broken in the sandbox"
    # (~0% score) from "remote works, something else is wrong" (high score).
    disable_solvers: bool = os.getenv("DISABLE_SOLVERS", "0").strip().lower() in ("1", "true", "yes")
    # Soft wall-clock budget: past this, remaining tasks skip local and go to
    # Fireworks (fast). main.py adds a HARD stop (+60s) that ends the loop and emits
    # empties, so a large/slow hidden set can never blow the 10-min budget (=ZERO).
    run_deadline_s: float = float(os.getenv("RUN_DEADLINE_S", "480"))
    # Baseline switch used by the eval harness: force every task to Fireworks.
    force_remote: bool = os.getenv("FORCE_REMOTE", "0").strip().lower() in ("1", "true", "yes")

    # Set by main._resolve_models when GET /models CONFIRMED which allowed models
    # the grader proxy actually serves. Routing may prefer instruct families only
    # on this proof — never on guesswork (gemma is unverifiable from outside).
    models_verified: bool = False

    # Raise every category's max_tokens ceiling to at least this. The per-category
    # ceilings are token-rank optimizations — but minimax THINKS in completion
    # tokens, and a hard task can legitimately need 2-4k of reasoning before the
    # answer. Naive agents with no cap let it finish and pass; our caps truncate
    # -> empty content -> salvage -> fail. Gate first, rank later: 0 disables.
    max_tokens_floor: int = int(os.getenv("MAX_TOKENS_FLOOR", "0"))

    def has_remote(self) -> bool:
        # Remote is usable if we can reach Fireworks at all. Gating this on the
        # model list alone meant a missing/renamed ALLOWED_MODELS silently routed
        # everything to the weak local path (zero API calls, failed accuracy gate).
        return bool(self.allowed_models or self.fireworks_api_key)


config = Config()
