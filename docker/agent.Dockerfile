# Track-1 submission image — LEAN solvers+remote token-efficient agent.
#
# Strategy: free deterministic solvers answer what they can prove (0 tokens);
# everything else goes to the harness-injected Fireworks models with tight,
# format-locked prompts. SHIP 18 drops the bundled 3.3GB local model: every
# model-bundling ship (13/15/16/17) died in the grader's PULL window — the Jul 12
# forensics show TIMEOUT with evaluationStartedAt=null, then PULL_ERROR on the
# auto-retry — i.e. the box never finished pulling/extracting the 8x410MB model
# layers, the agent itself never ran. A few-hundred-MB image pulls in seconds;
# a scored solvers+remote run strictly dominates an unranked TIMEOUT.
#
#   docker buildx build --platform linux/amd64 \
#     -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest --push .
#
# The harness injects FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS and
# mounts /input + /output. We only read those from the environment.
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/YashasviThakur/amd-tokenoptimizer" \
      org.opencontainers.image.description="AMD ACT II Track 1 - lean solvers+Fireworks token-efficient agent" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY agent/requirements.txt ./agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# Pre-bake the tiktoken vocab so the diagnostic token counter never hits the network.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY agent ./agent

# REMOTE_FIRST=0: free solvers first (0 tokens), then Fireworks for everything
#   else + low-confidence + near-deadline escalations.
# USE_LOCAL=0: no bundled model in this image — main.run() gets local=None and the
#   router never attempts a local rung (same measured code path as the selftest).
# The router calls ONLY the harness-injected ALLOWED_MODELS, each VERBATIM: the
#   judging proxy matches ids exactly, so any off-list string (a re-spelled id or an
#   always-on model not on the list) makes the whole submission a MODEL_VIOLATION.
# MAX_WORKERS=3 / REQUEST_TIMEOUT=25 / PER_TASK_BUDGET_S=28 / RUN_DEADLINE_S=420:
#   (STABILITY: a measured flaky task timed out at the old 22s and answered in 2s
#   on retry — slow reasoning generations need the headroom; 25+fast-404 failover
#   ~= 28 < the 30s/task limit.)
#   RELIABILITY over raw speed. The judging proxy rate-limits concurrent bursts
#   (429 -> after one backoff the task fails over / empties -> WRONG); 3 workers is
#   the measured burst the proxy tolerates (5 regressed it). The allowed models are
#   REASONING models whose trace can legitimately run >14s on the grader's box, and
#   a ReadTimeout is NOT retried -> an empty (wrong) answer; 22s (still <30s/task)
#   lets them finish, and a 27s per-task budget leaves room for a fast 404 failover
#   before the real call. Time is still bounded: ~19 tasks / 3 workers x 27s well
#   under 10min, and main.py's +60s hard stop (=480s) emits empties for any
#   straggler and always writes a valid results.json (a slow set scores, never TIMEOUTs).
# NO REMOTE_MODEL / ALLOWED_MODELS baked in: the router must call ONLY the models the
#   grader injects, VERBATIM. Baking a preferred id (gpt-oss-120b) merged it into the
#   resolved allow-list and got it called first -> MODEL_VIOLATION when the grader's
#   list didn't include it. The grader supplies the model list at eval; we never guess.
# MODEL_DISCOVERY=1 (SHIP 19): GET /models cross-check at startup. Safe by
#   construction (main._resolve_models): it only NARROWS the injected list to the
#   served intersection, or adopts the served list when NOTHING was injected — it
#   never adds an off-list id to a configured list, so no MODEL_VIOLATION surface.
#   Rescues the all-404 scenario (injected ids not deployed on the proxy).
# DISABLE_SOLVERS=0: MEASURED on the hidden set — the all-remote experiment scored
#   12/19 where the same code with solvers ON scored 13/19: the (misfire-fixed)
#   solvers win at least one task the model fumbles. Keep them.
ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    REMOTE_FIRST=0 \
    USE_LOCAL=0 \
    DISABLE_SOLVERS=0 \
    LOCAL_ONLY=0 \
    REASONING_EFFORT=low \
    REQUEST_TIMEOUT=25 \
    PER_TASK_BUDGET_S=28 \
    RUN_DEADLINE_S=360 \
    MAX_WORKERS=3 \
    MODEL_DISCOVERY=1 \
    MAX_TOKENS_FLOOR=2048 \
    FORCE_INSTRUCT_FIRST=1 \
    THINKING_OFF_SOFT=1 \
    THINKING_OFF_ALL=1 \
    ENABLE_BATCHING=1 \
    BATCH_CATEGORIES=factual,ner,sentiment

# Direct entrypoint — the agent process is READY in seconds and pre-seeds a valid
# results.json immediately (SIGKILL-proof); with USE_LOCAL=0 the lazy model loader
# is skipped entirely and routing is solvers -> Fireworks.
ENTRYPOINT ["python", "-m", "agent.main"]
