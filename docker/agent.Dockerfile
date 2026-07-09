# Track-1 submission image — SLIM remote+solver agent (no bundled local model).
#
# Strategy: free deterministic solvers answer what they can prove (0 tokens); every
# remaining task goes to a SERVERLESS Fireworks model (gpt-oss, always-on, no
# deployment) with a minimal prompt. No local model is bundled: the earlier hybrid
# image was ~2.3GB (2GB model + llama-cpp) and, on the 2-vCPU grading box, 19
# sequential CPU inferences blew the 10-min budget -> TIMEOUT whenever the grader's
# injected on-demand models 404'd every remote call. This image is ~150MB: fast to
# pull, instant to start, and has NO slow path anywhere (pull, ready, or runtime).
# A task the model can't answer emits empty FAST (scoreable) rather than hanging.
#
#   docker build -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest .
#
# The harness injects FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS and
# mounts /input + /output. We only read those from the environment.
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/YashasviThakur/amd-tokenoptimizer" \
      org.opencontainers.image.description="AMD ACT II Track 1 - slim token-efficient remote+solver agent" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

COPY agent/requirements.txt ./agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# Pre-bake the tiktoken vocab so the diagnostic token counter never hits the network.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY agent ./agent

# REMOTE_FIRST=1: free solvers first (0 tokens), everything else to Fireworks.
# USE_LOCAL=0: no bundled model — remote is serverless+fast; the slow CPU tier was
#   a pure liability (TIMEOUT). A 404'd task emits empty fast instead of hanging.
# The router calls ONLY the harness-injected ALLOWED_MODELS, each VERBATIM: the
#   judging proxy matches ids exactly, so any off-list string (a re-spelled id or an
#   always-on model not on the list) makes the whole submission a MODEL_VIOLATION.
# REQUEST_TIMEOUT=14 / PER_TASK_BUDGET_S=16 / RUN_DEADLINE_S=300 / MAX_WORKERS=5:
#   hard time bounds so a slow/hanging grader network can never exceed the 10-min /
#   30s-per-task limits; main.py adds a +60s hard stop that emits empties for any
#   unfinished task and always writes a valid results.json.
# REMOTE_MODEL=gpt-oss-120b: preferred order only — honored solely if it appears
#   VERBATIM in the injected ALLOWED_MODELS, otherwise silently ignored (never sent).
ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    REMOTE_FIRST=1 \
    USE_LOCAL=0 \
    DISABLE_SOLVERS=0 \
    LOCAL_ONLY=0 \
    REASONING_EFFORT= \
    REQUEST_TIMEOUT=14 \
    PER_TASK_BUDGET_S=16 \
    RUN_DEADLINE_S=300 \
    MAX_WORKERS=5 \
    MODEL_DISCOVERY=0 \
    REMOTE_MODEL=gpt-oss-120b

ENTRYPOINT ["python", "-m", "agent.main"]
