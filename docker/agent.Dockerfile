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
# MAX_WORKERS=3 / REQUEST_TIMEOUT=22 / PER_TASK_BUDGET_S=27 / RUN_DEADLINE_S=420:
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
# DISABLE_SOLVERS=1 (gate-pass mode): EVERY task goes to the model. Seven verified
#   solver misfires were fixed and the score moved only 12/19 -> 13/19 — hidden-set
#   phrasings we cannot enumerate keep slipping through regex solvers as confident
#   wrong answers at 0 tokens. The model measured 94.8-95.8% end-to-end; ~2 extra
#   remote tasks cost ~500 tokens, qualification is worth everything. The flag is
#   key-gated (config.py), so the offline CI selftest/smoke still answer via solvers.
ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    REMOTE_FIRST=1 \
    USE_LOCAL=0 \
    DISABLE_SOLVERS=1 \
    LOCAL_ONLY=0 \
    REASONING_EFFORT= \
    REQUEST_TIMEOUT=22 \
    PER_TASK_BUDGET_S=27 \
    RUN_DEADLINE_S=420 \
    MAX_WORKERS=3 \
    MODEL_DISCOVERY=0

ENTRYPOINT ["python", "-m", "agent.main"]
