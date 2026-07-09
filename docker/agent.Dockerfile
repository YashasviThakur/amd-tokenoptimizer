# Track-1 submission image — token-efficient agent (Fireworks-only).
#
# The agent answers as much as possible for 0 tokens with plain deterministic
# code solvers, and routes everything else to the cheapest allowed Fireworks
# model. It does NOT use a bundled local model (USE_LOCAL=0), so this image ships
# ONLY the tiny Python agent — no 2GB GGUF, no llama-cpp, no build toolchain.
# Result: a ~150MB image that pulls in seconds (the previous 2.3GB image, whose
# bulk was an unused model layer, risked slow-pull failures on the grading box).
#
#   docker build -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest .
#
# The harness injects FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS and
# mounts /input + /output. We only read those from the environment.
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/YashasviThakur/amd-tokenoptimizer" \
      org.opencontainers.image.description="AMD ACT II Track 1 - token-efficient code+Fireworks agent" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Runtime deps only: httpx (Fireworks client) + tiktoken (local token estimate).
COPY agent/requirements.txt ./agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# Pre-bake the tiktoken vocab so the diagnostic token counter never hits the network.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY agent ./agent

# USE_LOCAL=0: answer via exact code solvers (free) + Fireworks (accurate). The
# bundled local model was both failing the accuracy gate AND bloating the image;
# it is no longer shipped. REMOTE_MODEL is the fallback if the harness injects no
# ALLOWED_MODELS list; gpt-oss-120b measured clean + cheap.
ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    USE_LOCAL=0 \
    REASONING_EFFORT=low \
    REMOTE_MODEL=accounts/fireworks/models/gpt-oss-120b

ENTRYPOINT ["python", "-m", "agent.main"]
