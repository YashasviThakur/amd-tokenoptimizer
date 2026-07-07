# Track-1 submission image — General-Purpose AI Agent (rules-compliant).
# Per the organizers, only Fireworks calls are scored and there is NO local LLM:
# each task is answered by plain deterministic CODE (0 tokens) or a Fireworks call
# to the cheapest ALLOWED_MODELS model. So this image is tiny — just Python + the
# agent, no bundled model.
#
#   docker buildx build --platform linux/amd64 \
#     -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest --push .
#
# The harness injects FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS and
# mounts /input + /output. We only read those from the environment.
FROM python:3.11-slim

# Links this GHCR image to the GitHub repo (shows under the repo's Packages) so
# the judging harness can discover it from the repository URL.
LABEL org.opencontainers.image.source="https://github.com/YashasviThakur/amd-tokenoptimizer" \
      org.opencontainers.image.description="AMD ACT II Track 1 - General-Purpose Token-Efficient Agent" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY agent/requirements.txt ./agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# Pre-bake the tiktoken vocab so the diagnostic token counter never hits the
# network at container start (a hung download could eat the readiness window).
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY agent ./agent

ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    REASONING_EFFORT=low \
    REMOTE_MODEL=accounts/fireworks/models/gemma-4-31b-it

ENTRYPOINT ["python", "-m", "agent.main"]
