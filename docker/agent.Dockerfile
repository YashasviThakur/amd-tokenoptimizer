# Track-1 submission image — the Hybrid Token-Efficient Routing Agent.
# Bundles a small local model (via Ollama) so local inference is free at eval
# time. Build for the judging VM's architecture:
#
#   docker buildx build --platform linux/amd64 \
#     -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest --push .
#
# The harness injects FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS
# and mounts /input + /output. We only read those from the environment.
FROM python:3.11-slim

# Links this GHCR image to the GitHub repo (shows under the repo's Packages) so
# the judging harness can discover it from the repository URL.
LABEL org.opencontainers.image.source="https://github.com/YashasviThakur/amd-tokenoptimizer" \
      org.opencontainers.image.description="AMD ACT II Track 1 - Hybrid Token-Efficient Routing Agent" \
      org.opencontainers.image.licenses="MIT"

# curl + ca-certificates kept (entrypoint health check uses curl); zstd is
# required by the ollama installer to extract its archive.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates zstd \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY agent/requirements.txt ./agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# Local model baked into the image (free tokens). Sized to run on the CPU VM.
# Code is the routing bottleneck, so a code-capable ~3B model gives the biggest
# token cut (94% in eval vs 87% for a generic 2B). Alternatives: gemma2:2b
# (lightest), gemma3:4b, qwen2.5:3b.
ENV LOCAL_MODEL=qwen2.5-coder:3b
RUN set -e; \
    ollama serve >/tmp/ollama.log 2>&1 & \
    for i in $(seq 1 30); do curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done; \
    ollama pull "$LOCAL_MODEL"; \
    pkill ollama || true

COPY agent ./agent
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LOCAL_BASE_URL=http://localhost:11434/v1 \
    LOCAL_API_KEY=ollama \
    INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    ESCALATE_THRESHOLD=0.60 \
    REASONING_EFFORT=low \
    REMOTE_MODEL=accounts/fireworks/models/gemma-4-31b-it

ENTRYPOINT ["/entrypoint.sh"]
