# Track-1 submission image — token-efficient hybrid agent.
# Per the (updated) rules, local-model inference is FREE (0 Fireworks tokens) and
# the best possible token score, so we bundle a small CPU model and answer as much
# as possible locally, escalating only the hard tasks to the cheapest Fireworks
# model. Grading box = 4GB RAM / 2 vCPU, so a 2-3B 4-bit model fits.
#
#   docker buildx build --platform linux/amd64 \
#     -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest --push .
#
# The harness injects FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS and
# mounts /input + /output. We only read those from the environment.
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/YashasviThakur/amd-tokenoptimizer" \
      org.opencontainers.image.description="AMD ACT II Track 1 - hybrid local+Fireworks token-efficient agent" \
      org.opencontainers.image.licenses="MIT"

WORKDIR /app

# llama-cpp-python (CPU). Pin 0.2.90 — a conservative instruction baseline that
# loads on any x86-64 (newer wheels use AVX-512 and crash on some CPUs) and whose
# bundled llama.cpp supports Qwen2.5. Installed from the CPU wheel index.
RUN pip install --no-cache-dir "llama-cpp-python==0.2.90" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
 && pip install --no-cache-dir "huggingface_hub>=0.23"

# Bundle the local model weights in the image (downloaded at build time — CI has
# fast HF network; the grading box never downloads). ~1.9GB, well under the 10GB cap.
RUN python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('Qwen/Qwen2.5-3B-Instruct-GGUF','qwen2.5-3b-instruct-q4_k_m.gguf', local_dir='/models')"

COPY agent/requirements.txt ./agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# Pre-bake the tiktoken vocab so the diagnostic token counter never hits the network.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY agent ./agent

ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    USE_LOCAL=1 \
    LOCAL_MODEL_PATH=/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    LOCAL_THREADS=0 \
    REASONING_EFFORT=low \
    REMOTE_MODEL=accounts/fireworks/models/gemma-4-31b-it

ENTRYPOINT ["python", "-m", "agent.main"]
