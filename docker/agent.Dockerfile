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

# Build llama-cpp-python (CPU) FROM SOURCE with a portable AVX2 baseline
# (GGML_NATIVE=OFF). Prebuilt wheels are risky: the musllinux wheel won't load on
# glibc slim, and a -march=native / AVX-512 wheel can pass on the build CPU but
# crash with an illegal instruction on the grading VM. An explicit AVX2/FMA/F16C
# build runs on any modern x86-64 (universal on cloud) and links glibc.
# GGML_OPENMP=OFF -> llama.cpp uses its own pthread pool, so the compiled .so has
# no libgomp runtime dependency. Build tools are kept (only ~400MB; image stays
# well under 10GB) so every runtime lib the .so needs is present.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
 && CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=OFF" \
      pip install --no-cache-dir "llama-cpp-python==0.3.2" \
 && pip install --no-cache-dir "huggingface_hub>=0.23" \
 && rm -rf /var/lib/apt/lists/*

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

# USE_LOCAL=0: the bundled 3B model was failing the ACCURACY gate on the hidden
# set (excluded from the leaderboard). Answer via exact code solvers + Fireworks
# so we clear the gate; flip back to 1 only once the local model is proven.
ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    USE_LOCAL=0 \
    LOCAL_MODEL_PATH=/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    LOCAL_THREADS=2 \
    LOCAL_SAMPLES_HARD=1 \
    REASONING_EFFORT=low \
    REMOTE_MODEL=accounts/fireworks/models/gemma-4-31b-it

ENTRYPOINT ["python", "-m", "agent.main"]
