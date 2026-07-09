# Track-1 submission image — token-efficient hybrid agent (local-first).
#
# Strategy: answer as much as possible for ZERO Fireworks tokens — with plain
# deterministic code solvers AND a bundled local model — and use Fireworks only as
# an optional escalation for the hard tasks a small model can't be trusted on.
# Local inference is free (0 tokens = best score) AND immune to any Fireworks
# access/credit problems in the grading sandbox: if the injected Fireworks key is
# dead, the local model still answers (the router keeps the local answer rather
# than emitting an empty one). Grading box = 4GB RAM / 2 vCPU, so a 2-3B 4-bit
# model fits comfortably; the image stays well under the 10GB cap.
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

# USE_LOCAL=1: the local model carries the categories it's reliable on (factual,
#   sentiment, summarization, ner) for 0 tokens; solvers handle exact math/logic;
#   Fireworks is only an escalation and, if it's dead in the grader, the router
#   keeps the local answer instead of an empty one.
# DISABLE_SOLVERS=0: solvers ON (the earlier =1 was a temporary diagnostic).
# LOCAL_THREADS=2: matches the 2-vCPU grading cgroup. 0 (= all cores) reads the
#   HOST's core count through the cgroup and oversubscribes, slowing inference.
# LOCAL_SAMPLES_HARD=2: factual self-consistency — two agreeing draws are kept
#   locally (0 tokens); a lone or disagreeing draw still escalates (gate-safe).
# LOCAL_ONLY=0: flip to 1 for the ZERO-token mode (never call Fireworks) once the
#   leaderboard confirms local-only accuracy clears the gate. Do not flip blind.
# REMOTE_MODEL is the escalation fallback if the harness injects no ALLOWED_MODELS.
ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    USE_LOCAL=1 \
    DISABLE_SOLVERS=0 \
    LOCAL_ONLY=0 \
    LOCAL_MODEL_PATH=/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    LOCAL_THREADS=2 \
    LOCAL_SAMPLES_HARD=2 \
    REASONING_EFFORT=low \
    REMOTE_MODEL=accounts/fireworks/models/gemma-4-31b-it

ENTRYPOINT ["python", "-m", "agent.main"]
