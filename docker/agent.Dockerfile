# Track-1 submission image — token-efficient agent (remote-first, gate-pass).
#
# Strategy: free deterministic solvers answer what they can prove (0 tokens);
# every remaining task goes to the cheapest allowed Fireworks model with a minimal
# prompt — the profile of all four gate-passing leaderboard agents (each 84.2%).
# The bundled local model is a DEAD-REMOTE RESCUE only: if the grader's Fireworks
# access fails, the router answers locally rather than emitting an empty string
# (a local answer is sometimes right; an empty one is always wrong). Local-first
# was tried and failed the accuracy gate (the 3B's format-only confidence gates
# keep wrong-but-well-formed answers). Grading box = 4GB RAM / 2 vCPU, so a 2-3B
# 4-bit model fits comfortably; the image stays well under the 10GB cap.
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

# REMOTE_FIRST=1 (gate-pass profile): every non-solver task goes to the gateway
#   model — the profile of all four gate-passing agents (each exactly 84.2%).
#   Every local-first image failed the gate (10.5% -> 26.3%): the 3B's format-only
#   confidence gates keep wrong-but-well-formed answers.
# USE_LOCAL=1: the bundled model is now ONLY a dead-remote rescue — if the
#   grader's Fireworks access fails, a local answer strictly beats an empty one.
# DISABLE_SOLVERS=0: exact solvers ON; they are prove-or-defer and red-teamed
#   (14 misfire classes fixed after the adversarial audit).
# LOCAL_THREADS=2: matches the 2-vCPU grading cgroup.
# REASONING_EFFORT= (empty = never sent): nonstandard field; the allowed gemma-4
#   models don't use it, and rejected calls double the proxy-side request count.
# MAX_WORKERS=3: don't stampede the judging proxy (429 bursts sank other teams).
# MODEL_DISCOVERY=0: ALLOWED_MODELS (injected) is authoritative and matched
#   VERBATIM by the proxy — never call /models-derived or prefixed ids.
# REMOTE_MODEL: preferred pick WITHIN the allowed list (short name, verbatim
#   launch-day id); non-reasoning gemma = cheapest tokens. If the harness injects
#   no list at all, config.fallback_models supplies the launch-day five verbatim.
# USE_LOCAL=0 (was 1): the bundled 3B was a DEAD-REMOTE RESCUE, but on the 2-vCPU
#   grading box 19 sequential CPU inferences blow the 10-min budget -> TIMEOUT
#   (observed) whenever remote 404s every call. Remote is now fast+serverless
#   (gpt-oss ~1s/call) with a serverless safety net, so the slow local tier is a
#   pure liability. A 404'd task now emits empty FAST (scoreable) instead of hanging.
# REQUEST_TIMEOUT=14 / PER_TASK_BUDGET_S=16 / RUN_DEADLINE_S=330: hard time bounds
#   so even a slow/hanging grader network can never exceed the 10-min / 30s-per-task
#   limits. main.py adds a +60s hard stop that emits empties for anything unfinished.
# MAX_WORKERS=5 (was 3): finish the batch faster in wall-clock; serverless gpt-oss
#   tolerates the concurrency (the 429 concern was a private proxy, not the API).
ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    REMOTE_FIRST=1 \
    USE_LOCAL=0 \
    DISABLE_SOLVERS=0 \
    LOCAL_ONLY=0 \
    LOCAL_MODEL_PATH=/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    LOCAL_THREADS=2 \
    REASONING_EFFORT= \
    REQUEST_TIMEOUT=14 \
    PER_TASK_BUDGET_S=16 \
    RUN_DEADLINE_S=330 \
    MAX_WORKERS=5 \
    MODEL_DISCOVERY=0 \
    REMOTE_MODEL=gpt-oss-120b

ENTRYPOINT ["python", "-m", "agent.main"]
