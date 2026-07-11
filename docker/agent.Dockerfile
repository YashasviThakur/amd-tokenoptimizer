# Track-1 submission image — HYBRID local+remote token-efficient agent.
#
# Strategy: free deterministic solvers answer what they can prove (0 tokens); a
# bundled fine-tuned Qwen2.5-3B (Q8_0 GGUF, llama-cpp-python CPU) answers the
# categories it's reliable on — factual / sentiment / summarization — for 0 tokens;
# every other category (ner / code / leftover math+logic) and every low-confidence
# or near-deadline task escalates to a Fireworks model. The earlier hybrid TIMEOUT'd
# because routing could send all ~19 tasks to the slow serial CPU model; that is now
# bounded two ways — only three categories ever attempt local (router.LOCAL_OK), and
# past RUN_DEADLINE_S main.py flips the remaining tasks straight to Fireworks
# (prefer_remote), so a slow/large set can never blow the 10-min budget.
#
#   docker buildx build --platform linux/amd64 \
#     -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest \
#     --build-arg HF_GGUF_REPO=<you>/tokenopt-3b-gguf --push .
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

# Bundle the fine-tuned local model (Q8_0, downloaded at build from HF). Q8 PRESERVES
# accuracy — Q4/Q6 degrade this fine-tuned 3B to ~80% (measured). Ship #1 proved Q8
# FITS + RUNS in the grader (it returned an accuracy score 14/19, not a crash/timeout);
# the 2 lost tasks were factual hallucinations, now kept REMOTE (LOCAL_OK excludes it).
ARG HF_GGUF_REPO=yashasvithakur/tokenopt-3b-gguf
RUN python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('${HF_GGUF_REPO}','tokenopt-3b-q8_0.gguf', local_dir='/models')"

COPY agent/requirements.txt ./agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# Pre-bake the tiktoken vocab so the diagnostic token counter never hits the network.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY agent ./agent

# REMOTE_FIRST=0: free solvers first (0 tokens), then the bundled local model for
#   the categories it's reliable on (factual/sentiment/summarization), then Fireworks
#   for everything else + low-confidence + near-deadline escalations.
# USE_LOCAL=1 / LOCAL_MODEL_PATH: bundle the fine-tuned Qwen2.5-3B Q8_0 GGUF and answer
#   easy categories for 0 tokens. The old TIMEOUT is bounded now: only three categories
#   ever go local (router.LOCAL_OK) and past RUN_DEADLINE_S main.py flips the rest
#   straight to Fireworks (prefer_remote).
# LOCAL_SAMPLES_HARD=2: self-consistency — factual & sentiment keep a local answer only
#   when two draws AGREE; a lone/disagreeing draw escalates (gate-safe).
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
# DISABLE_SOLVERS=0: MEASURED on the hidden set — the all-remote experiment scored
#   12/19 where the same code with solvers ON scored 13/19: the (misfire-fixed)
#   solvers win at least one task the model fumbles. Keep them.
ENV INPUT_PATH=/input/tasks.json \
    OUTPUT_PATH=/output/results.json \
    REMOTE_FIRST=0 \
    USE_LOCAL=1 \
    LOCAL_MODEL_PATH=/models/tokenopt-3b-q8_0.gguf \
    LOCAL_SAMPLES_HARD=2 \
    DISABLE_SOLVERS=0 \
    LOCAL_ONLY=0 \
    REASONING_EFFORT= \
    REQUEST_TIMEOUT=25 \
    PER_TASK_BUDGET_S=28 \
    RUN_DEADLINE_S=360 \
    MAX_WORKERS=3 \
    MODEL_DISCOVERY=0 \
    MAX_TOKENS_FLOOR=2048 \
    FORCE_INSTRUCT_FIRST=1 \
    THINKING_OFF_SOFT=1 \
    THINKING_OFF_ALL=1 \
    ENABLE_BATCHING=1 \
    BATCH_CATEGORIES=factual \
    LOCAL_CODE_MAX_TOKENS=96

ENTRYPOINT ["python", "-m", "agent.main"]
