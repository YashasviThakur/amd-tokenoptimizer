# TokenOptimizer — Token-Efficient Agent

**AMD Developer Hackathon: ACT II · Track 1**

A batch AI agent that completes a fixed set of tasks using the **fewest Fireworks
tokens possible**. It answers as much as it can for **zero Fireworks tokens** —
with **plain deterministic code** and a **bundled local model** — and calls the
cheapest allowed Fireworks model only for the tasks a small local model can't be
trusted on.

## 🐳 Submission image (public, `linux/amd64`)

```
ghcr.io/yashasvithakur/tokenoptimizer-agent:latest
```
```bash
docker pull ghcr.io/yashasvithakur/tokenoptimizer-agent:latest
```
Built and pushed by [`.github/workflows/build.yml`](.github/workflows/build.yml) and
linked to this repo under **Packages** (OCI `image.source` label). Entrypoint
`python -m agent.main`: reads `/input/tasks.json` → writes `/output/results.json` →
exits 0. Bundles a 2-3B 4-bit local model (fits the 4 GB / 2 vCPU grading box);
image stays well under the 10 GB cap.

## How Track 1 is scored (and how we win it)

The judging harness runs the container headless: it injects `FIREWORKS_API_KEY` /
`FIREWORKS_BASE_URL` / `ALLOWED_MODELS`, mounts `/input/tasks.json`, and you write
`/output/results.json` and exit 0. Then:

1. **Accuracy gate** — an LLM-judge grades every answer. Below the threshold →
   **excluded from the leaderboard.**
2. **Token ranking** — passing submissions are ranked **ascending by total tokens
   through `FIREWORKS_BASE_URL`.** Fewer tokens = higher rank.
3. **Local inference is free.** Per the rules, a bundled local model's answers count
   fully toward accuracy but **not** toward the token score — so a task answered
   locally (or by code) costs **0 Fireworks tokens**, the best possible outcome.

So the game is: **answer as much as possible for free (code + local model), and for
the rest, spend the fewest Fireworks tokens per task.**

```
per task ──▶ classify category (free, heuristic)
          ──▶ deterministic solver? (arithmetic / % / ordering / syllogism …)
                 yes → answer with plain code ................... 0 tokens, exact
          ──▶ category a small local model handles well?
                 (factual / sentiment / summary / NER)
                 yes → answer with the bundled local model ...... 0 tokens
                       kept only if a free verifier + confidence check pass
          ──▶ otherwise (hard math / logic / code, or low confidence)
                 → cheapest capable Fireworks model, minimal prompt
```

The **local model** (Qwen2.5-3B-Instruct, Q4, via llama-cpp-python on CPU) answers
the categories a small model is reliable on — measured **100% on factual &
summarisation, 93% sentiment** on our unseen set — for zero tokens. The
**deterministic solvers** (exact arithmetic / % / powers / ordering / syllogism)
answer whole task-types for zero tokens and are red-teamed to **prove-or-defer**
(never emit a wrong answer). Everything a 3B model gets wrong — multi-step math,
constraint puzzles, code — **escalates to Fireworks** with a ~3-token system prompt,
so accuracy stays high while very few tasks cost tokens.

## Measured result (local eval harness)

Local + solver answers cost **0 tokens**; only escalations hit Fireworks.

| Set | Local-only accuracy (0 tokens) | With Fireworks escalation |
|-----|:------------------------------:|:-------------------------:|
| dev (32 tasks)            | **97%** | 97% |
| unseen/adversarial (96)   | **83%** | ~93% (est.) |

Fully local clears an 80% gate on both sets at **zero Fireworks tokens**; escalating
just the weak categories (math / code_debug) lifts accuracy to a comfortable margin
while sending only **~40%** of tasks to Fireworks. Reproduce the local measurement:

```bash
python -m eval.harness    # baseline vs agent (mock Fireworks + free code path)
```

## Run the agent

```bash
cp agent/.env.example .env                # FIREWORKS_API_KEY, ALLOWED_MODELS
# LOCAL_MODEL_PATH points at a bundled GGUF; USE_LOCAL=0 disables the local tier
INPUT_PATH=eval/datasets/practice_tasks.json OUTPUT_PATH=out/results.json \
  python -m agent.main
```

## Build & submit (Docker)

The image bundles the local model weights (downloaded at build time) plus a
`llama-cpp-python` CPU runtime **built from source with a portable AVX2 baseline**
(`GGML_NATIVE=OFF`, OpenMP off) so it loads on any x86-64 grading VM.

```bash
docker buildx build --platform linux/amd64 \
  -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest --push .
```
CI does this automatically on push to `main`; a branch workflow
(`hybrid-validate.yml`) builds and benchmarks the local model on a linux runner
without touching `:latest`.

## Robustness & validation

- **Solvers prove-or-defer** and the local model is kept only when a free verifier
  (valid number / JSON / label / length) + a confidence check pass — otherwise the
  task escalates. **Zero solver misfires** across dev + 96 unseen tasks.
- **Never a zero-score crash / timeout.** Atomic `results.json` write, a top-level
  guard that always exits 0 with valid JSON, defensive input parsing, and a **soft
  wall-clock deadline** that flips remaining tasks to Fireworks (fast) so slow CPU
  inference can never blow the 10-min budget.
- **Container self-test:** `python -m agent.main --selftest` validates the solver +
  output contract offline; `hybrid-validate.yml` asserts the local model actually
  loads and answers on linux/amd64.

## Rules we honor

- Reads `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` from the
  environment — never hardcoded, never a bundled `.env`.
- **All** remote calls go through `FIREWORKS_BASE_URL`; only `ALLOWED_MODELS` are
  used (read at runtime). Local inference is bundled and free; it never calls out.
- No cached/hardcoded answers — routing is decided live per task from the prompt.
- Always writes valid JSON and exits 0. Ready < 60s, < 30s per task, ≤ 10 min total,
  image < 10 GB.

## Layout

```
agent/
  main.py         /input/tasks.json → route each task → /output/results.json
  router.py       solver → local model → Fireworks, with confidence + deadline
  classifier.py   free 8-category classifier
  solvers.py      free deterministic solvers (0-token exact answers)
  verifiers.py    free checks + exact arithmetic calculator
  backends.py     LocalModel (llama-cpp, free) + Fireworks Model (metered)
  prompts.py      token-minimal prompts + per-category caps
eval/             mock-server harness, judge, labeled dev/stress/practice datasets
docker/agent.Dockerfile   the submission image (agent + bundled local model)
```

## License

MIT
