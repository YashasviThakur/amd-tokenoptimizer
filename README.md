# TokenOptimizer — Token-Efficient Agent

**AMD Developer Hackathon: ACT II · Track 1**

A batch AI agent that completes a fixed set of tasks using the **fewest Fireworks
tokens possible** — answering with **plain deterministic code** wherever it can
(zero tokens) and calling the **cheapest allowed Fireworks model** only when a
task genuinely needs an LLM.

## 🐳 Submission image (public, `linux/amd64`)

```
ghcr.io/yashasvithakur/tokenoptimizer-agent:latest
```
```bash
docker pull ghcr.io/yashasvithakur/tokenoptimizer-agent:latest
```
Built and pushed automatically by [`.github/workflows/build.yml`](.github/workflows/build.yml);
linked to this repo under **Packages**. Reads `/input/tasks.json` → writes
`/output/results.json`.

## How Track 1 is actually scored (and how we win it)

The judging harness runs your container headless: it mounts `/input/tasks.json`,
you write `/output/results.json`, exit 0. Then:

1. **Accuracy gate** — an LLM-judge grades every answer. Below the threshold →
   **excluded from the leaderboard.**
2. **Token ranking** — passing submissions are ranked **ascending by total tokens
   through `FIREWORKS_BASE_URL`.** Fewer tokens = higher rank.
3. **Only Fireworks calls are scored.** Per the organizers there is **no local
   LLM** — each task is answered either by **plain deterministic code (0 tokens)**
   or by a **Fireworks call** to an `ALLOWED_MODELS` model.

So the game is: **answer as much as possible with deterministic code, and for the
rest, spend the fewest Fireworks tokens per task.**

```
per task ──▶ classify category (free, heuristic)
          ──▶ deterministic solver? (arithmetic / % / average / ordering / syllogism …)
                 yes → answer with plain code ................... 0 tokens, exact
                 no  → cheapest capable Fireworks model,
                       minimal prompt · max_tokens cap · reasoning_effort=low
```

The **deterministic solvers** are the only free path — an exact arithmetic /
percentage / discount / speed / average calculator, a transitive-ordering solver
for comparative puzzles, and a syllogism checker — each answering whole task
types for **zero tokens**, and red-teamed to never emit a wrong answer (they
defer to Fireworks when unsure). For everything else, each Fireworks call is
minimized: a ~3-token system prompt, compressed input, a per-category
`max_tokens` cap, `reasoning_effort=low`, and cheapest-model selection.

## Measured result (local eval harness, 32 tasks across all 8 categories)

Every task the deterministic solvers handle costs **0 tokens**; the rest go to
Fireworks. Savings therefore come from solver coverage + minimal Fireworks calls:

| Config | Accuracy | Fireworks tokens | Cut vs baseline |
|--------|:--------:|:----------------:|:---------------:|
| baseline — every task → Fireworks | 96.9% | 2164 | — |
| **code + Fireworks (compliant)**  | **96.9%** | **1429** | **34%** |

Accuracy stays high because anything code can't solve goes to a strong Fireworks
model. Reproduce:

```bash
python -m eval.harness
```

The token cut scales directly with how many tasks plain code can answer — so the
lever is **broadening the deterministic solvers** (and minimizing tokens per
Fireworks call: `reasoning_effort=low`, tight `max_tokens`, cheapest model,
batching).

## Run the eval today (no GPU, no keys)

```bash
pip install -r requirements.txt         # fastapi/uvicorn/httpx/numpy for the harness
python -m eval.harness                   # baseline vs hybrid
python -m eval.harness --sweep           # threshold sweep — pick the operating point
```

The harness boots a mock model server (a weak local model + a strong Fireworks
model with realistic token usage), runs the **real agent code** against it, grades
with a local judge, and prints token savings. Swap the base URLs for real
endpoints and nothing in the agent changes.

## Run the agent for real

```bash
# 1) a local OpenAI-compatible model (free tokens)
ollama serve && ollama pull gemma2:2b        # or qwen2.5:3b, llama.cpp, vLLM…

# 2) env (the harness injects these at eval time; use your own key for dev)
cp agent/.env.example .env                    # fill FIREWORKS_API_KEY, ALLOWED_MODELS

# 3) run the batch
INPUT_PATH=eval/datasets/dev_tasks.json OUTPUT_PATH=out/results.json \
  python -m agent.main
```

## Submit (Docker)

The image bakes the local model in, so local inference is free at eval time.

```bash
docker buildx build --platform linux/amd64 \
  -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest --push .
```

The judging VM is `linux/amd64` — build for it or the image won't pull (scores
zero). Image stays well under the 10GB cap (gemma2:2b ≈ 1.6GB). See
[`docker/agent.Dockerfile`](docker/agent.Dockerfile) and
[`entrypoint.sh`](docker/entrypoint.sh).

## Robustness & validation

Malformed output or a wrong "free" answer both cost the win, so the agent is
hardened and continuously checked:

- **Solvers prove-or-defer.** A red-team pass (41 findings) drove `solve_ordering`
  and `solve_syllogism` to reject partial/disconnected orders, cross-dimension
  comparisons, cycles/contradictions, and unprovable syllogisms — returning
  `None` (escalate) rather than a possibly-wrong answer. **Zero solver misfires
  across 128 tasks** (dev + 96 unseen).
- **Bulletproof I/O.** Defensive `tasks.json` parsing (list / `{"tasks":[]}` /
  id-map / single / bare-string / alt keys), answers coerced to strings, output
  always valid JSON, always exit 0, and one retry on a transient Fireworks 5xx.
- **Offline container self-test:** `python -m agent.main --selftest` runs
  solver-answerable tasks with the network disabled and validates the output
  contract — a fast health check for the image.
- **Guardrails in CI-style scripts:** `python -m eval.stress_solvers` asserts a
  solver never fires with a wrong answer and reports classifier accuracy
  (95/96 on the held-out set).

## Headroom — where the optimum is, and what's left

The theoretical floor for this scoring (rank by fewest Fireworks tokens, local =
0) is the **irreducible-remote set**: the tasks only the frontier model gets
right. We approach it but can't hit zero — the accuracy gate needs the frontier
model for genuinely hard tasks, and we keep a safety margin because the exact
gate and ground truth aren't known at runtime. Remaining levers, by ROI:

- **Local code-execution verification** *(biggest — code is our largest remote
  sink)*: run the local model's code against derived tests; keep it if it passes
  (0 tokens). Needs the real local model to tune.
- **Operate at gate + ε**: we run ~14 pts above an 80% gate; a global optimizer
  that escalates only the least-confident tasks until predicted accuracy just
  clears the gate spends fewer tokens (traded against gate-miss risk).
- **A stronger/code-specialized local model**: the dominant real-world lever —
  every task it answers correctly is free (see the 87% → 94% jump above).
- **Request batching** (amortize per-message overhead) and **cheapest-concise
  model selection** from `ALLOWED_MODELS` (score is token *count*, not price).
- **More free solvers** (averages/sums shipped; units/dates/counting/boolean are
  pure upside but only help if the eval contains such tasks).

## Rules we honor

- Reads `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` from the
  environment — never hardcoded, never a bundled `.env`.
- **All** remote calls go through `FIREWORKS_BASE_URL`; only `ALLOWED_MODELS` are
  used (read at runtime).
- No cached/hardcoded answers — routing is decided live per task.
- Always writes valid JSON and exits 0; every task is wrapped so one failure
  can't sink the batch. Ready < 60s, < 30s per task, ≤ 10 min total.

## Layout

```
agent/                 THE deliverable — batch routing agent (goes in the image)
  main.py              /input/tasks.json → route → /output/results.json
  router.py            confidence-based, category-aware routing policy
  classifier.py        free 8-category classifier
  verifiers.py         free checks + exact arithmetic calculator
  prompts.py           token-minimal prompts + per-category caps
  backends.py          local (free) + Fireworks (metered) OpenAI-compatible clients
eval/
  harness.py           baseline vs hybrid, threshold sweep, token report (--profile, --tasks)
  stress_solvers.py    asserts solvers never fire a wrong answer + classifier accuracy
  judge.py             local accuracy judge (LLM-judge stand-in)
  mock_server.py       simulated local + Fireworks endpoints (generic/strong local profile)
  datasets/            dev (32) + stress (96 unseen, adversarial) labeled tasks
docker/                submission image (agent + baked local model)
web/                   optional: a live cost-cockpit dashboard (dev visualizer only —
                       not used for Track-1 scoring; see src/tokenoptimizer)
```

> Note: `src/tokenoptimizer/` + `web/` are an earlier interactive **gateway +
> cost dashboard**. They're not part of Track-1 scoring (which is headless), but
> they're a useful local visualizer and a ready base for a Track-3 "Unicorn"
> entry. The Track-1 winner is `agent/` + `eval/`.

## License

MIT
