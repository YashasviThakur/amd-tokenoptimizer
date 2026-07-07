# TokenOptimizer — Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II · Track 1**

A batch AI agent that completes a fixed set of tasks using the **fewest Fireworks
tokens possible** — answering everything it safely can on a free local model and
escalating to Fireworks AI only when a local answer can't be trusted.

## How Track 1 is actually scored (and how we win it)

The judging harness runs your container headless: it mounts `/input/tasks.json`,
you write `/output/results.json`, exit 0. Then:

1. **Accuracy gate** — an LLM-judge grades every answer. Below the threshold →
   **excluded from the leaderboard.**
2. **Token ranking** — passing submissions are ranked **ascending by total tokens
   through `FIREWORKS_BASE_URL`.** Fewer tokens = higher rank.
3. **Local tokens count as zero.**

So the entire game is a constrained optimization:

> **Answer as many tasks as possible on the free local model; escalate to
> Fireworks only where local would fail the gate; and when you do escalate, spend
> the minimum tokens.**

Our routing does exactly that, across all eight capability categories (factual,
math, sentiment, summarization, NER, code-debug, logic, code-gen):

```
per task ──▶ classify category (free, heuristic)
          ──▶ math? try exact calculator ....................... 0 tokens, exact
          ──▶ local model answer (+ self-consistency on unsure categories)
          ──▶ confidence = category prior + free verifiers + sample agreement
                 confident?  keep local answer ................. 0 tokens
                 unsure?     escalate → smallest capable Fireworks model,
                                        tiny prompt, hard token cap
```

Free/local levers do the work: **deterministic solvers** (exact arithmetic +
percentage/discount/speed word-problems, a transitive-ordering solver for
comparative puzzles, and a syllogism checker) that answer whole categories for
zero tokens; **self-consistency sampling** (disagreement = escalate); **hard
verifiers** (valid number / valid JSON / code compiles / real sentiment label /
length constraint); and **category priors**. When we must escalate, prompts are
compressed with a ~3-token system nudge and a per-category `max_tokens` cap, and
any chain-of-thought happens locally for free.

## Measured result (local eval harness, 32 tasks across all 8 categories)

The token cut is a direct function of local-model quality — the harness makes
that knob explicit:

**Dev set (32 tasks, tuned):**

| Local model (free tier)            | Accuracy | Fireworks tokens | Cut vs baseline |
|------------------------------------|:--------:|:----------------:|:---------------:|
| baseline — every task → Fireworks  |  96.9%   |       2140       |        —        |
| generic small (2B)                 |  90.6%   |        273       |    **87.2%**    |
| code-capable (~3-4B, e.g. Qwen2.5-Coder-3B) | 96.9% | **126**   |    **94.1%**    |

**Held-out generalization — 96 unseen, adversarially-generated tasks the router
was never tuned on (the honest number):**

| Local model | Accuracy (gate 80%) | Cut vs baseline |
|-------------|:-------------------:|:---------------:|
| generic 2B  |     88.5% ✓         |     64.6%       |
| code-capable ~3B | 93.8% ✓        |     77.3%       |

Reproduce:

```bash
python -m eval.harness --profile strong                         # dev
python -m eval.harness --profile strong \
  --tasks eval/datasets/stress_tasks.json \
  --expected eval/datasets/stress_expected.json                  # held-out
```

Absolute numbers depend on the real task mix and model; the point is a **working,
tunable pipeline** — free deterministic solvers (arithmetic, ordering,
syllogism), minimal/compressed remote prompts, self-consistency + verifier
routing — plus the harness to re-tune it once the real models are in hand.

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
