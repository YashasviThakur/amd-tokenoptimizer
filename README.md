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
Built and pushed automatically by [`.github/workflows/build.yml`](.github/workflows/build.yml)
and linked to this repo under **Packages** (via the OCI `image.source` label), so the
judging harness can discover it from the repository. Tiny `python:3.11-slim` image
(no bundled model). Entrypoint `python -m agent.main`: reads `/input/tasks.json` →
writes `/output/results.json` → exits 0.

## How Track 1 is scored (and how we win it)

The judging harness runs the container headless: it injects `FIREWORKS_API_KEY` /
`FIREWORKS_BASE_URL` / `ALLOWED_MODELS`, mounts `/input/tasks.json`, and you write
`/output/results.json` and exit 0. Then:

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
          ──▶ deterministic solver? (arithmetic / % / average / powers / ordering / syllogism …)
                 yes → answer with plain code ................... 0 tokens, exact
                 no  → cheapest capable Fireworks model,
                       minimal prompt · max_tokens cap · reasoning_effort=low
                       (short same-category lookups are batched into one call)
```

The **deterministic solvers** are the only free path — an exact arithmetic /
percentage / discount / speed / average / powers-roots-factorial-gcd calculator, a
transitive-ordering solver for comparative puzzles, and a syllogism checker — each
answering whole task types for **zero tokens**, and red-teamed to **prove-or-defer**
(they return `None` and escalate to Fireworks whenever they can't fully prove the
answer, so they never emit a wrong one). For everything else, each Fireworks call is
minimized: a ~3-token system prompt, compressed input, a per-category `max_tokens`
cap, `reasoning_effort=low`, cheapest-model selection, and **batching** of short
same-category tasks into a single call to amortize per-call overhead.

## Measured result (local eval harness)

Every task the deterministic solvers handle costs **0 tokens**; the rest go to
Fireworks. Savings come from solver coverage + minimal, batched Fireworks calls:

| Config | Accuracy | Fireworks tokens | Cut vs baseline |
|--------|:--------:|:----------------:|:---------------:|
| baseline — every task → Fireworks | 96.9% | 2164 | — |
| **code + Fireworks (compliant)**  | **96.9%** | **1369** | **37%** |

On a separate **96-task unseen/adversarial** set the agent holds **96.9%** accuracy
(well above an 80% gate) at **19% fewer** tokens. The mock server has no
reasoning-model overhead, so it *understates* batching's real benefit. Reproduce:

```bash
python -m eval.harness                                   # dev set
python -m eval.harness --tasks eval/datasets/stress_tasks.json \
                       --expected eval/datasets/stress_expected.json   # unseen
```

The token cut scales with how many tasks plain code can answer — so the lever is
**broadening the deterministic solvers**, plus minimizing tokens per Fireworks call
(`reasoning_effort=low`, tight `max_tokens`, cheapest model, batching).

## Run the eval today (no GPU, no keys)

```bash
pip install -r requirements.txt         # fastapi/uvicorn/httpx for the harness
python -m eval.harness                   # baseline vs compliant agent
python -m eval.harness --sweep           # threshold sweep — pick the operating point
```

The harness boots a mock server that simulates a Fireworks model (with realistic
token usage) alongside the free deterministic code path, runs the **real agent
code** against it, grades with a local judge, and prints token savings. Swap the
base URL for the real Fireworks endpoint and nothing in the agent changes.

## Run the agent for real

```bash
# env (the harness injects these at eval time; use your own key for dev)
cp agent/.env.example .env                # fill FIREWORKS_API_KEY, ALLOWED_MODELS

# run the batch
INPUT_PATH=eval/datasets/dev_tasks.json OUTPUT_PATH=out/results.json \
  python -m agent.main
```

## Build & submit (Docker)

No bundled model — the image is a tiny `python:3.11-slim` plus the agent, so it
stays far under the 10 GB cap and pulls fast.

```bash
docker buildx build --platform linux/amd64 \
  -f docker/agent.Dockerfile -t <registry>/tokenoptimizer-agent:latest --push .
```

The judging VM is `linux/amd64` — build for it or the image won't pull (scores
zero). CI does this automatically on push to `main`. See
[`docker/agent.Dockerfile`](docker/agent.Dockerfile).

## Robustness & validation

Malformed output or a wrong "free" answer both cost the win, so the agent is
hardened and continuously checked:

- **Solvers prove-or-defer.** A red-team pass drove `solve_ordering` and
  `solve_syllogism` to reject partial/disconnected orders, cross-dimension
  comparisons, cycles/contradictions, and unprovable syllogisms, and the math
  solvers to require an exact operand count (so compound problems defer instead of
  guessing) — always returning `None` (escalate) rather than a possibly-wrong
  answer. **Zero solver misfires across the dev + 96 unseen tasks.**
- **Never a zero-score crash.** Defensive `tasks.json` parsing (list /
  `{"tasks":[]}` / id-map / single / bare-string / alt keys), answers coerced to
  strings, an **atomic** output write, a top-level guard that always leaves a valid
  `results.json` and exits 0, a soft global deadline so a hung network still writes
  output within the time budget, and one retry on a transient Fireworks 5xx.
- **Offline container self-test:** `python -m agent.main --selftest` runs
  solver-answerable tasks with the network disabled and validates the output
  contract (including that compound math problems defer) — a fast image health check.
- **Solver guardrail:** `python -m eval.stress_solvers` asserts a solver never fires
  with a wrong answer and reports classifier accuracy.

## Rules we honor

- Reads `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` from the
  environment — never hardcoded, never a bundled `.env`.
- **No local LLM.** Every task is answered by plain deterministic code (0 tokens)
  or one Fireworks call. **All** remote calls go through `FIREWORKS_BASE_URL`; only
  `ALLOWED_MODELS` are used (read at runtime).
- No cached/hardcoded answers — routing is decided live per task from the prompt.
- Always writes valid JSON and exits 0; every task is wrapped so one failure can't
  sink the batch. Ready < 60s, < 30s per task, ≤ 10 min total.

## Layout

```
agent/                 THE deliverable — batch agent (the only thing in the image)
  main.py              /input/tasks.json → route → /output/results.json
  router.py            code-vs-Fireworks routing + batching policy
  classifier.py        free 8-category classifier
  solvers.py           free deterministic solvers (0-token answers)
  verifiers.py         free checks + exact arithmetic calculator
  prompts.py           token-minimal prompts + per-category caps
  backends.py          Fireworks (metered) OpenAI-compatible client
eval/
  harness.py           baseline vs compliant agent, threshold sweep, token report
  stress_solvers.py    asserts solvers never fire a wrong answer + classifier accuracy
  judge.py             local accuracy judge (LLM-judge stand-in)
  mock_server.py       simulated Fireworks endpoint (realistic token usage)
  datasets/            dev (32) + stress (96 unseen, adversarial) labeled tasks
docker/agent.Dockerfile   the submission image recipe
```

> `src/tokenoptimizer/` + `web/` are an earlier interactive gateway + cost
> dashboard, kept only as a local visualizer. They are **not** part of the Track-1
> submission (which is headless `agent/` + `eval/`) and never enter the scored image.

## License

MIT
