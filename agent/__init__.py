"""Track 1 — Token-Efficient Agent.

A batch agent for the AMD ACT II judging harness. Reads /input/tasks.json,
answers each task with the *fewest Fireworks tokens possible* — using plain
deterministic code (0 tokens) wherever it can and calling the cheapest allowed
Fireworks model only when a task genuinely needs an LLM — and writes
/output/results.json.

Scoring reminder: there is no local LLM; passing submissions are ranked by total
tokens through FIREWORKS_BASE_URL, ascending. So: solve in code when provable,
otherwise spend the fewest Fireworks tokens per task (minimal prompt, tight
max_tokens, cheapest model, batching).
"""

__version__ = "1.0.0"
