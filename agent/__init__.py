"""Track 1 — Hybrid Token-Efficient Routing Agent.

A batch agent for the AMD ACT II judging harness. Reads /input/tasks.json,
answers each task with the *fewest Fireworks tokens possible* — doing as much as
it can on a free local model and escalating to Fireworks only when a local
answer can't be trusted — and writes /output/results.json.

Scoring reminder: local tokens count as zero; passing submissions are ranked by
total tokens through FIREWORKS_BASE_URL, ascending. So: route local, verify
locally (free), escalate rarely, and keep escalation prompts tiny.
"""

__version__ = "1.0.0"
