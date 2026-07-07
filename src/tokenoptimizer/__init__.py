"""TokenOptimizer — cost-governance routing proxy for agent swarms, built on AMD.

An OpenAI-compatible gateway that answers cheap queries on a local model
running on an AMD GPU (ROCm) and escalates only the hard ones to a frontier
model via the Fireworks AI API — with a live GPU + savings cockpit that proves
the money it saves.
"""

__version__ = "0.1.0"
