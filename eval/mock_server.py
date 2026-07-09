"""A stand-in for the real local + Fireworks endpoints, for offline dev.

Exposes an OpenAI-compatible /v1/chat/completions. It simulates a *weak* local
model and a *strong* Fireworks model with per-category competence, returning the
canonical correct answer or a degraded wrong one, plus realistic token usage.
The agent code is 100% real and unaware it's talking to a simulator — swap the
base URLs for real servers and nothing else changes.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from fastapi import FastAPI, Request

from agent.tokens import count_messages, count_tokens

DATA = Path(__file__).resolve().parent / "datasets"
TASKS = json.loads(Path(os.getenv("MOCK_TASKS", str(DATA / "dev_tasks.json"))).read_text(encoding="utf-8"))
EXPECTED = json.loads(Path(os.getenv("MOCK_EXPECTED", str(DATA / "dev_expected.json"))).read_text(encoding="utf-8"))
PROMPT_TO_ID = {t["prompt"]: t["task_id"] for t in TASKS}

LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL", "local-small")

# How often the LOCAL model returns the correct answer, by category. This is a
# transparent knob: the token cut is a direct function of local-model quality.
#   generic = a small general 2B model
#   strong  = a code-capable ~3-4B model (e.g. Qwen2.5-Coder-3B) — recommended
COMPETENCE_PROFILES = {
    "generic": {"factual": 0.70, "math": 0.50, "sentiment": 0.90, "summarization": 0.80,
                "ner": 0.75, "code_debug": 0.50, "logic": 0.40, "code_gen": 0.45},
    "strong": {"factual": 0.88, "math": 0.65, "sentiment": 0.95, "summarization": 0.90,
               "ner": 0.88, "code_debug": 0.85, "logic": 0.55, "code_gen": 0.85},
}
LOCAL_COMPETENCE = COMPETENCE_PROFILES.get(os.getenv("MOCK_LOCAL_PROFILE", "generic"),
                                           COMPETENCE_PROFILES["generic"])
REMOTE_COMPETENCE = 0.96

# GRADER-FAITHFUL MODE: the real judging proxy fronts *reasoning* models that put
# the answer in `reasoning_content` and leave `content` EMPTY (documented as the
# 26.3%/36.8% failure). The public Fireworks API and the default mock both return
# clean `content`, so this failure NEVER shows up locally. MOCK_REASONING=1 makes
# the simulated remote model behave like the grader: empty content, answer buried
# in a realistic reasoning trace, and reasoning tokens billed as completion tokens.
REASONING = os.getenv("MOCK_REASONING", "0").strip().lower() in ("1", "true", "yes")


def _reasoning_trace(category: str, answer: str) -> str:
    """Wrap the answer in a realistic chain-of-thought whose FINAL line/marker is
    the answer — but where naive 'last line' or 'whole trace' extraction fails
    (esp. sentiment: the answer word is embedded mid-sentence, not alone)."""
    a = answer or ""
    first = a.splitlines()[0] if a else a  # short inline answers (word/number)
    if category == "sentiment":
        return (f"Let me read the text carefully. It expresses clear emotion and "
                f"some strong wording. Weighing the cues on balance, the overall "
                f"sentiment here comes across as {first}. So I'll go with {first}.")
    if category == "math":
        return (f"Let me work through this step by step, carefully checking each "
                f"operation. After doing the arithmetic, the result comes out to {first}.")
    if category == "ner":
        return (f"Scanning the passage for named entities — people, organizations, "
                f"locations, and dates. Collecting them into the required schema.\n{a}")
    if category == "summarization":
        return (f"The passage makes several points; let me identify the core idea "
                f"and condense it while honoring the length constraint.\n\n{a}")
    if category in ("code_debug", "code_gen"):
        return (f"Let me reason about the code, trace the logic, and determine the "
                f"correct fix before writing it out.\n\n```python\n{a}\n```")
    return (f"Let me think about this carefully and recall the relevant facts. "
            f"Considering everything, the final answer is {first}.")


app = FastAPI()


def _last_user(messages):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return m.get("content", "")
    return messages[-1].get("content", "") if messages else ""


# Prompts sorted longest-first so a containment match picks the most specific task.
_PROMPTS_BY_LEN = sorted(PROMPT_TO_ID, key=len, reverse=True)


def _match_task(user: str) -> str | None:
    """Map a user turn to a task id. The agent FOLDS the system prompt into the
    user turn (_fold_system) and COMPRESSES whitespace before sending, so an exact
    lookup misses — the mock would return 'I don't know' for every real call and
    silently score 0. Match by containment (longest prompt first) so folded /
    compressed / reformatted prompts still resolve to the right task, exactly as a
    real judged model would answer the underlying question."""
    if user in PROMPT_TO_ID:
        return PROMPT_TO_ID[user]
    norm = re.sub(r"\s+", " ", user or "").strip()
    for p in _PROMPTS_BY_LEN:
        if p in user or re.sub(r"\s+", " ", p).strip() in norm:
            return PROMPT_TO_ID[p]
    return None


def _simulate(task_id: str, exp: dict, is_local: bool, model: str, i: int) -> str:
    comp = LOCAL_COMPETENCE.get(exp["category"], 0.6) if is_local else REMOTE_COMPETENCE
    rng = random.Random(f"{task_id}|{model}|{i}")
    return exp["answer"] if rng.random() < comp else exp["wrong"]


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model", "")
    messages = body.get("messages", [])
    n = int(body.get("n", 1) or 1)
    is_local = (model == LOCAL_MODEL_NAME)

    user = _last_user(messages)

    # batched request? lines like "N) <prompt>" -> reply "N) <answer>"
    items = re.findall(r"(?m)^\s*(\d+)\)\s*(.+?)\s*$", user)
    category = "factual"
    if len(items) >= 2:
        out = []
        for num, ptext in items:
            tid = _match_task(ptext.strip())
            if tid is None:
                out.append(f"{num}) unknown")
            else:
                ans = _simulate(tid, EXPECTED[tid], is_local, model, 0).splitlines()[0]
                out.append(f"{num}) {ans}")
        contents = ["\n".join(out)]
    else:
        task_id = _match_task(user)
        if task_id is None:
            contents = ["I don't know." for _ in range(n)]
        else:
            exp = EXPECTED[task_id]
            category = exp.get("category", "factual")
            contents = [_simulate(task_id, exp, is_local, model, i) for i in range(n)]

    # Grader-faithful reasoning models: empty `content`, answer inside
    # `reasoning_content`. Only the REMOTE model behaves this way (a local GGUF
    # returns plain content). Billed tokens count the reasoning trace, exactly as
    # the real proxy does — so the token score stays realistic too.
    reasoning_mode = REASONING and not is_local
    choices, billed = [], []
    for i, c in enumerate(contents):
        if reasoning_mode:
            trace = _reasoning_trace(category, c)
            msg = {"role": "assistant", "content": "", "reasoning_content": trace}
            billed.append(trace)
        else:
            msg = {"role": "assistant", "content": c}
            billed.append(c)
        choices.append({"index": i, "message": msg, "finish_reason": "stop"})

    prompt_tokens = count_messages(messages)
    completion_tokens = sum(count_tokens(b) for b in billed)
    return {
        "id": "mock-cmpl",
        "object": "chat.completion",
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/health")
async def health():
    return {"ok": True, "tasks": len(TASKS)}
