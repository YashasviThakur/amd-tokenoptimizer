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

app = FastAPI()


def _last_user(messages):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return m.get("content", "")
    return messages[-1].get("content", "") if messages else ""


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
    task_id = PROMPT_TO_ID.get(user)

    contents = []
    if task_id is None:
        contents = ["I don't know." for _ in range(n)]
    else:
        exp = EXPECTED[task_id]
        contents = [_simulate(task_id, exp, is_local, model, i) for i in range(n)]

    prompt_tokens = count_messages(messages)
    completion_tokens = sum(count_tokens(c) for c in contents)
    return {
        "id": "mock-cmpl",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {"index": i, "message": {"role": "assistant", "content": c}, "finish_reason": "stop"}
            for i, c in enumerate(contents)
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/health")
async def health():
    return {"ok": True, "tasks": len(TASKS)}
