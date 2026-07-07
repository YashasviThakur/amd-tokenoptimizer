"""FastAPI app: an OpenAI-compatible proxy + the live savings/GPU cockpit.

Request flow for POST /v1/chat/completions:
    1. embed the user query
    2. semantic-cache lookup      -> hit  => return, 0 new tokens
    3. complexity router          -> local (AMD GPU) or remote (Fireworks)
    4. run backend, cache result, record savings + GPU spike
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .backends import MockBackend, OpenAICompatBackend, last_user_text
from .cache import SemanticCache
from .config import settings
from .embeddings import Embedder, normalize_query
from .gpu import GpuMonitor
from .metrics import MetricsStore
from .router import ComplexityRouter
from .tokens import count_message_tokens, count_tokens

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="TokenOptimizer", version="0.1.0")

embedder = Embedder(settings.embedding_model)
cache = SemanticCache(settings.cache_threshold, settings.cache_max_size)
router = ComplexityRouter(settings.router_complexity_threshold)
metrics = MetricsStore(settings.remote_model)
gpu = GpuMonitor(mode=settings.mode, device_name=settings.gpu_label)

if settings.mode == "live":
    local_backend = OpenAICompatBackend(
        "local", settings.local_base_url, settings.local_api_key, settings.local_model, settings.request_timeout
    )
    remote_backend = OpenAICompatBackend(
        "remote", settings.fireworks_base_url, settings.fireworks_api_key, settings.remote_model, settings.request_timeout
    )
else:
    local_backend = MockBackend("local", settings.local_model, "local", 60, 320)
    remote_backend = MockBackend("remote", settings.remote_model, "remote", 700, 2200)


def _openai_response(text, model, pt, ct, rec, route, complexity=None, reason="", cache_score=None) -> dict:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
        # non-standard extension the dashboard reads; harmless to other clients
        "x_tokenoptimizer": {
            "route": route,
            "complexity": complexity,
            "reason": reason,
            "cost_usd": rec["cost_usd"],
            "baseline_usd": rec["baseline_usd"],
            "saved_usd": rec["saved_usd"],
            "cache_score": cache_score,
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 512)
    query = last_user_text(messages)
    emb = embedder.encode(normalize_query(query))

    # 1) semantic cache
    t0 = time.perf_counter()
    hit = cache.lookup(emb)
    if hit:
        latency = (time.perf_counter() - t0) * 1000.0
        pt, ct = hit["prompt_tokens"], hit["completion_tokens"]
        rec = metrics.record(
            route="cache", query=query, prompt_tokens=pt, completion_tokens=ct,
            model=hit["model"], latency_ms=latency, reason="semantic cache hit",
            cache_score=round(hit["score"], 3),
        )
        return JSONResponse(_openai_response(
            hit["response_text"], hit["model"], pt, ct, rec,
            route="cache", reason="semantic cache hit", cache_score=round(hit["score"], 3),
        ))

    # 2) route
    decision = router.decide(query)
    route = decision["route"]
    backend = local_backend if route == "local" else remote_backend
    try:
        result = await backend.complete(messages, temperature, max_tokens)
    except Exception as primary_err:
        # graceful fallback: swap tiers rather than fail the request
        alt = remote_backend if backend is local_backend else local_backend
        try:
            result = await alt.complete(messages, temperature, max_tokens)
            route = "remote" if alt is remote_backend else "local"
            decision["reason"] += " (fell back after primary error)"
        except Exception:
            return JSONResponse({"error": {"message": str(primary_err)}}, status_code=502)

    pt = result.get("prompt_tokens") or count_message_tokens(messages)
    ct = result.get("completion_tokens") or count_tokens(result["text"])
    if route == "local":
        gpu.mark_local_inference(ct)  # light up the AMD GPU
    cache.add(emb, query, result["text"], pt, ct, result["model"])
    rec = metrics.record(
        route=route, query=query, prompt_tokens=pt, completion_tokens=ct,
        model=result["model"], latency_ms=result["latency_ms"],
        complexity=decision["complexity"], reason=decision["reason"],
    )
    return JSONResponse(_openai_response(
        result["text"], result["model"], pt, ct, rec,
        route=route, complexity=decision["complexity"], reason=decision["reason"],
    ))


@app.get("/api/stats")
async def api_stats():
    s = metrics.snapshot()
    s["cache"] = cache.stats()
    s["mode"] = settings.mode
    s["embedder"] = embedder.backend
    s["models"] = {"local": settings.local_model, "remote": settings.remote_model}
    return s


@app.get("/api/gpu")
async def api_gpu():
    return gpu.read()


@app.get("/api/recent")
async def api_recent():
    return {"records": metrics.recent()}


@app.get("/api/health")
async def api_health():
    return {"status": "ok", "mode": settings.mode}


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(WEB_DIR / "index.html"))
