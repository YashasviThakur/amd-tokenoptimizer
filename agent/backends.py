"""Model backends: a bundled local model (free) and a Fireworks client (metered).

`LocalModel` runs a small GGUF in-process via llama-cpp-python (CPU) — free, not
metered. `Model` talks to Fireworks via FIREWORKS_BASE_URL; the tokens it meters
are the score. Both expose the same .chat() signature so the router is agnostic.
"""
from __future__ import annotations

import re
import time

import httpx

from .tokens import count_messages, count_tokens

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def _clean_answer(text: str) -> str:
    """Strip any inline reasoning trace some models emit before the answer.

    Well-behaved reasoning models put the trace in a separate `reasoning_content`
    field and leave `content` clean, but a few emit a <think>...</think> block
    inline. Remove it so the judge sees only the answer (an unstripped trace is
    scored as a wrong answer)."""
    return _THINK.sub("", text or "").strip()


class RemoteMeter:
    """Tallies the only thing that counts: tokens sent through Fireworks."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Model:
    def __init__(self, base_url: str, api_key: str, timeout: float, meter: RemoteMeter | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.meter = meter  # set on the remote model only
        # one reused connection pool for the whole batch. Granular timeout: keep
        # a generous read timeout (batched generations legitimately take a while)
        # but fail FAST on connect/pool so a dead/blackholed network can't burn
        # the whole per-task/total time budget on a hung TCP connect.
        # connect=8s (was 5s): a grading sandbox may route FIREWORKS_BASE_URL through
        # an internal proxy with extra hops/cold-start latency vs. hitting the public
        # API directly, and a too-tight connect timeout would fail every single call.
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=8.0, pool=8.0),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def chat(self, model: str, messages: list[dict], max_tokens: int = 128,
             temperature: float = 0.0, n: int = 1, reasoning_effort: str | None = None) -> list[str]:
        payload = {"model": model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature}
        if n > 1:
            payload["n"] = n
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        url = f"{self.base_url}/chat/completions"
        # one retry on transient failure (timeout / 5xx) — cheap insurance
        data = None
        for attempt in range(2):
            try:
                r = self._client.post(url, json=payload)
                if r.status_code == 400 and "reasoning_effort" in payload:
                    # ANY 400 while this param is present: retry without it. A model
                    # that rejects reasoning_effort with a differently-worded error
                    # must not fall through to an empty answer (gate risk).
                    payload.pop("reasoning_effort")
                    r = self._client.post(url, json=payload)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError("server error", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(0.5)

        texts = [_clean_answer(c["message"].get("content")) for c in data["choices"]]
        if self.meter is not None:
            usage = data.get("usage") or {}
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if pt is None:
                pt = count_messages(messages)
            if ct is None:
                ct = sum(count_tokens(t) for t in texts)
            self.meter.add(pt, ct)
        return texts


class LocalModel:
    """In-process local model (llama-cpp-python) — a drop-in with the same .chat()
    signature as the Fireworks Model, so the router is unchanged. No server
    (Ollama isn't available in the grading box); weights are bundled in the image
    and loaded once at startup. Local inference is FREE (never metered)."""

    meter = None  # local answers cost 0 tokens

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int | None = None):
        from llama_cpp import Llama  # lazy: only the local-enabled image ships it
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_batch=256, verbose=False)

    def chat(self, model, messages, max_tokens: int = 128, temperature: float = 0.0,
             n: int = 1, reasoning_effort=None) -> list[str]:
        """Return n completions. n>1 = self-consistency samples (varied seed + a
        little temperature); the router treats agreement as a free signal."""
        outs = []
        for i in range(max(1, n)):
            temp = temperature if n == 1 else max(temperature, 0.4)
            r = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=temp, seed=1234 + i)
            outs.append(r["choices"][0]["message"]["content"])
        return outs
