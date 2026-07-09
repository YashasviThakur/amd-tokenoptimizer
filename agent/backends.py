"""Model backends: a bundled local model (free) and a Fireworks client (metered).

`LocalModel` runs a small GGUF in-process via llama-cpp-python (CPU) — free, not
metered. `Model` talks to Fireworks via FIREWORKS_BASE_URL; the tokens it meters
are the score. Both expose the same .chat() signature so the router is agnostic.
"""
from __future__ import annotations

import re
import sys
import threading
import time

import httpx

from .tokens import count_messages, count_tokens

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.S | re.I)  # unclosed/truncated trace tail


class RecoverableResponseError(Exception):
    """The gateway returned HTTP 200/4xx/5xx with a body we can't turn into an
    answer (an {"error":...} body, no "choices", etc.). Distinct from a transport
    error so the router can fail over to the NEXT allowed model instead of zeroing
    the task — the observed 26% failure was one bad model with no fallback."""


def _clean_answer(text: str) -> str:
    """Strip any inline reasoning trace some models emit before the answer.

    Well-behaved reasoning models put the trace in a separate `reasoning_content`
    field and leave `content` clean, but some emit a <think>...</think> block
    inline (closed, or unclosed if truncated). Remove it so the judge sees only
    the answer (an unstripped trace is scored as a wrong answer)."""
    t = _THINK.sub("", text or "")
    t = _THINK_OPEN.sub("", t)
    return t.strip()


def _extract_message_text(choice: dict) -> tuple[str, str]:
    """Pull (answer_text, reasoning_trace) out of one choice, defensively across
    gateway shapes: OpenAI `message.content` (str), a content PARTS list
    ([{type,text},...]), or a legacy `text` field. The `reasoning_content`
    channel is returned SEPARATELY and never as the answer: submitting a raw
    reasoning trace as the answer is judged wrong every time — that (empty
    content + trace-only responses at our old caps) reproduced the 26.3% run
    exactly in gateway simulation."""
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # content-parts array (a real OpenAI variant)
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not (isinstance(content, str) and content.strip()):
        alt = choice.get("text")  # completions-style field
        content = alt if isinstance(alt, str) else ""
    reasoning = msg.get("reasoning_content")
    return _clean_answer(content), (reasoning if isinstance(reasoning, str) else "")


_ANSWER_MARK = re.compile(r"(?:final answer|the answer is|answer)\s*[:\-]?\s*(.+?)\s*$",
                          re.I | re.M)


def _salvage_answer(trace: str) -> str:
    """Best-effort FINAL ANSWER pulled from a reasoning trace — used only as a
    last-resort floor when every retry/failover still returned no clean content.
    A short extracted answer is sometimes right; the raw trace never is."""
    t = _THINK.sub("", trace or "").strip()
    if not t:
        return ""
    m = None
    for m in _ANSWER_MARK.finditer(t):
        pass  # keep the LAST marker (reasoning often restates before concluding)
    if m and m.group(1).strip():
        cand = m.group(1).strip()
        if len(cand) <= 200:
            return cand
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    return last if 0 < len(last) <= 160 else ""


def _fold_system(messages: list[dict]) -> list[dict]:
    """Merge any system message into the first user turn. The Track-1 allowed
    models are mostly gemma-4, whose chat template can REJECT the system role
    outright through the judging proxy (competitor-confirmed failure + fix); a
    single user message is accepted by every model and costs the same tokens."""
    sys_txt = "\n".join(m.get("content", "") for m in messages
                        if m.get("role") == "system").strip()
    rest = [m for m in messages if m.get("role") != "system"]
    if not sys_txt:
        return messages
    if rest and rest[0].get("role") == "user":
        merged = {**rest[0], "content": f"{sys_txt}\n\n{rest[0].get('content', '')}"}
        return [merged] + rest[1:]
    return [{"role": "user", "content": sys_txt}] + rest


class RemoteMeter:
    """Tallies the only thing that counts: tokens sent through Fireworks.

    Thread-safe: tasks are routed concurrently, so many worker threads call add()
    at once. A lock keeps the running totals from racing (lost updates would
    under-count the token score)."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self._lock = threading.Lock()

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
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

    def list_models(self) -> list[str]:
        """GET {base_url}/models — the model IDs the gateway ACTUALLY serves.

        The grader injects its own FIREWORKS_BASE_URL, which may be a private/AMD
        deployment serving DIFFERENT model IDs than the public API. Calling a name
        that deployment doesn't host 404s every request -> every remote task empty
        -> the deterministic solver-only score we keep seeing. Discovering the real
        list lets us call names that exist. Best-effort: any failure returns []."""
        try:
            r = self._client.get(f"{self.base_url}/models", timeout=10.0)
            r.raise_for_status()
            data = r.json()
            items = data.get("data") if isinstance(data, dict) else data
            if isinstance(items, list):
                ids = [m.get("id") if isinstance(m, dict) else m for m in items]
                return [i for i in ids if isinstance(i, str) and i]
        except Exception as e:
            print(f"[agent] /models probe failed: {str(e)[:120]}", file=sys.stderr)
        return []

    def chat(self, model: str, messages: list[dict], max_tokens: int = 128,
             temperature: float = 0.0, n: int = 1, reasoning_effort: str | None = None,
             timeout: float | None = None) -> list[dict]:
        """One chat-completion call. Returns [{"text": <clean answer or "">,
        "finish": <finish_reason>, "salvage": <answer extracted from a reasoning
        trace, only when text is empty>}] — the router uses `finish` to detect
        truncation and fail over, and `salvage` only as a last-resort floor.
        Raises RecoverableResponseError for any MODEL-specific failure (5xx, 4xx,
        error body, no choices) so the router can try the next allowed model;
        raises httpx errors only for genuine transport problems."""
        # System role folded into the user turn on EVERY remote call (gemma-4
        # template safety, see _fold_system).
        payload = {"model": model, "messages": _fold_system(messages),
                   "max_tokens": max_tokens, "temperature": temperature}
        if n > 1:
            payload["n"] = n
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        url = f"{self.base_url}/chat/completions"
        post_kw = {"json": payload}
        if timeout is not None:
            post_kw["timeout"] = timeout
        # Retry policy:
        #  * transport blip (connect/pool)  -> one fast retry (0.5s)
        #  * 429 / 5xx                      -> one retry after a real backoff; the
        #    judging proxy rate-limits bursts, and a 429 with no backoff turned
        #    whole batches into fallback answers for other teams
        #  * ReadTimeout                    -> NOT retried (a slow model won't be
        #    faster on retry; it doubles wall time)
        #  * other 4xx / error body         -> RecoverableResponseError so the
        #    router fails over to the NEXT allowed model
        data = None
        for attempt in range(2):
            try:
                r = self._client.post(url, **post_kw)
                if (400 <= r.status_code < 500 and r.status_code != 429
                        and "reasoning_effort" in payload):
                    # gateway may reject the non-standard field with any 4xx —
                    # drop it and try once more before giving up on the model.
                    payload.pop("reasoning_effort")
                    r = self._client.post(url, **post_kw)
                if r.status_code == 429 or r.status_code >= 500:
                    if attempt == 0:
                        time.sleep(1.5)
                        continue
                    raise RecoverableResponseError(f"{model}: HTTP {r.status_code} (after backoff)")
                if r.status_code >= 400:
                    body = ""
                    try:
                        body = str(r.json().get("error"))[:120]
                    except Exception:
                        body = r.text[:120]
                    raise RecoverableResponseError(f"{model}: HTTP {r.status_code} {body}")
                data = r.json()
                break
            except (httpx.ReadTimeout, RecoverableResponseError):
                raise
            except Exception:
                if attempt == 1:
                    raise
                time.sleep(0.5)

        if not isinstance(data, dict) or data.get("error") or not data.get("choices"):
            reason = str(data.get("error"))[:120] if isinstance(data, dict) else "non-dict body"
            raise RecoverableResponseError(f"{model}: no choices ({reason})")

        results = []
        for c in data["choices"]:
            text, reasoning = _extract_message_text(c)
            results.append({"text": text, "finish": c.get("finish_reason"),
                            "salvage": "" if text else _salvage_answer(reasoning)})
        if self.meter is not None:
            usage = data.get("usage") or {}
            pt = usage.get("prompt_tokens")
            ct = usage.get("completion_tokens")
            if pt is None:
                pt = count_messages(messages)
            if ct is None:
                ct = sum(count_tokens(x["text"]) for x in results)
            self.meter.add(pt, ct)
        return results


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
        # main.py routes tasks from a thread pool (max_workers=8), but a llama.cpp
        # context is NOT thread-safe: two concurrent generations share one KV cache
        # and corrupt each other — or segfault, an uncatchable native fault that
        # kills the process mid-batch (= zero score). Serialize every generation;
        # the 2-vCPU grading box couldn't run two at once any faster anyway.
        self._lock = threading.Lock()

    def chat(self, model, messages, max_tokens: int = 128, temperature: float = 0.0,
             n: int = 1, reasoning_effort=None) -> list[str]:
        """Return n completions. n>1 = self-consistency samples (varied seed + a
        little temperature); the router treats agreement as a free signal."""
        outs = []
        with self._lock:
            for i in range(max(1, n)):
                temp = temperature if n == 1 else max(temperature, 0.4)
                r = self.llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens, temperature=temp, seed=1234 + i)
                outs.append(r["choices"][0]["message"]["content"])
        return outs
