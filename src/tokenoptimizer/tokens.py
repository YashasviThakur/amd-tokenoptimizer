"""Token counting with a graceful fallback when tiktoken isn't installed."""
from __future__ import annotations

import math

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken is optional / may need network
    _enc = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    # ~4 characters per token heuristic
    return max(1, math.ceil(len(text) / 4))


def _content(message) -> str:
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", "") or ""


def count_message_tokens(messages) -> int:
    total = 0
    for m in messages or []:
        total += count_tokens(_content(m)) + 4  # per-message overhead
    return total + 2
