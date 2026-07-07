"""Token counting (tiktoken when available; char heuristic otherwise)."""
from __future__ import annotations

import math

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    return max(1, math.ceil(len(text) / 4))


def count_messages(messages) -> int:
    total = 0
    for m in messages or []:
        content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        total += count_tokens(content or "") + 4
    return total + 2
