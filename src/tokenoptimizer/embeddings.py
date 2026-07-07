"""Sentence embeddings for the semantic cache + router.

Prefers `sentence-transformers` (all-MiniLM-L6-v2) when available; otherwise
falls back to a dependency-free hashed bag-of-words embedding so the whole
system still runs (and demos) on a bare machine.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")

_CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "who's": "who is", "it's": "it is",
    "that's": "that is", "there's": "there is", "how's": "how is", "where's": "where is",
    "when's": "when is", "don't": "do not", "doesn't": "does not", "can't": "cannot",
    "won't": "will not", "i'm": "i am", "you're": "you are", "isn't": "is not",
    "aren't": "are not", "let's": "let us", "we're": "we are",
}


def normalize_query(text: str) -> str:
    """Lowercase, expand common contractions, strip punctuation — a stable cache key.

    Lets paraphrases like "what's the capital of France?" and
    "what is the capital of France" collapse to the same embedding.
    """
    t = (text or "").lower().strip()
    for k, v in _CONTRACTIONS.items():
        t = t.replace(k, v)
    return " ".join(_WORD.findall(t))


def _tokenize(text: str):
    return _WORD.findall((text or "").lower())


class Embedder:
    def __init__(self, model_name: str, dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self._backend = "hash"
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
            self._backend = "sentence-transformers"
        except Exception:
            self._model = None
            self._backend = "hash"

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, text: str) -> np.ndarray:
        if self._model is not None:
            v = self._model.encode([text], normalize_embeddings=True)[0]
            return np.asarray(v, dtype=np.float32)
        return self._hash_encode(text)

    def _hash_encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 7) & 1 else -1.0
            vec[idx] += sign
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
