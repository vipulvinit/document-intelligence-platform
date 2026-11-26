"""Text embeddings (free, local).

`fastembed` runs a small open-source model (BAAI/bge-small-en-v1.5, 384 dimensions) on CPU
through ONNX, with no GPU or API key. `hashing` is a deterministic bag-of-words fallback
(feature hashing of words and word pairs): weaker on meaning, but it needs no download,
so tests and offline demos always work.

Vectors are L2-normalised float32, so cosine similarity is a plain dot product.
"""

import hashlib
import re
from functools import lru_cache

import numpy as np

from app.core.config import get_settings
from app.core.logging import logger

_DIM = 512
_TOKEN = re.compile(r"[a-z0-9]+")


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    name = "hashing-512"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower())
            for f in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                out[row, h % _DIM] += 1.0 if (h >> 9) % 2 else -1.0
        return _normalise(out)


class FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return _normalise(np.array(list(self._model.embed(texts)), dtype=np.float32))


@lru_cache
def get_embedder():
    s = get_settings()
    if s.embedding_backend == "fastembed":
        try:
            return FastEmbedder(s.embedding_model)
        except Exception as exc:  # not installed, or model download blocked
            logger.warning(f"fastembed unavailable ({exc}); falling back to hashing embeddings")
    return HashingEmbedder()


def to_bytes(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)
