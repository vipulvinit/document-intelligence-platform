"""Text embeddings.

`fastembed` runs a small open-source model (BAAI/bge-small-en-v1.5, 384 dimensions) on CPU
through ONNX, with no GPU or API key. `hashing` is a deterministic bag-of-words fallback
(feature hashing of words and word pairs): weaker on meaning, but it needs no download,
so tests and offline demos always work.
"""

import hashlib
import math
import re
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import logger

_DIM = 512
_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    name = "hashing-512"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * _DIM
        tokens = _TOKEN.findall(text.lower())
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for f in features:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % _DIM] += 1.0 if (h >> 9) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class FastEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        self.name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]


@lru_cache
def get_embedder():
    s = get_settings()
    if s.embedding_backend == "fastembed":
        try:
            return FastEmbedder(s.embedding_model)
        except Exception as exc:  # not installed, or model download blocked
            logger.warning(f"fastembed unavailable ({exc}); falling back to hashing embeddings")
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
