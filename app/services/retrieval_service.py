"""Hybrid retrieval: BM25 keyword search + embedding similarity, merged with reciprocal rank fusion.

Why hybrid: embeddings catch paraphrases ("high fever" ~ "pyrexia"), while BM25 catches exact
terms like drug names, gene names or "NS1" that embeddings can blur. RRF combines the two
rankings without having to put their scores on the same scale.

Speed: the index (one numpy matrix of all latest-version chunk vectors + a BM25 model) is built
once and kept in memory. It is rebuilt only when the chunk table changes (count or newest id),
which this process checks with one cheap query per search, so articles ingested by the
scheduler process show up automatically. A query is then one matrix-vector product:
~50,000 chunks x 384 dims takes a few milliseconds, no vector database needed.
"""

import threading
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.constants import MEDIA_KINDS
from app.llm.embeddings import from_bytes, get_embedder
from app.models import Article, ArticleVersion, Chunk, Node
from app.storage import get_storage

RRF_K = 60
MAX_CONTEXT_CHARS_PER_NODE = 4000


@dataclass
class Hit:
    chunk_id: int
    article_id: int
    version_id: int
    kind: str
    chunk_text: str
    node_keys: list[str]
    score: float
    cosine: float
    bm25: float


def _tokens(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 1]


class _Index:
    def __init__(self):
        self.key = None
        self.rows: list[
            tuple[int, int, int, str, str, list[str]]
        ] = []  # id, article, version, kind, text, keys
        self.matrix = np.zeros((0, 1), dtype=np.float32)
        self.bm25: BM25Okapi | None = None
        self.lock = threading.Lock()

    def ensure(self, db: Session, model_name: str) -> None:
        count, max_id = db.execute(
            select(func.count(Chunk.id), func.max(Chunk.id)).where(
                Chunk.is_latest.is_(True), Chunk.embedding_model == model_name
            )
        ).one()
        key = (count, max_id, model_name)
        if key == self.key:
            return
        with self.lock:
            if key == self.key:
                return
            chunks = db.execute(
                select(
                    Chunk.id,
                    Chunk.article_id,
                    Chunk.version_id,
                    Chunk.kind,
                    Chunk.text,
                    Chunk.node_keys,
                    Chunk.embedding,
                )
                .where(Chunk.is_latest.is_(True), Chunk.embedding_model == model_name)
                .order_by(Chunk.id)
            ).all()
            self.rows = [tuple(c[:6]) for c in chunks]
            self.matrix = (
                np.vstack([from_bytes(c[6]) for c in chunks]) if chunks else np.zeros((0, 1), np.float32)
            )
            self.bm25 = BM25Okapi([_tokens(c[4]) for c in chunks]) if chunks else None
            self.key = key


_index = _Index()


def reset_index() -> None:
    """Forget the cached index (tests use a fresh database each time)."""
    _index.key = None


def search(
    db: Session, query: str, article_ids: list[int] | None = None, top_k: int | None = None
) -> list[Hit]:
    top_k = top_k or get_settings().retrieval_top_k
    embedder = get_embedder()
    _index.ensure(db, embedder.name)
    if not _index.rows:
        return []

    excluded = set(db.scalars(select(Article.id).where(Article.retracted.is_(True))).all())
    allowed = np.array(
        [r[1] not in excluded and (article_ids is None or r[1] in article_ids) for r in _index.rows]
    )
    if not allowed.any():
        return []

    q_vec = embedder.embed([query])[0]
    cos = _index.matrix @ q_vec
    bm25 = _index.bm25.get_scores(_tokens(query)) if _index.bm25 else np.zeros(len(_index.rows))
    cos = np.where(allowed, cos, -np.inf)
    bm25 = np.where(allowed, bm25, -np.inf)

    # Ranks (1 = best) for reciprocal rank fusion; BM25 contributes only where a query term matched.
    r_cos = np.empty(len(cos), dtype=np.int64)
    r_cos[np.argsort(-cos, kind="stable")] = np.arange(1, len(cos) + 1)
    r_bm = np.empty(len(bm25), dtype=np.int64)
    r_bm[np.argsort(-bm25, kind="stable")] = np.arange(1, len(bm25) + 1)
    fused = 1 / (RRF_K + r_cos) + np.where(bm25 > 0, 1 / (RRF_K + r_bm), 0.0)
    fused = np.where(allowed, fused, -np.inf)

    best = np.argsort(-fused, kind="stable")[:top_k]
    hits = []
    for i in best:
        if not allowed[i]:
            break
        cid, aid, vid, kind, text, keys = _index.rows[i]
        hits.append(
            Hit(
                cid,
                aid,
                vid,
                kind,
                text,
                list(keys),
                round(float(fused[i]), 5),
                round(float(cos[i]), 4),
                round(float(bm25[i]), 3),
            )
        )
    return hits


@dataclass
class ContextItem:
    key: str  # "PMC1234567#results/p2": what an answer cites
    text: str
    article_id: int
    node_key: str
    content_hash: str
    section: str
    kind: str
    figure_url: str | None = None


def context_for_hits(db: Session, hits: list[Hit]) -> list[ContextItem]:
    """Expand hits to the individual paragraphs, figures and tables they cover, in reading order."""
    items: list[ContextItem] = []
    seen: set[str] = set()
    storage = get_storage()
    for h in hits:
        article = db.get(Article, h.article_id)
        nodes = db.scalars(
            select(Node).where(Node.version_id == h.version_id, Node.node_key.in_(h.node_keys))
        ).all()
        for n in sorted(nodes, key=lambda n: n.position):
            key = f"{article.external_id}#{n.node_key}"
            if key in seen:
                continue
            seen.add(key)
            text = n.text[:MAX_CONTEXT_CHARS_PER_NODE]
            if n.heading_path:
                text = f"({n.heading_path}) {text}"
            items.append(
                ContextItem(
                    key=key,
                    text=f"[{article.title}] {text}",
                    article_id=article.id,
                    node_key=n.node_key,
                    content_hash=n.content_hash,
                    section=n.heading_path,
                    kind=n.kind,
                    figure_url=storage.url(n.object_key) if n.object_key and n.kind in MEDIA_KINDS else None,
                )
            )
    return items


def latest_version(db: Session, article_id: int) -> ArticleVersion | None:
    return db.scalar(
        select(ArticleVersion)
        .where(ArticleVersion.article_id == article_id)
        .order_by(ArticleVersion.id.desc())
        .limit(1)
    )
