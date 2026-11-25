"""Hybrid retrieval: BM25 keyword search + embedding similarity, merged with reciprocal rank fusion.

Why hybrid: embeddings catch paraphrases ("lock out" ~ "suspend access"), while BM25 catches
exact terms like form codes or drug names that embeddings can blur. RRF combines the two
rankings without having to put their scores on the same scale.
"""

from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.llm.embeddings import cosine, get_embedder
from app.models import Chunk, Document, DocumentVersion, Node

RRF_K = 60


@dataclass
class Hit:
    document: str
    version: str
    chunk_text: str
    node_keys: list[str]
    score: float
    cosine: float
    bm25: float


def _tokens(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 1]


def latest_versions(db: Session, document_name: str | None = None) -> list[DocumentVersion]:
    docs = (
        db.scalars(select(Document).where(Document.name == document_name))
        if document_name
        else db.scalars(select(Document))
    )
    return [d.versions[-1] for d in docs if d.versions]


def search(db: Session, query: str, document_name: str | None = None, top_k: int | None = None) -> list[Hit]:
    top_k = top_k or get_settings().retrieval_top_k
    versions = latest_versions(db, document_name)
    rows = []
    for v in versions:
        for c in db.scalars(select(Chunk).where(Chunk.version_id == v.id)).all():
            rows.append((v, c))
    if not rows:
        return []

    q_vec = get_embedder().embed([query])[0]
    cos = [cosine(q_vec, c.embedding) for _, c in rows]
    bm25 = BM25Okapi([_tokens(c.text) for _, c in rows]).get_scores(_tokens(query))

    def ranks(scores):
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return {i: r for r, i in enumerate(order, start=1)}

    r_cos, r_bm = ranks(cos), ranks(list(bm25))
    fused = {
        i: 1 / (RRF_K + r_cos[i]) + (1 / (RRF_K + r_bm[i]) if bm25[i] > 0 else 0) for i in range(len(rows))
    }
    best = sorted(fused, key=fused.get, reverse=True)[:top_k]
    return [
        Hit(
            rows[i][0].document.name,
            rows[i][0].label,
            rows[i][1].text,
            rows[i][1].node_keys,
            round(fused[i], 5),
            round(cos[i], 4),
            round(float(bm25[i]), 3),
        )
        for i in best
    ]


def context_for_hits(db: Session, hits: list[Hit]) -> dict[str, str]:
    """Expand hits to {"document#node_key": text} so the LLM can cite individual paragraphs."""
    context: dict[str, str] = {}
    for h in hits:
        version = db.scalar(
            select(DocumentVersion)
            .join(Document)
            .where(Document.name == h.document, DocumentVersion.label == h.version)
        )
        nodes = db.scalars(
            select(Node).where(Node.version_id == version.id, Node.node_key.in_(h.node_keys))
        ).all()
        for n in sorted(nodes, key=lambda n: n.position):
            context[f"{h.document}#{n.node_key}"] = (
                f"({n.heading_path}) {n.text}" if n.heading_path else n.text
            )
    return context
