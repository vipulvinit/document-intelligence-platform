"""Ingestion: parsed article -> compare with its previous version -> store nodes, figures, chunks.

Used by both the literature connectors (scheduled and on-demand) and manual uploads:

  new article          -> version 1.0.0
  fetched again, same  -> "unchanged", nothing stored (the scheduler sees this a lot)
  fetched again, edited-> new version with a semantic bump; answers citing changed passages go stale

Figures: the image is stored in object storage (when the licence allows it) and, if a vision
model is configured, described in a few sentences so it can be found by search.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.constants import MEDIA_KINDS, ContentKind, NodeKind
from app.core.logging import logger
from app.core.safety import detect_injection
from app.llm.embeddings import get_embedder, to_bytes
from app.llm.vision import describe_figure
from app.models import Article, ArticleVersion, Chunk, Node
from app.parsers.extractor import extract_blocks
from app.parsers.hierarchy_builder import build_hierarchy, slugify
from app.schemas.node import ParsedDocument, ParsedNode
from app.services.chunking import build_chunks
from app.services.staleness_service import mark_stale_answers
from app.storage import get_storage, safe_key
from app.versioning.differ import next_version, summarize
from app.versioning.hashing import file_hash
from app.versioning.matcher import NodeRef, match_nodes

AssetFetcher = Callable[[str], tuple[bytes, str] | None]  # asset_ref -> (bytes, extension)


class EmptyDocumentError(ValueError):
    pass


@dataclass
class ArticleMeta:
    external_id: str
    title: str
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None
    journal: str | None = None
    authors: str | None = None
    published_date: str | None = None
    url: str | None = None
    license: str | None = None
    is_preprint: bool = False
    source_id: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class IngestResult:
    status: str  # new | updated | unchanged
    article: Article
    version: ArticleVersion | None
    summary: dict = field(default_factory=dict)


def _ref(n) -> NodeRef:
    return NodeRef(n.node_key, str(n.kind), n.text, n.content_hash, n.heading_path)


def licence_allows_storing(licence: str | None) -> bool:
    """Open licences (Creative Commons, public domain) allow keeping a copy of figures."""
    if not licence:
        return False
    text = licence.lower()
    return any(t in text for t in ("creativecommons", "creative commons", "public domain", "cc0")) or (
        text.startswith("cc")
    )


def _store_figure(
    meta: ArticleMeta, n: ParsedNode, fetch_asset: AssetFetcher | None, allow_download: bool
) -> tuple[str | None, bytes | None, str | None]:
    """Returns (object_key, image bytes, extension) for a figure node, or Nones."""
    image, ext = n.asset_bytes, n.asset_ext
    if image is None and n.asset_ref and fetch_asset and allow_download:
        try:
            got = fetch_asset(n.asset_ref)
        except Exception as exc:
            logger.warning(f"Could not download {n.asset_ref} for {meta.external_id}: {exc}")
            got = None
        if got:
            image, ext = got
    if image is None:
        return None, None, None
    ext = (ext or "png").lower().lstrip(".")
    name = slugify(n.node_key.replace("/", "-"), max_len=120)
    key = safe_key("articles", meta.external_id, f"{name}.{ext}")
    get_storage().put(key, image)
    return key, image, ext


def ingest_parsed(
    db: Session,
    meta: ArticleMeta,
    parsed: ParsedDocument,
    raw_hash: str,
    content_kind: ContentKind,
    fetch_asset: AssetFetcher | None = None,
) -> IngestResult:
    settings = get_settings()
    article = db.scalar(select(Article).where(Article.external_id == meta.external_id))
    previous = article.versions[-1] if article and article.versions else None

    if previous and previous.source_hash == raw_hash:
        return IngestResult("unchanged", article, previous)

    # An abstract never replaces full text we already hold (the full-text server may fail one day)
    if previous and previous.content_kind == ContentKind.FULL_TEXT and content_kind == ContentKind.ABSTRACT:
        return IngestResult("unchanged", article, previous, {"reason": "full text already stored"})

    old_refs = [_ref(n) for n in previous.nodes] if previous else []
    result = match_nodes(old_refs, [_ref(n) for n in parsed.nodes])
    diff = summarize(result)
    if previous and diff.bump is None:
        previous.source_hash = raw_hash  # formatting-only change: remember it, store nothing
        db.commit()
        return IngestResult("unchanged", article, previous)
    change_by_key = {m.new.node_key: m.change.value for m in result.matches if m.new}

    flagged = {n.node_key: hits for n in parsed.nodes if (hits := detect_injection(n.text))}
    if flagged:
        logger.warning(f"Possible prompt injection in {len(flagged)} node(s) of {meta.external_id}")

    is_new = article is None
    if is_new:
        article = Article(external_id=meta.external_id, title=meta.title)
        db.add(article)
    for attr in ("title", "pmid", "pmcid", "doi", "journal", "authors", "published_date", "url", "license"):
        value = getattr(meta, attr)
        if value:
            setattr(article, attr, value)
    article.is_preprint = meta.is_preprint
    if meta.source_id and not article.source_id:
        article.source_id = meta.source_id
    article.extra = (article.extra or {}) | meta.extra
    db.flush()

    label = next_version(previous.label if previous else None, diff.bump if previous else None)
    summary = diff.to_dict() | {
        "bump": diff.bump if previous else "initial",
        "previous_version": previous.label if previous else None,
        "injection_flags": flagged,
    }
    version = ArticleVersion(
        article_id=article.id,
        label=label,
        content_kind=content_kind,
        source_hash=raw_hash,
        change_summary=summary,
    )
    db.add(version)
    db.flush()

    allow_download = settings.download_figures and (
        content_kind == ContentKind.UPLOAD or licence_allows_storing(meta.license)
    )
    index_texts: dict[str, str] = {}  # node_key -> text to embed, when it differs from node.text
    figures = 0
    for n in parsed.nodes:
        object_key, extra = None, dict(n.extra)
        if n.kind == NodeKind.FIGURE:
            object_key, image, ext = _store_figure(meta, n, fetch_asset, allow_download)
            if image is not None:
                figures += 1
                # Each description is one vision API call: cap them so one figure-heavy article
                # cannot use up a free daily quota. The rest are still found by their caption.
                description = (
                    describe_figure(image, ext, n.text)
                    if figures <= settings.vision_max_figures_per_article
                    else None
                )
                if description:
                    extra["description"] = description
                    index_texts[n.node_key] = f"{n.text}\nFigure shows: {description}"
        db.add(
            Node(
                version_id=version.id,
                node_key=n.node_key,
                parent_key=n.parent_key,
                kind=n.kind,
                level=n.level,
                title=n.title,
                text=n.text if "description" not in extra else index_texts[n.node_key],
                heading_path=n.heading_path,
                page=n.page,
                position=n.position,
                content_hash=n.content_hash,  # source text only: no new version from descriptions
                change_type=change_by_key.get(n.node_key, "added"),
                object_key=object_key,
                extra=extra,
            )
        )

    # Re-index: the previous version's chunks stop being searched
    if previous:
        db.execute(update(Chunk).where(Chunk.article_id == article.id).values(is_latest=False))
    nodes_for_chunks = [
        n.model_copy(update={"text": index_texts[n.node_key]}) if n.node_key in index_texts else n
        for n in parsed.nodes
    ]
    drafts = build_chunks(nodes_for_chunks, settings.chunk_size_chars, settings.chunk_overlap_chars)
    embedder = get_embedder()
    vectors = embedder.embed([f"{meta.title}\n{d.text}" for d in drafts]) if drafts else []
    for i, (d, vec) in enumerate(zip(drafts, vectors)):
        db.add(
            Chunk(
                version_id=version.id,
                article_id=article.id,
                is_latest=True,
                position=i,
                kind=d.kind,
                heading_path=d.heading_path,
                text=d.text,
                node_keys=d.node_keys,
                embedding=to_bytes(vec),
                embedding_model=embedder.name,
            )
        )

    stale = []
    if previous:
        stale = mark_stale_answers(
            db, article.id, {n.node_key: n.content_hash for n in parsed.nodes}, version_label=label
        )
    version.change_summary = summary | {
        "stale_answer_ids": stale,
        "figures_stored": figures,
        "media_nodes": sum(1 for n in parsed.nodes if n.kind in MEDIA_KINDS),
    }
    db.commit()
    db.refresh(version)
    logger.info(
        f"Ingested {meta.external_id} as v{label} ({content_kind}): {len(parsed.nodes)} nodes, "
        f"{len(drafts)} chunks, {figures} figures stored, {len(stale)} answers marked stale"
    )
    return IngestResult("new" if is_new else "updated", article, version, version.change_summary)


def ingest_upload(db: Session, title: str, filename: str, data: bytes) -> IngestResult:
    """A guideline or article uploaded by hand (PDF, JATS XML, Markdown or text)."""
    blocks = extract_blocks(filename, data)
    if not blocks:
        raise EmptyDocumentError("No text could be extracted from the file.")
    parsed = build_hierarchy(blocks, title=title)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:100] or "document"
    meta = ArticleMeta(external_id=f"upload:{slug}", title=parsed.title or title, license="uploaded")
    key = safe_key("uploads", slug, filename)
    get_storage().put(key, data)
    meta.extra = {"original_file": key}
    return ingest_parsed(db, meta, parsed, file_hash(data), ContentKind.UPLOAD)


def get_article(db: Session, article_id: int) -> Article | None:
    return db.get(Article, article_id)


def get_version(db: Session, article_id: int, label: str | None = None) -> ArticleVersion | None:
    article = db.get(Article, article_id)
    if not article or not article.versions:
        return None
    if label is None:
        return article.versions[-1]
    return next((v for v in article.versions if v.label == label), None)


def set_retracted(db: Session, article_id: int, retracted: bool = True) -> list[int]:
    """Retracted articles are excluded from search, and answers that cited them go stale."""
    article = db.get(Article, article_id)
    if not article:
        return []
    article.retracted = retracted
    stale = (
        mark_stale_answers(db, article_id, None, reason=f"{article.external_id} was retracted")
        if retracted
        else []
    )
    db.commit()
    return stale
