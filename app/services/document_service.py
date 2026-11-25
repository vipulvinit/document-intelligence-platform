"""Ingestion: parse -> build tree -> compare with the previous version -> store -> index.

One call does the whole pipeline and returns the new version, including what changed and
which generated test cases went stale.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import logger
from app.core.safety import detect_injection
from app.llm.embeddings import get_embedder
from app.models import Chunk, Document, DocumentVersion, Node
from app.parsers.extractor import extract_blocks
from app.parsers.hierarchy_builder import build_hierarchy
from app.services.chunking import build_chunks
from app.services.staleness_service import mark_stale_artifacts
from app.versioning.differ import next_version, summarize
from app.versioning.hashing import file_hash
from app.versioning.matcher import NodeRef, match_nodes


class DuplicateVersionError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def _ref(n) -> NodeRef:
    return NodeRef(n.node_key, str(n.kind), n.text, n.content_hash, n.heading_path)


def ingest_document(db: Session, name: str, filename: str, data: bytes) -> tuple[DocumentVersion, dict]:
    settings = get_settings()
    blocks = extract_blocks(filename, data)
    if not blocks:
        raise EmptyDocumentError("No text could be extracted from the file.")
    parsed = build_hierarchy(blocks, title=name)

    doc = db.scalar(select(Document).where(Document.name == name))
    previous = doc.versions[-1] if doc and doc.versions else None

    # Compare with the previous version
    old_refs = [_ref(n) for n in previous.nodes] if previous else []
    result = match_nodes(old_refs, [_ref(n) for n in parsed.nodes])
    diff = summarize(result)
    if previous and diff.bump is None:
        raise DuplicateVersionError(f"No content changes compared with version {previous.label}.")
    change_by_key = {m.new.node_key: m.change.value for m in result.matches if m.new}

    flagged = {n.node_key: hits for n in parsed.nodes if (hits := detect_injection(n.text))}
    if flagged:
        logger.warning(f"Possible prompt injection in {len(flagged)} node(s) of {name!r}")

    if doc is None:
        doc = Document(name=name)
        db.add(doc)
        db.flush()

    label = next_version(previous.label if previous else None, diff.bump if previous else None)
    summary = diff.to_dict() | {
        "bump": diff.bump if previous else "initial",
        "previous_version": previous.label if previous else None,
        "injection_flags": flagged,
    }
    version = DocumentVersion(
        document_id=doc.id,
        label=label,
        source_filename=filename,
        source_hash=file_hash(data),
        change_summary=summary,
    )
    db.add(version)
    db.flush()

    for n in parsed.nodes:
        db.add(
            Node(
                version_id=version.id,
                node_key=n.node_key,
                parent_key=n.parent_key,
                kind=n.kind,
                level=n.level,
                title=n.title,
                text=n.text,
                heading_path=n.heading_path,
                page=n.page,
                position=n.position,
                content_hash=n.content_hash,
                change_type=change_by_key.get(n.node_key, "added"),
            )
        )

    drafts = build_chunks(parsed.nodes, settings.chunk_size_chars, settings.chunk_overlap_chars)
    vectors = get_embedder().embed([d.text for d in drafts]) if drafts else []
    for i, (d, vec) in enumerate(zip(drafts, vectors)):
        db.add(
            Chunk(
                version_id=version.id,
                position=i,
                heading_path=d.heading_path,
                text=d.text,
                node_keys=d.node_keys,
                embedding=vec,
            )
        )

    stale = mark_stale_artifacts(db, doc.id, version.id, {n.node_key: n.content_hash for n in parsed.nodes})
    version.change_summary = summary | {"stale_artifact_ids": stale}
    db.commit()
    db.refresh(version)
    logger.info(
        f"Ingested {name!r} as v{label}: {len(parsed.nodes)} nodes, {len(drafts)} chunks, "
        f"{len(stale)} artifacts marked stale"
    )
    return version, version.change_summary


def get_document(db: Session, name: str) -> Document | None:
    return db.scalar(select(Document).where(Document.name == name))


def get_version(db: Session, name: str, label: str | None = None) -> DocumentVersion | None:
    doc = get_document(db, name)
    if not doc or not doc.versions:
        return None
    if label is None:
        return doc.versions[-1]
    return next((v for v in doc.versions if v.label == label), None)
