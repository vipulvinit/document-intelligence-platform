"""Traceability: when a new version changes a node, artifacts generated from it go stale."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.constants import ArtifactStatus
from app.models import GeneratedArtifact


def mark_stale_artifacts(
    db: Session, document_id: int, version_id: int, current_hashes: dict[str, str]
) -> list[int]:
    """Compare each fresh artifact's recorded source hashes with the new version's hashes."""
    stale_ids = []
    artifacts = db.scalars(
        select(GeneratedArtifact).where(
            GeneratedArtifact.document_id == document_id, GeneratedArtifact.status == ArtifactStatus.FRESH
        )
    ).all()
    current_by_hash = set(current_hashes.values())
    for art in artifacts:
        reasons = []
        for key, old_hash in art.source_hashes.items():
            new_hash = current_hashes.get(key)
            if new_hash == old_hash:
                continue
            if old_hash in current_by_hash:  # text still exists, just moved: not stale
                continue
            reasons.append(f"{key} was {'removed' if new_hash is None else 'changed'}")
        if reasons:
            art.status = ArtifactStatus.STALE
            art.stale_reason = "; ".join(reasons) + f" (in version id {version_id})"
            stale_ids.append(art.id)
    return stale_ids


def list_artifacts(db: Session, document_id: int, status: str | None = None) -> list[GeneratedArtifact]:
    q = select(GeneratedArtifact).where(GeneratedArtifact.document_id == document_id)
    if status:
        q = q.where(GeneratedArtifact.status == status)
    return list(db.scalars(q.order_by(GeneratedArtifact.id)).all())
