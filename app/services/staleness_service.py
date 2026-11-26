"""Traceability: when a new version of an article changes or removes a passage, or the article is
retracted, every answer that cited it is marked stale, with the reason."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.constants import AnswerStatus
from app.models import Answer, AnswerSource


def mark_stale_answers(
    db: Session,
    article_id: int,
    current_hashes: dict[str, str] | None,
    version_label: str | None = None,
    reason: str | None = None,
) -> list[int]:
    """`current_hashes` = node_key -> content hash of the article's new version.
    Pass None with a `reason` to mark every answer citing the article stale (e.g. retraction)."""
    rows = db.execute(
        select(AnswerSource, Answer)
        .join(Answer, Answer.id == AnswerSource.answer_id)
        .where(AnswerSource.article_id == article_id, Answer.status == AnswerStatus.FRESH)
    ).all()
    still_present = set(current_hashes.values()) if current_hashes else set()
    reasons: dict[int, list[str]] = {}
    answers: dict[int, Answer] = {}
    for src, ans in rows:
        answers[ans.id] = ans
        if current_hashes is None:
            reasons.setdefault(ans.id, []).append(reason or "source no longer valid")
            continue
        new_hash = current_hashes.get(src.node_key)
        if new_hash == src.content_hash or src.content_hash in still_present:  # same, or only moved
            continue
        what = "removed" if new_hash is None else "changed"
        reasons.setdefault(ans.id, []).append(f"{src.node_key} was {what}")
    for answer_id, why in reasons.items():
        ans = answers[answer_id]
        ans.status = AnswerStatus.STALE
        suffix = f" (version {version_label})" if version_label else ""
        ans.stale_reason = "; ".join(why) + suffix
    return sorted(reasons)


def list_answers(db: Session, status: str | None = None, limit: int = 100) -> list[Answer]:
    q = select(Answer).order_by(Answer.id.desc()).limit(limit)
    if status:
        q = q.where(Answer.status == status)
    return list(db.scalars(q).all())
