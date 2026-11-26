"""Version history of an article and on-demand diffs between any two versions."""

from sqlalchemy.orm import Session

from app.models import Article
from app.versioning.differ import summarize
from app.versioning.matcher import NodeRef, match_nodes


def list_versions(db: Session, article_id: int) -> list[dict]:
    article = db.get(Article, article_id)
    if not article:
        return []
    return [
        {
            "label": v.label,
            "content_kind": v.content_kind,
            "created_at": v.created_at.isoformat(),
            "bump": v.change_summary.get("bump"),
            "counts": v.change_summary.get("counts", {}),
            "stale_answer_ids": v.change_summary.get("stale_answer_ids", []),
        }
        for v in article.versions
    ]


def diff_versions(
    db: Session, article_id: int, from_label: str | None = None, to_label: str | None = None
) -> dict:
    article = db.get(Article, article_id)
    if not article or (len(article.versions) < 2 and not (from_label and to_label)):
        return {"error": "Need at least two versions to compare."}
    by_label = {v.label: v for v in article.versions}
    new = by_label.get(to_label) if to_label else article.versions[-1]
    if new is None:
        return {"error": f"Unknown version. Available: {list(by_label)}"}
    old = by_label.get(from_label) if from_label else article.versions[article.versions.index(new) - 1]
    if old is None or old is new:
        return {"error": f"Unknown version. Available: {list(by_label)}"}
    ref = lambda n: NodeRef(n.node_key, n.kind, n.text, n.content_hash, n.heading_path)  # noqa: E731
    summary = summarize(match_nodes([ref(n) for n in old.nodes], [ref(n) for n in new.nodes]))
    return {"article_id": article_id, "from": old.label, "to": new.label} | summary.to_dict()
