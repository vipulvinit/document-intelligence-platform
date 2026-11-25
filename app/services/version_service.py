"""Version history and on-demand diffs between any two versions."""

from sqlalchemy.orm import Session

from app.services.document_service import get_document
from app.versioning.differ import summarize
from app.versioning.matcher import NodeRef, match_nodes


def list_versions(db: Session, name: str) -> list[dict]:
    doc = get_document(db, name)
    if not doc:
        return []
    return [
        {
            "label": v.label,
            "file": v.source_filename,
            "created_at": v.created_at.isoformat(),
            "bump": v.change_summary.get("bump"),
            "counts": v.change_summary.get("counts", {}),
        }
        for v in doc.versions
    ]


def diff_versions(db: Session, name: str, from_label: str | None = None, to_label: str | None = None) -> dict:
    doc = get_document(db, name)
    if not doc or len(doc.versions) < 2 and not (from_label and to_label):
        return {"error": "Need at least two versions to compare."}
    by_label = {v.label: v for v in doc.versions}
    new = by_label.get(to_label) if to_label else doc.versions[-1]
    old = by_label.get(from_label) if from_label else doc.versions[doc.versions.index(new) - 1]
    if old is None or new is None:
        return {"error": f"Unknown version. Available: {list(by_label)}"}
    ref = lambda n: NodeRef(n.node_key, n.kind, n.text, n.content_hash, n.heading_path)  # noqa: E731
    summary = summarize(match_nodes([ref(n) for n in old.nodes], [ref(n) for n in new.nodes]))
    return {"document": name, "from": old.label, "to": new.label} | summary.to_dict()
