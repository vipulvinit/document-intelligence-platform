"""Selections: pick sections of a document to generate from. Picking a heading includes its subtree."""

from sqlalchemy.orm import Session

from app.core.constants import NodeKind
from app.models import DocumentVersion, Node, Selection


def expand_selection(version: DocumentVersion, node_keys: list[str]) -> list[Node]:
    """Return the selected nodes plus every descendant of a selected heading, in reading order."""
    keys = set(node_keys)
    chosen = []
    for n in version.nodes:
        if n.node_key in keys or any(n.node_key.startswith(k + "/") for k in keys):
            chosen.append(n)
    return chosen


def save_selection(db: Session, document_id: int, name: str, node_keys: list[str]) -> Selection:
    sel = Selection(document_id=document_id, name=name, node_keys=node_keys)
    db.add(sel)
    db.commit()
    return sel


def content_nodes(nodes: list[Node]) -> list[Node]:
    return [n for n in nodes if n.kind not in (NodeKind.DOCUMENT, NodeKind.HEADING)]
