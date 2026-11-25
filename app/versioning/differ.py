"""Summarises a match result and decides the semantic version bump.

Semantic versioning for documents (MAJOR.MINOR.PATCH):
  MAJOR  structure changed: a section heading was added or removed
  MINOR  content changed: paragraphs or list items added, removed or substantially edited
         (also a heading renamed)
  PATCH  only cosmetic edits (typos, punctuation; similarity >= 0.95)
  none   nothing changed after normalisation -> upload is rejected as a duplicate
"""

from dataclasses import dataclass, field

from app.core.constants import ChangeType, NodeKind
from app.versioning.matcher import MatchResult


@dataclass
class DiffSummary:
    bump: str | None  # "major" | "minor" | "patch" | None
    counts: dict[str, int] = field(default_factory=dict)
    changes: list[dict] = field(default_factory=list)  # human-readable change list

    def to_dict(self) -> dict:
        return {"bump": self.bump, "counts": self.counts, "changes": self.changes}


def summarize(result: MatchResult) -> DiffSummary:
    counts = {c.value: 0 for c in ChangeType}
    changes = []
    structural = content = cosmetic = False

    for m in result.matches:
        counts[m.change.value] += 1
        node = m.new or m.old
        if node.kind == NodeKind.DOCUMENT or m.change == ChangeType.UNCHANGED:
            continue
        is_heading = node.kind == NodeKind.HEADING
        if m.change in (ChangeType.ADDED, ChangeType.REMOVED):
            structural |= is_heading
            content |= not is_heading
        elif m.change == ChangeType.MODIFIED:
            content = True
        elif m.change == ChangeType.MINOR_EDIT:
            cosmetic = True
        changes.append(
            {
                "change": m.change.value,
                "kind": node.kind,
                "node_key": node.node_key,
                "old_key": m.old.node_key if m.old else None,
                "section": node.heading_path,
                "old_text": m.old.text if m.old else None,
                "new_text": m.new.text if m.new else None,
                "similarity": round(m.similarity, 3),
            }
        )

    bump = "major" if structural else "minor" if content else "patch" if cosmetic else None
    return DiffSummary(bump=bump, counts=counts, changes=changes)


def next_version(previous: str | None, bump: str | None) -> str:
    if previous is None:
        return "1.0.0"
    major, minor, patch = (int(x) for x in previous.split("."))
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    return previous
