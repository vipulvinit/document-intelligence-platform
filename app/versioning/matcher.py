"""Matches nodes of a new version to nodes of the previous version.

Passes, in order (each only looks at nodes not yet matched):
  1. Same key and same content hash          -> unchanged
  2. Same content hash, different key        -> unchanged (moved or renumbered)
  3. Same key, similar text                  -> edited
  4. Best similar text of the same kind      -> edited (paragraph shifted position)
Anything left over is added (new side) or removed (old side).
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.core.constants import MATCH_SIMILARITY, MINOR_EDIT_SIMILARITY, ChangeType
from app.versioning.hashing import normalize


@dataclass(frozen=True)
class NodeRef:
    """The minimum we need from a node (old DB row or new parsed node) to compare it."""

    node_key: str
    kind: str
    text: str
    content_hash: str
    heading_path: str = ""


@dataclass
class Match:
    new: NodeRef | None
    old: NodeRef | None
    change: ChangeType
    similarity: float = 1.0


@dataclass
class MatchResult:
    matches: list[Match] = field(default_factory=list)

    def by_change(self, change: ChangeType) -> list[Match]:
        return [m for m in self.matches if m.change == change]


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_NEGATION = re.compile(r"\b(not|no|never|cannot|must not|shall not|without)\b", re.IGNORECASE)


def meaning_changed(old: str, new: str) -> bool:
    """High text similarity can hide a real change: "5 days" -> "3 days", "must" -> "must not"."""
    return _NUMBER.findall(old) != _NUMBER.findall(new) or len(_NEGATION.findall(old)) != len(
        _NEGATION.findall(new)
    )


def _edit_type(sim: float, old: str, new: str) -> ChangeType:
    if sim >= MINOR_EDIT_SIMILARITY and not meaning_changed(old, new):
        return ChangeType.MINOR_EDIT
    return ChangeType.MODIFIED


def match_nodes(old: list[NodeRef], new: list[NodeRef]) -> MatchResult:
    result = MatchResult()
    old_left = {n.node_key: n for n in old}
    new_left = {n.node_key: n for n in new}

    def take(n_new: NodeRef, n_old: NodeRef | None, change: ChangeType, sim: float = 1.0):
        result.matches.append(Match(n_new, n_old, change, sim))
        new_left.pop(n_new.node_key, None)
        if n_old is not None:
            old_left.pop(n_old.node_key, None)

    # 1. same key, same hash
    for key, n_new in list(new_left.items()):
        n_old = old_left.get(key)
        if n_old and n_old.content_hash == n_new.content_hash:
            take(n_new, n_old, ChangeType.UNCHANGED)

    # 2. same hash, different key
    by_hash: dict[str, list[NodeRef]] = {}
    for n in old_left.values():
        by_hash.setdefault(n.content_hash, []).append(n)
    for n_new in list(new_left.values()):
        candidates = [c for c in by_hash.get(n_new.content_hash, []) if c.node_key in old_left]
        if candidates:
            take(n_new, candidates[0], ChangeType.UNCHANGED)

    # 3. same key, similar enough text
    for key, n_new in list(new_left.items()):
        n_old = old_left.get(key)
        if n_old and n_old.kind == n_new.kind:
            sim = similarity(n_old.text, n_new.text)
            if sim >= MATCH_SIMILARITY:
                take(n_new, n_old, _edit_type(sim, n_old.text, n_new.text), sim)

    # 4. best similar node of the same kind anywhere (small bonus for staying in the same section)
    for n_new in list(new_left.values()):
        best, best_score, best_sim = None, 0.0, 0.0
        for n_old in old_left.values():
            if n_old.kind != n_new.kind:
                continue
            sim = similarity(n_old.text, n_new.text)
            score = sim + (0.02 if n_old.heading_path == n_new.heading_path else 0.0)
            if sim >= MATCH_SIMILARITY and score > best_score:
                best, best_score, best_sim = n_old, score, sim
        if best is not None:
            take(n_new, best, _edit_type(best_sim, best.text, n_new.text), best_sim)

    for n_new in new_left.values():
        result.matches.append(Match(n_new, None, ChangeType.ADDED, 0.0))
    for n_old in old_left.values():
        result.matches.append(Match(None, n_old, ChangeType.REMOVED, 0.0))
    return result
