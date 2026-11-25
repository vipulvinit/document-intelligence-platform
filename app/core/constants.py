"""Shared enums and constants."""

from enum import StrEnum


class NodeKind(StrEnum):
    DOCUMENT = "document"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"


class ChangeType(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    MINOR_EDIT = "minor_edit"  # wording tweak, similarity >= MINOR_EDIT_SIMILARITY
    UNCHANGED = "unchanged"


class ArtifactStatus(StrEnum):
    FRESH = "fresh"
    STALE = "stale"


# Two versions of a paragraph count as "the same paragraph, edited" above this similarity.
MATCH_SIMILARITY = 0.6
# Above this similarity an edit is cosmetic (typo, punctuation) and only bumps the PATCH number.
MINOR_EDIT_SIMILARITY = 0.95
