"""Shared enums and constants."""

from enum import StrEnum


class NodeKind(StrEnum):
    DOCUMENT = "document"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    FIGURE = "figure"
    TABLE = "table"


# Figures and tables are retrieved on their own: never merged with neighbouring paragraphs.
MEDIA_KINDS = (NodeKind.FIGURE, NodeKind.TABLE)
STRUCTURAL_KINDS = (NodeKind.DOCUMENT, NodeKind.HEADING)


class ChangeType(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    MINOR_EDIT = "minor_edit"  # wording tweak, similarity >= MINOR_EDIT_SIMILARITY
    UNCHANGED = "unchanged"


class AnswerStatus(StrEnum):
    FRESH = "fresh"
    STALE = "stale"  # a cited source changed or was removed after the answer was given


class ContentKind(StrEnum):
    FULL_TEXT = "full_text"
    ABSTRACT = "abstract"  # full text unavailable (not open access, or the server failed)
    UPLOAD = "upload"


class SourceKind(StrEnum):
    EUROPEPMC = "europepmc"


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"  # some articles failed
    FAILED = "failed"


# Two versions of a paragraph count as "the same paragraph, edited" above this similarity.
MATCH_SIMILARITY = 0.6
# Above this similarity an edit is cosmetic (typo, punctuation) and only bumps the PATCH number.
MINOR_EDIT_SIMILARITY = 0.95
