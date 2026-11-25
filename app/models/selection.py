from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.document import utcnow


class Selection(Base):
    """A saved set of sections a user picked (for example, "all of section 3") for generation."""

    __tablename__ = "selections"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    node_keys: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class GeneratedArtifact(Base):
    """LLM-generated content (QA test cases) with full traceability to its source nodes.

    `source_hashes` records the content hash of every node it was generated from. When a
    later version changes one of those nodes, the artifact is marked stale.
    """

    __tablename__ = "generated_artifacts"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("document_versions.id"))
    kind: Mapped[str] = mapped_column(String(32), default="qa_test_case")
    payload: Mapped[dict] = mapped_column(JSON)
    source_hashes: Mapped[dict] = mapped_column(JSON)  # node_key -> content_hash
    status: Mapped[str] = mapped_column(String(16), default="fresh")
    stale_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
