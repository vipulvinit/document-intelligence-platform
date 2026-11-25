from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.document import utcnow


class DocumentVersion(Base):
    """One uploaded revision, labelled with a semantic version (MAJOR.MINOR.PATCH)."""

    __tablename__ = "document_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    label: Mapped[str] = mapped_column(String(32))  # e.g. "1.2.0"
    source_filename: Mapped[str] = mapped_column(String(255))
    source_hash: Mapped[str] = mapped_column(String(64))  # SHA-256 of the raw file
    change_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped["Document"] = relationship(back_populates="versions")  # noqa: F821
    nodes: Mapped[list["Node"]] = relationship(  # noqa: F821
        back_populates="version", order_by="Node.position", cascade="all, delete-orphan"
    )
