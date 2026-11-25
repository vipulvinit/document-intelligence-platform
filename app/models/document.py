from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    """A logical document (for example "Clinical Data Management SOP"). Each upload adds a version."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    versions: Mapped[list["DocumentVersion"]] = relationship(  # noqa: F821
        back_populates="document", order_by="DocumentVersion.id", cascade="all, delete-orphan"
    )
