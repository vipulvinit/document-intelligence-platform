from sqlalchemy import JSON, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Chunk(Base):
    """A retrieval unit: neighbouring nodes of one section merged (or a long node split) to ~800 chars."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("document_versions.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text)
    node_keys: Mapped[list[str]] = mapped_column(JSON)
    embedding: Mapped[list[float]] = mapped_column(JSON)
