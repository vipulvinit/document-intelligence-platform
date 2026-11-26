from sqlalchemy import JSON, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Chunk(Base):
    """A retrieval unit: neighbouring paragraphs of one section merged (or a long one split),
    or a single figure or table.

    Only chunks of an article's latest version are searched. `embedding` holds float32 bytes,
    which numpy reads without copying (4x smaller than JSON).
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("article_versions.id"), index=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), index=True)
    is_latest: Mapped[bool] = mapped_column(default=True, index=True)
    position: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(32), default="text")  # text | figure | table
    heading_path: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text)
    node_keys: Mapped[list[str]] = mapped_column(JSON)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    embedding_model: Mapped[str] = mapped_column(String(128))
