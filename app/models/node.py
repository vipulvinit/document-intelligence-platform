from sqlalchemy import JSON, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Node(Base):
    """One structural element of a version: a heading, paragraph or list item.

    `node_key` is a stable, human-readable path such as "3-data-validation/p2". It lets
    us match the same element across versions and trace generated content back to it.
    """

    __tablename__ = "nodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("document_versions.id"), index=True)
    node_key: Mapped[str] = mapped_column(String(512), index=True)
    parent_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))
    level: Mapped[int] = mapped_column(Integer)  # 0 = document root, 1 = top heading...
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    heading_path: Mapped[str] = mapped_column(Text, default="")  # "Scope > Roles"
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    position: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    change_type: Mapped[str] = mapped_column(String(16), default="added")  # vs previous version
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)

    version: Mapped["DocumentVersion"] = relationship(back_populates="nodes")  # noqa: F821
