from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Article(Base):
    """One journal article, preprint, guideline or uploaded file. Each fetch that changes it adds a version.

    `external_id` identifies it across fetches: "PMC1234567", "MED:4321" (PubMed abstract only),
    "PPR:PPR98765" (preprint) or "upload:<slug>" for files uploaded by hand.
    """

    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("sources.id"), nullable=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    pmid: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    pmcid: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    journal: Mapped[str | None] = mapped_column(String(512), nullable=True)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_date: Mapped[str | None] = mapped_column(String(32), nullable=True)  # as the source gives it
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    license: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_preprint: Mapped[bool] = mapped_column(Boolean, default=False)
    retracted: Mapped[bool] = mapped_column(Boolean, default=False)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    versions: Mapped[list["ArticleVersion"]] = relationship(
        back_populates="article", order_by="ArticleVersion.id", cascade="all, delete-orphan"
    )

    def citation(self) -> dict:
        """What a user sees for a cited source."""
        return {
            "article_id": self.id,
            "external_id": self.external_id,
            "title": self.title,
            "journal": self.journal,
            "published": self.published_date,
            "doi": self.doi,
            "url": self.url or (f"https://doi.org/{self.doi}" if self.doi else None),
            "license": self.license,
            "preprint": self.is_preprint,
            "retracted": self.retracted,
        }


class ArticleVersion(Base):
    """One fetched or uploaded revision, labelled with a semantic version (MAJOR.MINOR.PATCH)."""

    __tablename__ = "article_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), index=True)
    label: Mapped[str] = mapped_column(String(32))  # e.g. "1.2.0"
    content_kind: Mapped[str] = mapped_column(String(16))  # full_text | abstract | upload
    source_hash: Mapped[str] = mapped_column(String(64))  # SHA-256 of the raw XML / file
    change_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    article: Mapped["Article"] = relationship(back_populates="versions")
    nodes: Mapped[list["Node"]] = relationship(  # noqa: F821
        back_populates="version", order_by="Node.position", cascade="all, delete-orphan"
    )
