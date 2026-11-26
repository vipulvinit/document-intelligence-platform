from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base
from app.models.article import utcnow


class Answer(Base):
    """An answer the helpline gave, kept with exactly what it cited.

    Each cited passage is stored with its content hash (`AnswerSource`). When a later fetch of
    that article changes or removes the passage, or the article is retracted, the answer is marked
    stale with the reason. The judge agent's scores are attached to this row.
    """

    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)  # stored with personal data masked
    answer: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)  # quotes, attempts, model, prompt version
    status: Mapped[str] = mapped_column(String(16), default="fresh")
    stale_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    sources: Mapped[list["AnswerSource"]] = relationship(
        back_populates="answer", cascade="all, delete-orphan", order_by="AnswerSource.id"
    )


class AnswerSource(Base):
    __tablename__ = "answer_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    answer_id: Mapped[int] = mapped_column(ForeignKey("answers.id"), index=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id"), index=True)
    node_key: Mapped[str] = mapped_column(String(512))
    content_hash: Mapped[str] = mapped_column(String(64))

    answer: Mapped["Answer"] = relationship(back_populates="sources")
