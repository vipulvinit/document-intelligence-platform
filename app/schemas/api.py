"""Request and response bodies of the HTTP API."""

from datetime import datetime

from pydantic import BaseModel, Field


class IngestOut(BaseModel):
    status: str  # new | updated | unchanged
    article_id: int
    external_id: str
    title: str
    version: str | None
    previous_version: str | None = None
    bump: str | None = None
    counts: dict[str, int] = {}
    changes: list[dict] = []
    stale_answer_ids: list[int] = []
    injection_flags: dict[str, list[str]] = {}
    figures_stored: int = 0
    media_nodes: int = 0


class ArticleOut(BaseModel):
    id: int
    external_id: str
    title: str
    journal: str | None
    published_date: str | None
    doi: str | None
    url: str | None
    license: str | None
    is_preprint: bool
    retracted: bool
    source_id: int | None
    latest_version: str | None = None
    content_kind: str | None = None

    model_config = {"from_attributes": True}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    article_ids: list[int] | None = None  # limit the search to these articles


class AgentRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=2000)


class SourceIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    query: str = Field(min_length=1, description="Europe PMC query, see https://europepmc.org/searchsyntax")
    kind: str = "europepmc"
    poll_interval_hours: int = Field(default=24, ge=1, le=24 * 30)
    enabled: bool = True


class SourcePatch(BaseModel):
    query: str | None = None
    poll_interval_hours: int | None = Field(default=None, ge=1, le=24 * 30)
    enabled: bool | None = None
    watermark: str | None = Field(default=None, description="Restart from this index date (YYYY-MM-DD)")


class SourceOut(BaseModel):
    id: int
    name: str
    kind: str
    query: str
    enabled: bool
    poll_interval_hours: int
    watermark: str | None
    last_polled_at: datetime | None

    model_config = {"from_attributes": True}


class RunOut(BaseModel):
    id: int
    source_id: int | None
    trigger: str
    status: str
    found: int
    new: int
    updated: int
    unchanged: int
    failed: int
    errors: list
    details: dict
    started_at: datetime
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class LiteratureSearchIn(BaseModel):
    query: str = Field(min_length=2, description="Europe PMC query, e.g. 'Oropouche virus treatment'")
    limit: int = Field(default=5, ge=1, le=25)


class AnswerOut(BaseModel):
    id: int
    question: str
    answer: str
    status: str
    stale_reason: str | None
    created_at: datetime
    payload: dict

    model_config = {"from_attributes": True}
