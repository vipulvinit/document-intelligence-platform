from pydantic import BaseModel


class VersionOut(BaseModel):
    document: str
    version: str
    previous_version: str | None
    bump: str | None
    counts: dict[str, int]
    changes: list[dict]
    stale_artifact_ids: list[int]
    injection_flags: dict[str, list[str]]


class AskRequest(BaseModel):
    question: str
    document_name: str | None = None


class AgentRequest(BaseModel):
    goal: str
