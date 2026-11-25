from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    node_keys: list[str] = Field(min_length=1)
    version: str | None = None  # default: latest
    max_cases: int = Field(default=8, ge=1, le=30)


class ArtifactOut(BaseModel):
    id: int
    status: str
    stale_reason: str | None
    payload: dict
    source_hashes: dict[str, str]

    model_config = {"from_attributes": True}
