from pydantic import BaseModel, Field


class SelectionRequest(BaseModel):
    name: str = "default"
    node_keys: list[str] = Field(min_length=1, description="Heading keys include all their sub-sections")
