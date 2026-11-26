"""Validated data structures produced by the parsing pipeline."""

from pydantic import BaseModel, Field, field_validator

from app.core.constants import NodeKind


class RawBlock(BaseModel):
    """A block of content as it comes out of a file, before structure is inferred."""

    text: str
    page: int | None = None
    font_size: float | None = None
    bold: bool = False
    heading_level: int | None = None  # known directly for Markdown ("## Title" -> 2) and JATS XML
    is_list_item: bool = False
    # Figures and tables
    media_kind: NodeKind | None = None  # NodeKind.FIGURE or NodeKind.TABLE
    label: str | None = None  # "Figure 2", "Table 1"
    asset_ref: str | None = None  # figure image: file name in the article package
    asset_bytes: bytes | None = Field(default=None, repr=False)  # image already in hand (from a PDF)
    asset_ext: str | None = None  # "png", "jpeg"... for asset_bytes
    table_markdown: str | None = None

    @field_validator("text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        return " ".join(v.split())


class ParsedNode(BaseModel):
    """One node of the document tree, ready to store and compare across versions."""

    node_key: str = Field(min_length=1)
    parent_key: str | None
    kind: NodeKind
    level: int = Field(ge=0, le=6)
    title: str | None = None
    text: str = Field(min_length=1)
    heading_path: str = ""
    page: int | None = None
    position: int = Field(ge=0)
    content_hash: str = Field(min_length=64, max_length=64)
    asset_ref: str | None = None
    asset_bytes: bytes | None = Field(default=None, repr=False)
    asset_ext: str | None = None
    extra: dict = Field(default_factory=dict)  # label, caption, table_markdown


class ParsedDocument(BaseModel):
    title: str
    nodes: list[ParsedNode]

    @field_validator("nodes")
    @classmethod
    def keys_unique_and_parents_exist(cls, nodes: list[ParsedNode]) -> list[ParsedNode]:
        keys = [n.node_key for n in nodes]
        if len(keys) != len(set(keys)):
            raise ValueError("node_key values must be unique")
        known = set(keys)
        for n in nodes:
            if n.parent_key is not None and n.parent_key not in known:
                raise ValueError(f"parent {n.parent_key!r} of {n.node_key!r} does not exist")
        return nodes


class NodeOut(BaseModel):
    node_key: str
    parent_key: str | None
    kind: str
    level: int
    title: str | None
    text: str
    heading_path: str
    page: int | None
    change_type: str
    object_key: str | None = None

    model_config = {"from_attributes": True}
