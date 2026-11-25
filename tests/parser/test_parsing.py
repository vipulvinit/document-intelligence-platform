import pytest
from pydantic import ValidationError

from app.parsers.extractor import UnsupportedFileType, extract_blocks
from app.parsers.hierarchy_builder import build_hierarchy
from app.schemas.node import ParsedDocument
from tests.conftest import SAMPLES


def _tree(name):
    data = (SAMPLES / name).read_bytes()
    return build_hierarchy(extract_blocks(name, data), title="SOP")


def test_markdown_tree_structure():
    doc = _tree("sop_v1.md")
    keys = [n.node_key for n in doc.nodes]
    assert doc.title == "Clinical Data Management SOP"  # single H1 becomes the document title
    assert "3-data-validation/3-2-query-management/p1" in keys
    item = next(n for n in doc.nodes if n.node_key == "2-roles/li1")
    assert item.kind == "list_item" and item.parent_key == "2-roles"
    assert next(n for n in doc.nodes if n.node_key.endswith("3-1-edit-checks")).level == 3


def test_pdf_and_markdown_give_the_same_tree():
    md, pdf = _tree("sop_v1.md"), _tree("sop_v1.pdf")
    assert [(n.node_key, n.kind) for n in pdf.nodes] == [(n.node_key, n.kind) for n in md.nodes]


def test_schema_rejects_duplicate_keys():
    doc = _tree("sop_v1.md")
    nodes = doc.nodes + [doc.nodes[1]]
    with pytest.raises(ValidationError):
        ParsedDocument(title="x", nodes=nodes)


def test_unsupported_file_type():
    with pytest.raises(UnsupportedFileType):
        extract_blocks("data.xlsx", b"...")
