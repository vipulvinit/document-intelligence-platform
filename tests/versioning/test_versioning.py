from app.core.constants import ChangeType
from app.parsers.extractor import extract_blocks
from app.parsers.hierarchy_builder import build_hierarchy
from app.versioning.differ import next_version, summarize
from app.versioning.matcher import NodeRef, match_nodes
from tests.conftest import SAMPLES


def refs(name):
    doc = build_hierarchy(extract_blocks(name, (SAMPLES / name).read_bytes()), "SOP")
    return [NodeRef(n.node_key, n.kind, n.text, n.content_hash, n.heading_path) for n in doc.nodes]


def test_content_change_is_minor():
    diff = summarize(match_nodes(refs("sop_v1.md"), refs("sop_v2.md")))
    assert diff.bump == "minor"
    kinds = {(c["change"], c["node_key"]) for c in diff.changes}
    assert ("modified", "3-data-validation/3-2-query-management/p1") in kinds or (
        "minor_edit",
        "3-data-validation/3-2-query-management/p1",
    ) in kinds
    assert ("added", "2-roles/li3") in kinds


def test_new_section_is_major_and_typo_is_cosmetic():
    diff = summarize(match_nodes(refs("sop_v2.md"), refs("sop_v3.md")))
    assert diff.bump == "major"
    purpose = next(c for c in diff.changes if c["node_key"] == "1-purpose/p1")
    assert purpose["change"] == ChangeType.MINOR_EDIT


def test_identical_upload_has_no_bump():
    assert summarize(match_nodes(refs("sop_v1.md"), refs("sop_v1.md"))).bump is None


def test_moved_paragraph_is_unchanged():
    old = [
        NodeRef("a/p1", "paragraph", "Alpha text here.", "h1"),
        NodeRef("a/p2", "paragraph", "Beta text.", "h2"),
    ]
    new = [
        NodeRef("a/p1", "paragraph", "Beta text.", "h2"),
        NodeRef("a/p2", "paragraph", "Alpha text here.", "h1"),
    ]
    result = match_nodes(old, new)
    assert all(m.change == ChangeType.UNCHANGED for m in result.matches)


def test_meaning_change_detection():
    from app.versioning.matcher import meaning_changed

    assert meaning_changed("within 5 days", "within 3 days")
    assert meaning_changed("Records can be edited.", "Records can not be edited.")
    assert not meaning_changed("collected, validated and locked", "collected, validated, and locked")


def test_next_version():
    assert next_version(None, None) == "1.0.0"
    assert next_version("1.2.3", "major") == "2.0.0"
    assert next_version("1.2.3", "minor") == "1.3.0"
    assert next_version("1.2.3", "patch") == "1.2.4"
