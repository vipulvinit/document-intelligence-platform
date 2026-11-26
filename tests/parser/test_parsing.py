import pytest
from pydantic import ValidationError

from app.core.constants import NodeKind
from app.parsers.extractor import UnsupportedFileType, extract_blocks
from app.parsers.hierarchy_builder import build_hierarchy
from app.parsers.jats_parser import parse_jats, table_to_markdown
from app.schemas.node import ParsedDocument
from samples.make_pdf import md_to_pdf
from tests.conftest import SAMPLES

MONITORING_KEY = "3-management/3-2-monitoring/p1"


def _tree(name, data=None):
    data = data if data is not None else (SAMPLES / name).read_bytes()
    return build_hierarchy(extract_blocks(name, data), title="Guideline")


def test_markdown_tree_structure():
    doc = _tree("dengue_guideline_v1.md")
    keys = [n.node_key for n in doc.nodes]
    assert doc.title == "Dengue Care Guideline (synthetic test fixture)"  # single H1 becomes the title
    assert MONITORING_KEY in keys
    item = next(n for n in doc.nodes if n.node_key == "2-warning-signs/li1")
    assert item.kind == "list_item" and item.parent_key == "2-warning-signs"
    assert next(n for n in doc.nodes if n.node_key.endswith("3-1-fever-control")).level == 3


def test_pdf_gives_the_markdown_tree_plus_its_figure():
    md = _tree("dengue_guideline_v1.md")
    pdf = _tree("dengue_guideline_v1.pdf", md_to_pdf(SAMPLES / "dengue_guideline_v1.md"))
    text_nodes = [(n.node_key, n.kind) for n in pdf.nodes if n.kind != NodeKind.FIGURE]
    assert text_nodes == [(n.node_key, n.kind) for n in md.nodes]
    figures = [n for n in pdf.nodes if n.kind == NodeKind.FIGURE]
    assert len(figures) == 1
    fig = figures[0]
    assert fig.title == "Figure 1" and "Platelet count" in fig.text
    assert fig.asset_bytes and fig.asset_ext == "png"
    assert not any(n.text.startswith("Figure 1.") for n in pdf.nodes if n.kind == NodeKind.PARAGRAPH)


def test_jats_metadata_sections_figures_and_tables():
    art = parse_jats((SAMPLES / "sample_article.xml").read_bytes())
    assert art.pmcid == "PMC9990001" and art.pmid == "99990001" and art.doi == "10.0000/synthetic.0001"
    assert art.title == "Oral rehydration in Vibrio cholerae infection: a synthetic trial"
    assert art.journal == "Journal of Synthetic Test Medicine"
    assert art.license == "https://creativecommons.org/licenses/by/4.0/"
    assert art.published_date == "2026-11-03"
    assert "Rao A." in art.authors and "Synthetic Trials Group" in art.authors

    doc = build_hierarchy(art.blocks, art.title, detect_title=False)
    keys = {n.node_key: n for n in doc.nodes}
    assert "abstract/background/p1" in keys and "methods/participants/li2" in keys
    # back matter and declarations are skipped
    assert not any("interests" in k or "funding" in k for k in keys)
    # citation markers are removed from the text
    assert (
        keys["introduction/p1"].text
        == "Dehydration is the main cause of death in cholera. Early rehydration saves lives."
    )

    fig = keys["results/figure-1"]
    assert fig.kind == NodeKind.FIGURE and fig.asset_ref == "synthetic-g001"
    assert fig.text.startswith("Figure 1: Cumulative stool output")
    table = keys["results/table-1"]
    assert table.kind == NodeKind.TABLE
    md = table.extra["table_markdown"]
    assert "| Outcome | Low-osmolarity ORS | Standard ORS |" in md
    assert "| Need for IV fluids | Not significantly different | Not significantly different |" in md
    assert "Note: ORS, oral rehydration solution." in md


def test_table_markdown_escapes_pipes_and_limits_rows():
    import xml.etree.ElementTree as ET

    rows = "".join(f"<tr><td>a|{i}</td></tr>" for i in range(70))
    md = table_to_markdown(
        ET.fromstring(f"<table><thead><tr><th>h</th></tr></thead><tbody>{rows}</tbody></table>")
    )
    assert "a\\|0" in md and "10 more rows not shown" in md


def test_schema_rejects_duplicate_keys():
    doc = _tree("dengue_guideline_v1.md")
    with pytest.raises(ValidationError):
        ParsedDocument(title="x", nodes=doc.nodes + [doc.nodes[1]])


def test_unsupported_file_type():
    with pytest.raises(UnsupportedFileType):
        extract_blocks("data.xlsx", b"...")


def test_jats_licence_linked_inside_text():
    """MDPI-style: the licence URL only appears as a link inside the licence sentence."""
    xml = b"""<article xmlns:xlink="http://www.w3.org/1999/xlink"><front><article-meta>
      <title-group><article-title>T</article-title></title-group>
      <permissions><license><license-p>Licensee MDPI. Distributed under the
        <ext-link xlink:href="https://creativecommons.org/licenses/by/4.0/">CC BY license</ext-link>.
      </license-p></license></permissions></article-meta></front>
      <body><p>Text.</p></body></article>"""
    assert parse_jats(xml).license == "https://creativecommons.org/licenses/by/4.0/"
