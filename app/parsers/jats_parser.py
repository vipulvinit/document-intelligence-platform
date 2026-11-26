"""Parses JATS XML, the format of open-access full text from Europe PMC / PubMed Central.

JATS already marks sections, paragraphs, figures and tables, so unlike a PDF nothing has to be
guessed from font sizes. Output: article metadata plus RawBlocks in reading order:
  - "Abstract" and each <sec> title become headings (nested <sec> = deeper level)
  - <p> and list items become paragraphs / list items (citation markers like [3] are dropped)
  - <fig> becomes a figure block (label, caption, image file name)
  - <table-wrap> becomes a table block (label, caption, table as Markdown)
Back matter (references, funding, conflicts of interest...) is skipped: it rarely answers a
medical question and would crowd out useful passages in retrieval.

Security: the XML comes from the internet. Python's bundled Expat (>= 2.4.1) refuses
entity-expansion bombs, and no external entities or DTDs are ever fetched.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from app.core.constants import NodeKind
from app.schemas.node import RawBlock

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"
MAX_TABLE_ROWS = 60

_SKIP_SEC_TYPES = {"coi-statement", "supplementary-material", "data-availability", "author-contributions"}
_SKIP_TITLES = re.compile(
    r"^(acknowledg|funding|conflicts? of interest|competing interests?|declaration of interests?|"
    r"author(s')? contributions?|contributors|data availability|supplementary|ethics (approval|statement)|"
    r"abbreviations|editor note|references)",
    re.IGNORECASE,
)
# Inline elements whose text we drop: citation markers, and floats handled separately.
_DROP_INLINE = {"fig", "table-wrap", "supplementary-material", "disp-formula", "tex-math", "graphic"}


@dataclass
class JatsArticle:
    title: str
    pmcid: str | None = None
    pmid: str | None = None
    doi: str | None = None
    journal: str | None = None
    authors: str | None = None
    published_date: str | None = None
    license: str | None = None
    blocks: list[RawBlock] = field(default_factory=list)

    @property
    def has_body(self) -> bool:
        return any(b.heading_level is None and b.media_kind is None for b in self.blocks)


def _clean(text: str) -> str:
    return " ".join(text.split())


def _text(el: ET.Element | None) -> str:
    """Readable text of an element, without citation markers or nested floats."""
    if el is None:
        return ""
    parts: list[str] = []

    def walk(e: ET.Element):
        if e.tag in _DROP_INLINE:
            if e.tail:
                parts.append(e.tail)
            return
        is_citation = e.tag == "xref" and e.get("ref-type") in {"bibr", "fig", "table", "fn"}
        if e.text and not is_citation:
            parts.append(e.text)
        for child in e:
            walk(child)
        if e.tail and e is not el:
            parts.append(e.tail)

    walk(el)
    text = _clean("".join(parts))
    # Drop leftovers of removed citation markers ("( )", "[ , ]") and the space they leave before punctuation
    text = re.sub(r"\s*[\[(][\s,;–-]*[\])]", "", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()


def _cell(el: ET.Element) -> str:
    return _text(el).replace("|", "\\|") or " "


def table_to_markdown(table: ET.Element | None) -> str | None:
    if table is None:
        return None
    rows: list[list[str]] = []
    header_rows = 0
    for section in ("thead", "tbody", "tfoot"):
        for tr in table.iter(section):
            for row in tr.iter("tr"):
                cells = []
                for c in row:
                    if c.tag in ("th", "td"):
                        span = int(c.get("colspan") or 1) if (c.get("colspan") or "1").isdigit() else 1
                        cells.extend([_cell(c)] * max(span, 1))
                rows.append(cells)
                if section == "thead":
                    header_rows += 1
    if not rows:  # tables without thead/tbody
        rows = [[_cell(c) for c in tr if c.tag in ("th", "td")] for tr in table.iter("tr")]
        header_rows = 1 if rows else 0
    rows = [r for r in rows if r]
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [" "] * (width - len(r)) for r in rows]
    header = rows[0] if header_rows else [" "] * width
    body = rows[1:] if header_rows else rows
    extra = len(body) - MAX_TABLE_ROWS
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body[:MAX_TABLE_ROWS]]
    if extra > 0:
        lines.append(f"(... {extra} more rows not shown)")
    return "\n".join(lines)


def _figure_block(fig: ET.Element) -> RawBlock | None:
    label = _clean("".join(fig.findtext("label") or "")) or None
    caption_el = fig.find("caption")
    caption = _text(caption_el) if caption_el is not None else ""
    graphic = fig.find(".//graphic")
    href = graphic.get(XLINK_HREF) if graphic is not None else None
    if not (caption or label):
        return None
    return RawBlock(
        text=caption or label or "Figure", media_kind=NodeKind.FIGURE, label=label, asset_ref=href
    )


def _table_block(tw: ET.Element) -> RawBlock | None:
    label = _clean(tw.findtext("label") or "") or None
    caption_el = tw.find("caption")
    caption = _text(caption_el) if caption_el is not None else ""
    markdown = table_to_markdown(tw.find(".//table"))
    foot = _text(tw.find("table-wrap-foot"))
    if foot:
        markdown = f"{markdown}\nNote: {foot}" if markdown else f"Note: {foot}"
    if not (caption or markdown):
        return None
    graphic = tw.find("graphic")  # some tables are only an image
    return RawBlock(
        text=caption or label or "Table",
        media_kind=NodeKind.TABLE,
        label=label,
        table_markdown=markdown,
        asset_ref=graphic.get(XLINK_HREF) if graphic is not None else None,
    )


def _content(el: ET.Element, level: int, out: list[RawBlock]) -> None:
    """Walk the children of a section-like element, appending blocks in reading order."""
    for child in el:
        tag = child.tag
        if tag == "sec":
            title = _text(child.find("title"))
            if (child.get("sec-type") or "").lower() in _SKIP_SEC_TYPES or (
                title and _SKIP_TITLES.match(title)
            ):
                continue
            if title:
                out.append(RawBlock(text=title, heading_level=min(level, 6)))
            _content(child, level + 1 if title else level, out)
        elif tag == "p":
            text = _text(child)
            if text:
                out.append(RawBlock(text=text))
            # figures/tables can sit inside a paragraph
            for nested in child:
                if nested.tag in ("fig", "table-wrap", "list"):
                    _content(_wrap(nested), level, out)
        elif tag == "list":
            for item in child.iter("list-item"):
                text = _text(item)
                if text:
                    out.append(RawBlock(text=text, is_list_item=True))
        elif tag == "fig":
            if b := _figure_block(child):
                out.append(b)
        elif tag == "fig-group":
            _content(child, level, out)
        elif tag == "table-wrap":
            if b := _table_block(child):
                out.append(b)
        elif tag in ("boxed-text", "disp-quote", "table-wrap-group"):
            _content(child, level, out)


def _wrap(el: ET.Element) -> ET.Element:
    holder = ET.Element("holder")
    holder.append(el)
    return holder


def _meta(article: ET.Element) -> dict:
    meta = article.find("front/article-meta")
    ids = {}
    if meta is not None:
        ids = {i.get("pub-id-type"): (i.text or "").strip() for i in meta.findall("article-id")}
    pmcid = ids.get("pmcid") or (f"PMC{ids['pmc']}" if ids.get("pmc") else None)
    if pmcid and not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    authors = []
    if meta is not None:
        for contrib in meta.findall("contrib-group/contrib[@contrib-type='author']"):
            name = contrib.find("name")
            if name is not None:
                authors.append(
                    _clean(f"{name.findtext('surname') or ''} {name.findtext('given-names') or ''}")
                )
            elif contrib.findtext("collab"):
                authors.append(_clean(contrib.findtext("collab")))
    date = None
    if meta is not None:
        for pd in meta.findall("pub-date"):
            year = pd.findtext("year")
            if year:
                month, day = pd.findtext("month"), pd.findtext("day")
                date = "-".join(x.zfill(2) for x in [year, month, day] if x)
                if (pd.get("pub-type") or pd.get("date-type")) in ("epub", "pub"):
                    break
    lic = None
    if meta is not None:
        lic_el = meta.find("permissions/license")
        if lic_el is not None:
            ref = next((e.text for e in lic_el.iter() if e.tag.endswith("license_ref") and e.text), None)
            # Some publishers (e.g. MDPI) only link the licence inside the text: <ext-link xlink:href=...>
            linked = next(
                (h for e in lic_el.iter() if (h := e.get(XLINK_HREF)) and "creativecommons.org" in h), None
            )
            lic = ref or lic_el.get(XLINK_HREF) or linked or _text(lic_el)[:120] or None
    return {
        "title": _text(meta.find("title-group/article-title")) if meta is not None else "",
        "pmcid": pmcid,
        "pmid": ids.get("pmid"),
        "doi": ids.get("doi"),
        "journal": _clean(article.findtext("front/journal-meta/journal-title-group/journal-title") or "")
        or None,
        "authors": ", ".join(a for a in authors if a)[:2000] or None,
        "published_date": date,
        "license": lic,
    }


def parse_jats(data: bytes) -> JatsArticle:
    root = ET.fromstring(data)
    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        raise ValueError("Not a JATS article: no <article> element")
    info = JatsArticle(**_meta(article))

    blocks: list[RawBlock] = []
    abstract = article.find("front/article-meta/abstract")
    if abstract is not None:
        blocks.append(RawBlock(text="Abstract", heading_level=1))
        _content(abstract, 2, blocks)
    body = article.find("body")
    if body is not None:
        _content(body, 1, blocks)
    floats = article.find("floats-group")
    if floats is not None and len(floats):
        blocks.append(RawBlock(text="Figures and tables", heading_level=1))
        _content(floats, 2, blocks)
    info.blocks = [b for b in blocks if b.text]
    if not info.title:
        info.title = info.pmcid or "Untitled article"
    return info
