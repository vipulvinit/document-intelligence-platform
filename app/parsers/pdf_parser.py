"""PDF parsing with PyMuPDF.

Text keeps its font size and boldness so headings can be inferred. Images become figure blocks
(their caption is the nearest text below that starts with "Figure"/"Fig."), and tables found
by PyMuPDF's table finder become table blocks with Markdown; text inside a table is not repeated
as paragraphs.
"""

import re
import statistics

import pymupdf

from app.core.constants import NodeKind
from app.core.logging import logger
from app.parsers.ocr import ocr_page
from app.schemas.node import RawBlock

MIN_IMAGE_SIDE = 80  # pixels: smaller images are logos, icons or decorations
_CAPTION = re.compile(r"^(fig(?:ure)?\.?\s*\d+[a-z]?)[.:\s]", re.IGNORECASE)
_TABLE_CAPTION = re.compile(r"^(table\s*\d+)[.:\s]", re.IGNORECASE)


def parse_pdf(data: bytes) -> list[RawBlock]:
    blocks: list[RawBlock] = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for page_no, page in enumerate(pdf, start=1):
            page_blocks = _page_blocks(page, page_no)
            if not any(b.media_kind is None for b in page_blocks):  # scanned page: no text layer
                text = ocr_page(page)
                if text:
                    page_blocks += [RawBlock(text=t, page=page_no) for t in text.split("\n\n") if t.strip()]
                elif not page_blocks:
                    logger.warning(f"Page {page_no} has no text layer and OCR is unavailable")
            blocks.extend(page_blocks)
    return [b for b in blocks if b.text]


def _inside(bbox, rects) -> bool:
    r = pymupdf.Rect(bbox)
    return any(r.intersects(t) and (r & t).get_area() > 0.5 * r.get_area() for t in rects)


def _find_tables(page) -> list[tuple[pymupdf.Rect, str]]:
    try:
        found = page.find_tables()
    except Exception as exc:  # the table finder is heuristic; never let it break parsing
        logger.debug(f"Table detection failed on page {page.number + 1}: {exc}")
        return []
    out = []
    for t in found.tables:
        try:
            md = t.to_markdown().strip()
        except Exception:
            continue
        if md and t.row_count >= 2:
            out.append((pymupdf.Rect(t.bbox), md))
    return out


def _page_blocks(page, page_no: int) -> list[RawBlock]:
    tables = _find_tables(page)
    table_rects = [r for r, _ in tables]
    raw = page.get_text("dict")["blocks"]

    texts = []  # (bbox, RawBlock) for text outside tables
    images = []  # (bbox, bytes, ext)
    for block in raw:
        if block.get("type") == 1:
            if min(block.get("width", 0), block.get("height", 0)) >= MIN_IMAGE_SIDE and block.get("image"):
                images.append((block["bbox"], block["image"], block.get("ext", "png")))
            continue
        if block.get("type") != 0 or _inside(block["bbox"], table_rects):
            continue
        spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        text = " ".join(" ".join(s["text"] for s in line["spans"]) for line in block["lines"])
        sizes = [s["size"] for s in spans]
        bold = all(("bold" in s["font"].lower()) or (s["flags"] & 16) for s in spans)
        stripped = text.strip()
        is_item = stripped[:1] in {"•", "-", "–", "*"}
        texts.append(
            (
                block["bbox"],
                RawBlock(
                    text=stripped.lstrip("•-–* ") if is_item else stripped,
                    page=page_no,
                    font_size=round(statistics.median(sizes), 1),
                    bold=bool(bold),
                    is_list_item=is_item,
                ),
            )
        )

    used_captions: set[int] = set()
    media: list[tuple[float, RawBlock]] = []

    for bbox, image, ext in images:
        label, caption = None, ""
        below = sorted(
            (i for i, (tb, b) in enumerate(texts) if tb[1] >= bbox[3] - 5 and _CAPTION.match(b.text)),
            key=lambda i: texts[i][0][1] - bbox[3],
        )
        if below and texts[below[0]][0][1] - bbox[3] < 120:
            i = below[0]
            used_captions.add(i)
            m = _CAPTION.match(texts[i][1].text)
            label, caption = m.group(1), texts[i][1].text[m.end() :].strip()
        media.append(
            (
                bbox[1],
                RawBlock(
                    text=caption or label or f"Image on page {page_no}",
                    page=page_no,
                    media_kind=NodeKind.FIGURE,
                    label=label,
                    asset_bytes=image,
                    asset_ext=ext,
                ),
            )
        )

    for rect, md in tables:
        label, caption = None, ""
        above = [
            i for i, (tb, b) in enumerate(texts) if 0 <= rect.y0 - tb[3] < 60 and _TABLE_CAPTION.match(b.text)
        ]
        if above:
            i = above[-1]
            used_captions.add(i)
            m = _TABLE_CAPTION.match(texts[i][1].text)
            label, caption = m.group(1), texts[i][1].text[m.end() :].strip()
        media.append(
            (
                rect.y0,
                RawBlock(
                    text=caption or label or f"Table on page {page_no}",
                    page=page_no,
                    media_kind=NodeKind.TABLE,
                    label=label,
                    table_markdown=md,
                ),
            )
        )

    # Keep PyMuPDF's text order (it handles columns better than sorting by position) and slot each
    # figure or table in before the first text block that starts below it.
    out: list[RawBlock] = []
    media.sort(key=lambda t: t[0])
    j = 0
    for i, (bbox, block) in enumerate(texts):
        while j < len(media) and media[j][0] <= bbox[1]:
            out.append(media[j][1])
            j += 1
        if i not in used_captions:
            out.append(block)
    out.extend(b for _, b in media[j:])
    return out
