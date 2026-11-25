"""PDF parsing with PyMuPDF: keeps font size and boldness so headings can be inferred."""

import statistics

import pymupdf

from app.core.logging import logger
from app.parsers.ocr import ocr_page
from app.schemas.node import RawBlock


def parse_pdf(data: bytes) -> list[RawBlock]:
    blocks: list[RawBlock] = []
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        for page_no, page in enumerate(pdf, start=1):
            page_blocks = _page_blocks(page, page_no)
            if not page_blocks:  # scanned page: no text layer
                text = ocr_page(page)
                if text:
                    page_blocks = [RawBlock(text=t, page=page_no) for t in text.split("\n\n") if t.strip()]
                else:
                    logger.warning(f"Page {page_no} has no text layer and OCR is unavailable")
            blocks.extend(page_blocks)
    return [b for b in blocks if b.text]


def _page_blocks(page, page_no: int) -> list[RawBlock]:
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # 0 = text, 1 = image
            continue
        spans = [s for line in block["lines"] for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        text = " ".join(" ".join(s["text"] for s in line["spans"]) for line in block["lines"])
        sizes = [s["size"] for s in spans]
        bold = all(("bold" in s["font"].lower()) or (s["flags"] & 16) for s in spans)
        stripped = text.strip()
        is_item = stripped[:1] in {"•", "-", "–", "*"}
        out.append(
            RawBlock(
                text=stripped.lstrip("•-–* ") if is_item else stripped,
                page=page_no,
                font_size=round(statistics.median(sizes), 1),
                bold=bool(bold),
                is_list_item=is_item,
            )
        )
    return out
