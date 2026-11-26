"""Turns an uploaded file into raw blocks, choosing the parser by file type."""

import re

from app.parsers.jats_parser import parse_jats
from app.parsers.pdf_parser import parse_pdf
from app.schemas.node import RawBlock

_HEADING_MD = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_MD = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


class UnsupportedFileType(ValueError):
    pass


def extract_blocks(filename: str, data: bytes) -> list[RawBlock]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return parse_pdf(data)
    if name.endswith((".md", ".markdown", ".txt")):
        return parse_markdown(data.decode("utf-8", errors="replace"))
    if name.endswith(".xml"):
        return parse_jats(data).blocks
    raise UnsupportedFileType(f"Unsupported file type: {filename}. Use PDF, JATS XML, Markdown or TXT.")


def parse_markdown(text: str) -> list[RawBlock]:
    """Headings from '#' marks, list items from '-'/'1.', paragraphs split on blank lines."""
    blocks: list[RawBlock] = []
    paragraph: list[str] = []

    def flush():
        if paragraph:
            blocks.append(RawBlock(text=" ".join(paragraph)))
            paragraph.clear()

    for line in text.splitlines():
        if not line.strip():
            flush()
            continue
        if m := _HEADING_MD.match(line):
            flush()
            blocks.append(RawBlock(text=m.group(2), heading_level=len(m.group(1))))
        elif m := _LIST_MD.match(line):
            flush()
            blocks.append(RawBlock(text=m.group(1), is_list_item=True))
        else:
            paragraph.append(line.strip())
    flush()
    return [b for b in blocks if b.text]
