"""Builds samples/sop_v1.pdf from sop_v1.md with real font sizes, to exercise the PDF parser."""

from pathlib import Path

import pymupdf

STYLES = {"# ": (18, "hebo"), "## ": (14, "hebo"), "### ": (12, "hebo")}


def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    for line in md_path.read_text().splitlines():
        if not line.strip():
            y += 6
            continue
        size, font, text = 10, "helv", line
        for prefix, (s, f) in STYLES.items():
            if line.startswith(prefix):
                size, font, text = s, f, line[len(prefix) :]
        if text.startswith("- "):
            text = "- " + text[2:]
        rect = pymupdf.Rect(60, y, 540, y + 200)
        used = page.insert_textbox(rect, text, fontsize=size, fontname=font)
        height = 200 - used if used >= 0 else size * 2
        y += height + 8
        if y > 760:
            page, y = doc.new_page(), 60
    doc.save(pdf_path)


if __name__ == "__main__":
    here = Path(__file__).parent
    md_to_pdf(here / "sop_v1.md", here / "sop_v1.pdf")
    print("wrote", here / "sop_v1.pdf")
