"""Builds a PDF from a Markdown sample, with real font sizes and an embedded figure, to exercise
the PDF parser (headings from fonts, figures with captions).

    python samples/make_pdf.py            # writes samples/dengue_guideline_v1.pdf
"""

from pathlib import Path

import pymupdf

STYLES = {"# ": (18, "hebo"), "## ": (14, "hebo"), "### ": (12, "hebo")}


def _figure_png() -> bytes:
    """A 240x160 two-colour bar chart, drawn without any image library."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 240, 160), False)
    pix.clear_with(255)
    for x0, x1, h, colour in ((40, 100, 120, (200, 60, 60)), (140, 200, 70, (60, 90, 200))):
        for x in range(x0, x1):
            for y in range(160 - h, 150):
                pix.set_pixel(x, y, colour)
    return pix.tobytes("png")


def md_to_pdf(md_path: Path, pdf_path: Path | None = None, with_figure: bool = True) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            y += 6
            continue
        size, font, text = 10, "helv", line
        for prefix, (s, f) in STYLES.items():
            if line.startswith(prefix):
                size, font, text = s, f, line[len(prefix) :]
        rect = pymupdf.Rect(60, y, 540, y + 200)
        used = page.insert_textbox(rect, text, fontsize=size, fontname=font)
        height = 200 - used if used >= 0 else size * 2
        y += height + 8
        if y > 600:
            page, y = doc.new_page(), 60
    if with_figure:
        page.insert_image(pymupdf.Rect(60, y + 10, 300, y + 170), stream=_figure_png())
        page.insert_textbox(
            pymupdf.Rect(60, y + 175, 540, y + 215),
            "Figure 1. Platelet count before and after recovery (synthetic).",
            fontsize=9,
            fontname="helv",
        )
    data = doc.tobytes()
    if pdf_path:
        pdf_path.write_bytes(data)
    return data


if __name__ == "__main__":
    here = Path(__file__).parent
    md_to_pdf(here / "dengue_guideline_v1.md", here / "dengue_guideline_v1.pdf")
    print("wrote", here / "dengue_guideline_v1.pdf")
