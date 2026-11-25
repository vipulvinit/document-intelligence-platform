"""OCR fallback for scanned PDF pages. Optional: needs `pytesseract` and the Tesseract binary."""

from app.core.logging import logger


def ocr_page(page) -> str:
    try:
        import io

        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        pix = page.get_pixmap(dpi=300)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(image)
    except Exception as exc:  # Tesseract missing or failed
        logger.warning(f"OCR failed: {exc}")
        return ""
