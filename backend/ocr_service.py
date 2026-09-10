from __future__ import annotations

from .errors import OCRUnavailable


def extract_page_text(page) -> str:
    """OCR one PyMuPDF page only when normal extraction returned no text."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise OCRUnavailable(
            "This PDF appears to be scanned. Install pytesseract and Pillow, "
            "and install the Tesseract OCR executable, then retry."
        ) from error
    try:
        pixmap = page.get_pixmap(matrix=__import__("fitz").Matrix(2, 2), alpha=False)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return (pytesseract.image_to_string(image) or "").strip()
    except Exception as error:
        raise OCRUnavailable(
            "OCR could not process this page. Check that the Tesseract executable is installed."
        ) from error
