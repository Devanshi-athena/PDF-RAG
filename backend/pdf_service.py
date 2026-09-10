from __future__ import annotations

import io
from typing import Callable

from .errors import BackendError
from .ocr_service import extract_page_text


def extract_pages(pdf_bytes: bytes, ocr: Callable | None = None) -> list[tuple[int, str]]:
    try:
        import fitz
        document = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    except Exception as error:
        raise BackendError("The uploaded file is not a readable PDF.", 422, "invalid_pdf") from error
    pages: list[tuple[int, str]] = []
    try:
        for number, page in enumerate(document, start=1):
            text = (page.get_text("text") or "").strip()
            if not text:
                text = (ocr or extract_page_text)(page)
            if text:
                pages.append((number, text))
    finally:
        document.close()
    if not pages:
        raise BackendError("No text could be extracted from this PDF.", 422, "empty_document")
    return pages


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int = 1200, overlap: int = 200) -> list[dict]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
    chunks: list[dict] = []
    for page, text in pages:
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            value = text[start:end].strip()
            if value:
                chunks.append({"id": f"p{page}-{len(chunks)}", "text": value, "page": page})
            if end == len(text):
                break
            start = end - overlap
    return chunks
