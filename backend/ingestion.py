from __future__ import annotations

import time
from dataclasses import dataclass

from .document_structure import annotate_chunks, build_structure
from .pdf_service import chunk_document, extract_document


@dataclass
class IngestResult:
    pages: int
    chunks: int
    sections: int
    units: int


def ingest_pdf(data: bytes, store, chunk_size: int, chunk_overlap: int) -> IngestResult:
    """Extract, detect structure, chunk by section, embed and index one PDF into a thread store."""
    started = time.perf_counter()
    parsed = extract_document(data)
    print(f"[PERF] pdf_extraction_seconds={time.perf_counter() - started:.4f}")
    print(f"[PERF] pages_processed={len(parsed.pages)}")
    structure = build_structure(parsed)
    chunking_started = time.perf_counter()
    chunks = chunk_document(parsed, chunk_size, chunk_overlap)
    annotate_chunks(chunks, structure)
    print(f"[PERF] chunking_seconds={time.perf_counter() - chunking_started:.4f}")
    print(f"[PERF] chunks_created={len(chunks)}")
    index_started = time.perf_counter()
    try:
        store.replace(chunks, structure)
    finally:
        print(f"[PERF] embedding_indexing_call_seconds={time.perf_counter() - index_started:.4f}")
    return IngestResult(
        pages=structure["page_count"],
        chunks=len(chunks),
        sections=len(structure["sections"]),
        units=len(structure["units"]),
    )
