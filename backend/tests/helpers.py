"""Shared fakes: an in-memory Chroma collection, a deterministic embedder, a recording LLM and a PDF builder."""
from __future__ import annotations

import hashlib
import math
import re
from types import SimpleNamespace

import pymupdf

from backend.vector_store import STOP_WORDS, tokenize


class MemoryCollection:
    name = "thread_memory"

    def __init__(self):
        self.records = {}

    def get(self, ids=None, include=None):
        include = include if include is not None else ["documents", "metadatas"]
        items = [(key, value) for key, value in self.records.items() if ids is None or key in ids]
        result = {"ids": [key for key, _ in items]}
        if "documents" in include:
            result["documents"] = [value["document"] for _, value in items]
        if "metadatas" in include:
            result["metadatas"] = [value["metadata"] for _, value in items]
        if "embeddings" in include:
            result["embeddings"] = [value["embedding"] for _, value in items]
        return result

    def add(self, ids, documents, metadatas, embeddings):
        for key, document, metadata, embedding in zip(ids, documents, metadatas, embeddings):
            self.records[key] = {"document": document, "metadata": metadata, "embedding": embedding}

    def delete(self, ids):
        for key in ids:
            self.records.pop(key, None)

    def count(self):
        return len(self.records)


class MemoryClient:
    def __init__(self, collection=None):
        self.collection = collection or MemoryCollection()
        self.deleted = False

    def get_or_create_collection(self, name):
        return self.collection

    def delete_collection(self, name):
        self.deleted = True
        self.collection.records.clear()


class HashEmbeddings:
    """Bag-of-words hashing embedder: similar wording -> similar vectors, unrelated text -> ~0."""

    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in tokenize(text):
                if token in STOP_WORDS:
                    continue
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimensions
                vector[bucket] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class RecordingLLM:
    def __init__(self, answer=None):
        self.calls = []
        self.answer = answer

    def _text(self, kwargs):
        if self.answer is not None:
            return self.answer
        prompt = kwargs["messages"][-1]["content"]
        page = re.search(r"\[Page (\d+)\]", prompt)
        return f"Grounded answer [Page {page.group(1) if page else 1}]."

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        text = self._text(kwargs)
        if kwargs.get("stream"):
            return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def prompt(self, index=-1):
        return self.calls[index]["messages"][-1]["content"]


def build_pdf(pages: list[list[tuple[str, float, bool]]], header: str | None = None, page_numbers: bool = False) -> bytes:
    """pages: list of [(text, font size, bold)] lines, laid out top to bottom."""
    document = pymupdf.open()
    for number, lines in enumerate(pages, start=1):
        page = document.new_page(width=595, height=842)
        if header:
            page.insert_text((72, 30), header, fontsize=8, fontname="helv")
        y = 80
        for text, size, bold in lines:
            y += size * 1.6
            page.insert_text((72, y), text, fontsize=size, fontname="hebo" if bold else "helv")
        if page_numbers:
            page.insert_text((290, 825), str(number), fontsize=8, fontname="helv")
    data = document.tobytes()
    document.close()
    return data


def build_layout_pdf(pages: list[list[tuple[float, float, str, float, bool]]]) -> bytes:
    """pages: list of [(x, y, text, font size, bold)] placed exactly (for columns and tables)."""
    document = pymupdf.open()
    for items in pages:
        page = document.new_page(width=595, height=842)
        for x, y, text, size, bold in items:
            page.insert_text((x, y), text, fontsize=size, fontname="hebo" if bold else "helv")
    data = document.tobytes()
    document.close()
    return data


def body(sentence: str, lines: int = 6) -> list[tuple[str, float, bool]]:
    return [(f"{sentence} line {index}.", 11, False) for index in range(1, lines + 1)]
