from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import BackendSettings
from backend.main import create_app
from backend.models import Source
from backend.pdf_service import ParsedDocument, chunk_pages
from backend.thread_store import ThreadStore


class FakeCollection:
    def __init__(self):
        self.chunks = []

    def replace(self, chunks, structure=None):
        self.chunks = chunks
        self.structure = structure

    def delete(self):
        self.chunks = []


class FakeChat:
    def __init__(self):
        self.collections = {}

    def store(self, thread_id):
        return self.collections.setdefault(thread_id, FakeCollection())

    def ask(self, thread, question):
        return "Grounded answer [Page 1]", [Source(page=1, text="A source", pdf_name=thread.pdf_name or "PDF")]


@pytest.fixture
def client(tmp_path, monkeypatch):
    settings = BackendSettings(
        chroma_dir=tmp_path / "chroma",
        threads_file=tmp_path / "threads.json",
        documents_dir=tmp_path / "documents",
    )
    app = create_app(settings=settings, thread_store=ThreadStore(settings.threads_file), chat_service=FakeChat())
    monkeypatch.setattr(
        "backend.ingestion.extract_document",
        lambda data: ParsedDocument(pages=[(1, "A page of text.")], page_count=1),
    )
    return TestClient(app)


def test_health_and_thread_ids(client):
    assert client.get("/api/health").json()["status"] == "ok"
    created = client.post("/api/threads", json={"name": "First"}).json()
    assert created["id"]
    assert client.get(f"/api/threads/{created['id']}").json()["name"] == "First"
    assert len(client.get("/api/threads").json()) == 1


def test_not_found_and_document_processing(client):
    assert client.get("/api/threads/not-a-thread").status_code == 404
    thread = client.post("/api/threads").json()
    response = client.post(
        f"/api/threads/{thread['id']}/document",
        files={"file": ("notes.pdf", b"fake pdf", "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["document"]["pages"] == 1
    assert response.json()["document"]["chunks"] == 1
    assert client.get(f"/api/threads/{thread['id']}/document").json()["name"] == "notes.pdf"


def test_upload_reports_document_ingestion_timings(client, capsys):
    thread = client.post("/api/threads").json()

    response = client.post(
        f"/api/threads/{thread['id']}/document",
        files={"file": ("notes.pdf", b"fake pdf", "application/pdf")},
    )

    assert response.status_code == 200
    logs = capsys.readouterr().out
    assert "[PERF] file_upload_start" in logs
    assert "[PERF] file_upload_seconds=" in logs
    assert "[PERF] pdf_extraction_seconds=" in logs
    assert "[PERF] pages_processed=1" in logs
    assert "[PERF] chunking_seconds=" in logs
    assert "[PERF] chunks_created=1" in logs
    assert "[PERF] embedding_indexing_call_seconds=" in logs
    assert "[PERF] total_document_ingestion_seconds=" in logs
    assert "[PERF] file_upload_end" in logs


def test_chat_is_thread_isolated_and_empty_question_rejected(client):
    first = client.post("/api/threads").json()
    second = client.post("/api/threads").json()
    for thread in (first, second):
        client.post(f"/api/threads/{thread['id']}/document", files={"file": ("a.pdf", b"x", "application/pdf")})
    assert client.post(f"/api/threads/{first['id']}/chat", json={"question": "What?"}).json()["answer"]
    assert len(client.get(f"/api/threads/{second['id']}").json()["messages"]) == 0
    assert client.post(f"/api/threads/{first['id']}/chat", json={"question": "  "}).status_code == 422


def test_chunk_page_metadata():
    chunks = chunk_pages([(3, "abcdefghij")], chunk_size=6, overlap=2)
    assert chunks[0]["page"] == 3
    assert chunks[0]["id"].startswith("p3-")


def test_delete_removes_thread_and_document(client, tmp_path):
    thread = client.post("/api/threads").json()
    response = client.post(
        f"/api/threads/{thread['id']}/document",
        files={"file": ("notes.pdf", b"fake pdf", "application/pdf")},
    )
    assert response.status_code == 200
    assert client.delete(f"/api/threads/{thread['id']}").status_code == 204
    assert client.get(f"/api/threads/{thread['id']}").status_code == 404
    assert client.get("/api/threads").json() == []
