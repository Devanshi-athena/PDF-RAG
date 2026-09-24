from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .chat_service import ChatService
from .config import BackendSettings
from .errors import BackendError, NotFoundError, http_error
from .models import ChatRequest, ChatResponse, DocumentInfo, Message, Thread, ThreadCreate, TranscriptionResponse
from .pdf_service import chunk_pages, extract_pages
from .speech_service import transcribe
from .thread_store import ThreadStore

logger = logging.getLogger("pdf_rag.backend")


def create_app(settings: BackendSettings | None = None, thread_store=None, chat_service=None) -> FastAPI:
    settings = settings or BackendSettings.from_env()
    settings.ensure_directories()
    threads = thread_store or ThreadStore(settings.threads_file)
    chat = chat_service or ChatService(settings)
    app = FastAPI(title="PDF RAG Assistant API", version="1.0")
    app.state.settings = settings
    app.state.threads = threads
    app.state.chat = chat
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(BackendError)
    async def backend_error_handler(request: Request, error: BackendError):
        return await _json_error(error)

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, error: RuntimeError):
        return await _json_error(BackendError(str(error), 503, "service_unavailable"))

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "pdf-rag-assistant"}

    @app.post("/api/threads", response_model=Thread, status_code=201)
    def create_thread(payload: ThreadCreate | None = None):
        return threads.create(payload.name if payload else None)

    @app.get("/api/threads", response_model=list[Thread])
    def list_threads():
        return threads.list()

    @app.get("/api/threads/{thread_id}", response_model=Thread)
    def get_thread(thread_id: str):
        return threads.get(thread_id)

    @app.delete("/api/threads/{thread_id}", status_code=204)
    def delete_thread(thread_id: str):
        removed = threads.get(thread_id)
        try:
            chat.store(removed.id).delete()
        except Exception as error:
            logger.exception("Failed to delete Chroma collection for thread %s", removed.id)
            raise http_error(
                BackendError(
                    "The thread could not be fully deleted from vector storage. Try again.",
                    500,
                    "thread_delete_failed",
                )
            ) from error
        path = settings.documents_dir / f"{removed.id}.pdf"
        path.unlink(missing_ok=True)
        threads.delete(thread_id)
        return None

    @app.post("/api/threads/{thread_id}/document", response_model=Thread)
    async def upload_document(thread_id: str, file: UploadFile = File(...)):
        thread = threads.get(thread_id)
        if file.content_type not in ("application/pdf", "application/x-pdf", None):
            raise http_error(BackendError("Only PDF files are accepted.", 415, "invalid_file_type"))
        data = await file.read(settings.max_pdf_bytes + 1)
        if len(data) > settings.max_pdf_bytes:
            raise http_error(BackendError("The PDF exceeds the configured size limit.", 413, "file_too_large"))
        try:
            pages = extract_pages(data)
            chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
            chat.store(thread.id).replace(chunks)
        except BackendError:
            raise
        except ValueError as error:
            raise http_error(BackendError(str(error), 422, "invalid_chunk_config"))
        path = settings.documents_dir / f"{thread.id}.pdf"
        path.write_bytes(data)
        name = Path(file.filename or "document.pdf").name
        now = datetime.now(timezone.utc).isoformat()
        thread.pdf_name = name
        thread.name = Path(name).stem[:40] or thread.name
        thread.document = DocumentInfo(name=name, pages=len(pages), chunks=len(chunks), path=str(path), uploaded_at=now)
        threads.save(thread)
        return thread

    @app.get("/api/threads/{thread_id}/document", response_model=DocumentInfo)
    def get_document(thread_id: str):
        thread = threads.get(thread_id)
        if not thread.document:
            raise NotFoundError("This thread has no uploaded document.")
        return thread.document

    @app.post("/api/threads/{thread_id}/chat", response_model=ChatResponse)
    def ask_question(thread_id: str, payload: ChatRequest):
        thread = threads.get(thread_id)
        if not thread.document:
            raise http_error(BackendError("Upload a PDF before asking questions.", 409, "document_required"))
        user_message = Message(role="user", content=payload.question)
        thread.messages.append(user_message)
        try:
            answer, sources = chat.ask(thread, payload.question)
        except BackendError:
            thread.messages.pop()
            raise
        assistant = Message(role="assistant", content=answer, sources=sources)
        thread.messages.append(assistant)
        threads.save(thread)
        return ChatResponse(answer=answer, sources=sources, message=assistant)

    @app.post("/api/threads/{thread_id}/chat/stream")
    def stream_question(thread_id: str, payload: ChatRequest):
        from fastapi.responses import StreamingResponse
        import json

        thread = threads.get(thread_id)
        if not thread.document:
            raise http_error(BackendError("Upload a PDF before asking questions.", 409, "document_required"))
        user_message = Message(role="user", content=payload.question)
        thread.messages.append(user_message)
        try:
            tokens, sources = chat.stream(thread, payload.question)
        except BackendError:
            thread.messages.pop()
            raise

        def events():
            answer_parts = []
            try:
                for token in tokens:
                    answer_parts.append(token)
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"
                answer = "".join(answer_parts).strip()
                final_sources = [] if answer == "I couldn't find that in the uploaded PDF." else sources
                assistant = Message(role="assistant", content=answer, sources=final_sources)
                thread.messages.append(assistant)
                threads.save(thread)
                yield f"event: done\ndata: {json.dumps({'answer': answer, 'sources': [source.model_dump() for source in final_sources], 'message': assistant.model_dump()})}\n\n"
            except BackendError as error:
                thread.messages.pop()
                yield f"event: error\ndata: {json.dumps({'message': error.message, 'code': error.code})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/api/threads/{thread_id}/transcribe", response_model=TranscriptionResponse)
    async def transcribe_audio(thread_id: str, file: UploadFile = File(...)):
        threads.get(thread_id)
        data = await file.read(10 * 1024 * 1024 + 1)
        if len(data) > 10 * 1024 * 1024:
            raise http_error(BackendError("The audio recording is too large.", 413, "file_too_large"))
        return TranscriptionResponse(text=transcribe(settings, data, file.content_type))

    return app


async def _json_error(error: BackendError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=error.status_code, content={"detail": {"code": error.code, "message": error.message}})


app = create_app()
