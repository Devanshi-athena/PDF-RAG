# PDF RAG Assistant

A local PDF question-answering app with a React/Vite frontend and FastAPI backend.
Each thread has one PDF, its own chat history, and an isolated ChromaDB collection.

## Stack

- Frontend: React, Vite
- Backend: FastAPI, Uvicorn
- PDF extraction: PyMuPDF
- OCR fallback: Tesseract via pytesseract and Pillow
- Chunking: 1,200-character chunks with 200-character overlap
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector database: persistent ChromaDB
- LLM: Hugging Face `meta-llama/Llama-3.1-8B-Instruct`
- Speech-to-text: Hugging Face Whisper `openai/whisper-large-v3-turbo`
- Text-to-speech: browser `SpeechSynthesis`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `HF_TOKEN` in `.env`. Do not commit `.env`.

Install Tesseract OCR separately and verify:

```powershell
tesseract --version
```

Start the backend from the repository root:

```powershell
uvicorn backend.main:app --reload
```

Start the frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

The frontend uses `http://localhost:8000/api` by default. Set `VITE_API_URL` if required.

## Application flow

1. Create a thread and upload one PDF.
2. The UI shows `Processing...` with a spinner while extraction, OCR, chunking, embedding, and indexing run.
3. Ask a text question or record an English voice question.
4. Whisper transcribes voice input and sends the transcript to chat.
5. The backend retrieves the top five chunks from the active thread collection.
6. The LLM receives those excerpts and recent thread history.
7. The answer and PDF page/source excerpts are displayed.
8. `Listen` uses browser speech synthesis.
9. Switching threads changes the PDF, history, and vector collection used.

## Configuration

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token |
| `HF_CHAT_MODEL` | Chat model |
| `HF_ASR_MODEL` | Speech-to-text model |
| `HF_ASR_URL` | Speech-to-text endpoint |
| `HF_PROVIDER` | Optional Hugging Face provider |
| `EMBEDDING_MODEL` | Sentence Transformer model |
| `CHUNK_SIZE` | Chunk size; default `1200` |
| `CHUNK_OVERLAP` | Chunk overlap; default `200` |
| `TOP_K` | Retrieved chunks; default `5` |
| `TEMPERATURE` | LLM temperature |
| `MAX_PDF_BYTES` | Upload size limit |
| `CHROMA_DIR` | Chroma storage path |
| `THREADS_FILE` | Thread metadata path |
| `DOCUMENTS_DIR` | PDF storage path |

## Thread isolation

Each thread has a UUID. The PDF is stored as `data/documents/<thread_id>.pdf`,
and Chroma uses a collection named `thread_<thread_id_without_hyphens>`.
Retrieval uses only the active thread ID. Deleting a thread removes its JSON
metadata, PDF, and Chroma collection.

## API

- `GET /api/health`
- `POST /api/threads`
- `GET /api/threads`
- `GET /api/threads/{id}`
- `DELETE /api/threads/{id}`
- `POST /api/threads/{id}/document`
- `GET /api/threads/{id}/document`
- `POST /api/threads/{id}/chat`
- `POST /api/threads/{id}/transcribe`

## Grounding behavior

The chat prompt instructs the LLM to use only retrieved PDF excerpts, cite pages,
and return `I couldn't find that in the uploaded PDF.` when the excerpts do not
contain the answer. Retrieved sources include the PDF name, page number, and excerpt.

## Validation

```powershell
pytest backend\tests
cd frontend
npm run build
```

The backend tests cover API behavior, chunk page metadata, thread isolation, and
delete cleanup. The active application is in `backend/` and `frontend/`; the old
Streamlit prototype and previous runtime data are archived under `dump/`.

## Known limitations

- JSON storage is intended for local single-user use.
- Tesseract must be installed for scanned-page OCR.
- Hugging Face availability depends on the selected model/provider.
- Browser speech quality depends on installed browser/OS voices.
- No authentication or hosted deployment is included.
