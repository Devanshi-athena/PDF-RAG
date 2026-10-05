# PDF RAG Assistant

A local PDF question-answering app with a React/Vite frontend and a FastAPI backend.
Each thread holds one PDF, its own chat history and its own isolated search index.
Answers come only from the uploaded PDF and cite page numbers.

It is built for long, structured documents - policies and guidelines, technical
reports, research papers and programme documents of hundreds of pages - and handles
direct questions, multi-part questions, comparisons, analysis questions, structural
questions ("how many chapters?", "list the annexes") and short or vague questions.

## Stack

| Part | Technology |
|---|---|
| Frontend | React, Vite; browser `SpeechSynthesis` for the Listen button |
| Backend | FastAPI, Uvicorn |
| PDF reading | PyMuPDF (text, fonts, positions); Tesseract OCR for scanned pages (optional) |
| Search index | ChromaDB (on disk) plus an in-memory NumPy vector index and BM25 keyword index |
| Models | Hugging Face Inference Providers, configured in `.env` (see below) |

## Models

All models and providers are set in `.env`; none are hard-coded. The defaults in
`.env.example` are the lowest-cost options, so a demo fits in a Hugging Face
account's monthly included credits.

| Role | Default | Provider | Required |
|---|---|---|---|
| Answers | `meta-llama/Llama-3.1-8B-Instruct` | `novita` | Yes |
| Embeddings (search by meaning) | `sentence-transformers/all-MiniLM-L6-v2` | `hf-inference` (or local) | Yes |
| Speech-to-text | `openai/whisper-large-v3-turbo` | `hf-inference` | Only for voice questions |
| Second model for analysis-type questions | empty = the answer model | `HF_ANALYSIS_PROVIDER` | No |
| Reranker | empty = off | runs locally | No |

Counting chapters, listing sections, finding where a topic is discussed and reading
tables are done by code, not by a model.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `HF_TOKEN` in `.env` (create one at https://huggingface.co/settings/tokens).
Do not commit `.env`.

Optional:

- Scanned PDFs: install the Tesseract OCR program and check `tesseract --version`.
- Local embeddings or a reranker (`EMBEDDING_BACKEND=hybrid` or `local`,
  `RERANKER_MODEL`): `pip install sentence-transformers`.

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

The frontend calls `http://localhost:8000/api` by default; set `VITE_API_URL` to change it.

## How it works

**Uploading a PDF** (once per document)

1. PyMuPDF reads every page with its layout. Headers, footers and page numbers are
   removed, wrapped lines are joined (also in two-column papers), ligatures are
   normalised, and tables are rebuilt as rows (`Name | Dept | Score`).
2. The document's structure is detected: chapters, parts, sections, annexes,
   articles and labelled items such as "Recommendation 11", with their numbers and
   pages, plus figure/table captions, tables and section roles (summary, methods,
   findings, recommendations, budget...). Bookmarks are used when the PDF has them,
   otherwise heading fonts and numbering. Contents pages and running titles are not
   treated as sections.
3. The text is cut into ~1,000-character pieces that never cross a section boundary.
4. Each piece is embedded and stored. The structure is saved as
   `data/structure/<thread_id>.json`.

**Asking a question** (every time)

1. The question is classified (`backend/query_router.py`):
   - **Structural** - counts and lists of chapters/annexes/recommendations/pages/tables,
     the table of contents, sub-sections of a section, titles, and "which section
     discusses X?" are answered from the saved structure instantly, without a model.
   - **References** - "chapter 3", "Part Two", "Annex B", "Article 5",
     "section 3.2.4", "AC-2", "Table 2", "page 12" or a section title (typos tolerated)
     focus the answer on that part of the document.
   - **Overview** - "what does the document contain?" uses the abstract/executive
     summary, the outline and the start of every chapter.
   - **Open questions** get a mode: direct, multi-part (each part searched
     separately), comparison (evidence gathered per side), analysis (wider context,
     inferences marked), enumeration, or ambiguous (each meaning covered).
2. Search combines meaning (embeddings) and keywords (BM25), then widens the best
   matches to their whole section or neighbouring text. If nothing matches, the app
   replies "I couldn't find that in the uploaded PDF." without calling a model.
3. The model receives only that text and must answer from it, cite `[Page N]` and say
   when something is not in the PDF. The answer streams to the screen with the cited
   pages as sources.

Comparisons ("compare", "difference between") are answered as a table with one column per
item and page citations in the cells; other answers use a table when several items share
the same attributes. The chat window renders these as real tables.

Answer length follows the question: "briefly" or "in 5 points" gives a short answer,
"in detail" a full one, and everything else only as long as needed. A guard stops the
model if it starts repeating itself. Long sections asked "in detail" are explained from
per-sub-section summaries, and chapter ranges
("explain chapters 10-20") are summarized chapter by chapter in parallel; both are
cached. Follow-up questions ("summarize it") refer back to the previous question's
section. Details are in [docs/architecture.md](docs/architecture.md).

## Configuration

All settings are read from `.env` by `backend/config.py`.

**Models and providers**

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face token |
| `HF_CHAT_MODEL` | Model that writes answers |
| `HF_PROVIDER` | Provider for the answer model (`novita` in the example); empty lets Hugging Face choose |
| `HF_ANALYSIS_MODEL` | Optional second model for analysis, comparison, multi-part, list and vague questions; empty = answer model |
| `HF_ANALYSIS_PROVIDER` | Provider for `HF_ANALYSIS_MODEL`; empty = `HF_PROVIDER` |
| `EMBEDDING_MODEL` | Embedding model |
| `HF_EMBEDDING_PROVIDER` | Provider for API embeddings; default `hf-inference` |
| `EMBEDDING_BACKEND` | `hf` (API), `hybrid` (documents via API, questions locally - fastest answers) or `local` (no API use; slower indexing) |
| `RERANKER_MODEL` | Optional local cross-encoder, e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `HF_ASR_MODEL` | Speech-to-text model |
| `HF_ASR_URL` | Speech-to-text endpoint |
| `TEMPERATURE` | Answer randomness; default `0` |

**Retrieval and answer size**

| Variable | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Size of indexed text pieces |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_WORKERS` | `32` / `4` | Embedding batch size and parallel requests during upload |
| `TOP_K` | `6` | Best matches used before widening the context |
| `RETRIEVAL_MIN_SIMILARITY` | `0.30` | Below this (and with few keyword matches) the app says "not found"; tuned for MiniLM |
| `QA_CONTEXT_CHARS` | `14000` | Context for normal questions |
| `ANALYSIS_CONTEXT_CHARS` | `22000` | Context for multi-part, comparison and analysis questions |
| `EXPLAIN_CONTEXT_CHARS` | `28000` | Context for explaining or comparing sections |
| `RANGE_UNIT_CONTEXT_CHARS` | `8000` | Context per chapter when summarizing a range |
| `OVERVIEW_CONTEXT_CHARS` | `20000` | Context for "what does the document contain" |
| `MAX_ANSWER_TOKENS` | `1000` | Answer length (explanations get up to 50% more) |
| `LLM_WORKERS` | `8` | Parallel model calls for chapter ranges and long sections |
| `MAX_UNITS_PER_REQUEST` | `25` | Most chapters summarized in one answer |

**Storage and limits**

| Variable | Default | Purpose |
|---|---|---|
| `CHROMA_DIR` | `data/chroma` | Search index |
| `STRUCTURE_DIR` | `data/structure` | Document structure and cached summaries |
| `DOCUMENTS_DIR` | `data/documents` | Uploaded PDFs |
| `THREADS_FILE` | `data/threads.json` | Threads and chat history |
| `MAX_PDF_BYTES` | `52428800` (50 MB) | Upload size limit |

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/threads` | Create a thread |
| GET | `/api/threads` | List threads |
| GET | `/api/threads/{id}` | Get a thread with its messages |
| DELETE | `/api/threads/{id}` | Delete a thread, its PDF, index and structure |
| POST | `/api/threads/{id}/document` | Upload and index a PDF |
| GET | `/api/threads/{id}/document` | Document details |
| POST | `/api/threads/{id}/chat` | Ask a question (complete answer) |
| POST | `/api/threads/{id}/chat/stream` | Ask a question (streamed answer; used by the frontend) |
| POST | `/api/threads/{id}/transcribe` | Speech-to-text for voice questions |

## Thread isolation

Each thread has a UUID. Its PDF is `data/documents/<thread_id>.pdf`, its index is the
Chroma collection `thread_<thread_id_without_hyphens>`, and its structure and cached
summaries are `data/structure/<thread_id>.json` and `.digests.json`. Questions search
only the active thread. Deleting a thread removes all of these and its messages.
Threads indexed by an older version are re-indexed from the saved PDF on their first
question.

## Tests

```powershell
pytest backend\tests
cd frontend
npm run build
```

The tests use generated PDFs and fake models (no Hugging Face calls). They cover
layout extraction (columns, tables, captions, contents pages, numbering), structure
detection, question routing, structural answers, context building for every
question type, abstention, source filtering, re-indexing, thread isolation and the API.

The app lives in `backend/` and `frontend/`. `dump/` (git-ignored) holds archived
material that the app does not use: the old Streamlit prototype, previous runtime
data, obsolete tests and caches.

## Known limitations

- Answer models are called through Hugging Face Inference Providers and use the
  account's credits; there is no fully offline answer model.
- `novita` serves Llama-3.1-8B with a 16k-token context; keep the `*_CONTEXT_CHARS`
  limits near their defaults with that provider.
- Images and charts are not read; figure questions use captions and surrounding text.
- Scanned pages need the Tesseract program installed.
- JSON thread storage is meant for local single-user use; there is no authentication.
- Browser speech quality depends on the browser and operating-system voices.
