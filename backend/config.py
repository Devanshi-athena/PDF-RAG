from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class BackendSettings:
    hf_token: str = ""
    chat_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "hf"
    reranker_model: str = ""
    analysis_model: str = ""
    analysis_provider: str = ""
    embedding_provider: str = "hf-inference"
    asr_model: str = "openai/whisper-large-v3-turbo"
    asr_url: str = "https://router.huggingface.co/hf-inference"
    hf_provider: str = ""
    chroma_dir: Path = Path("data/chroma")
    threads_file: Path = Path("data/threads.json")
    documents_dir: Path = Path("data/documents")
    structure_dir: Path = Path("data/structure")
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_batch_size: int = 32
    embedding_workers: int = 4
    top_k: int = 6
    retrieval_min_similarity: float = 0.30
    qa_context_chars: int = 14000
    analysis_context_chars: int = 22000
    explain_context_chars: int = 28000
    range_unit_context_chars: int = 8000
    overview_context_chars: int = 20000
    max_answer_tokens: int = 1000
    llm_workers: int = 8
    max_units_per_request: int = 25
    temperature: float = 0.0
    max_pdf_bytes: int = 50 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "BackendSettings":
        return cls(
            hf_token=os.getenv("HF_TOKEN", ""),
            chat_model=os.getenv("HF_CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            embedding_backend=(os.getenv("EMBEDDING_BACKEND", cls.embedding_backend).strip().casefold() or "hf"),
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model).strip(),
            analysis_model=os.getenv("HF_ANALYSIS_MODEL", cls.analysis_model).strip(),
            analysis_provider=os.getenv("HF_ANALYSIS_PROVIDER", cls.analysis_provider).strip(),
            embedding_provider=os.getenv("HF_EMBEDDING_PROVIDER", cls.embedding_provider).strip(),
            asr_model=os.getenv("HF_ASR_MODEL", cls.asr_model),
            asr_url=os.getenv("HF_ASR_URL", cls.asr_url),
            hf_provider=os.getenv("HF_PROVIDER", ""),
            chroma_dir=Path(os.getenv("CHROMA_DIR", str(cls.chroma_dir))),
            threads_file=Path(os.getenv("THREADS_FILE", str(cls.threads_file))),
            documents_dir=Path(os.getenv("DOCUMENTS_DIR", str(cls.documents_dir))),
            structure_dir=Path(os.getenv("STRUCTURE_DIR", str(cls.structure_dir))),
            chunk_size=_int_env("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int_env("CHUNK_OVERLAP", cls.chunk_overlap, 0),
            embedding_batch_size=_int_env("EMBEDDING_BATCH_SIZE", cls.embedding_batch_size),
            embedding_workers=_int_env("EMBEDDING_WORKERS", cls.embedding_workers),
            top_k=_int_env("TOP_K", cls.top_k),
            retrieval_min_similarity=_float_env("RETRIEVAL_MIN_SIMILARITY", cls.retrieval_min_similarity),
            qa_context_chars=_int_env("QA_CONTEXT_CHARS", cls.qa_context_chars, 2000),
            analysis_context_chars=_int_env("ANALYSIS_CONTEXT_CHARS", cls.analysis_context_chars, 2000),
            explain_context_chars=_int_env("EXPLAIN_CONTEXT_CHARS", cls.explain_context_chars, 2000),
            range_unit_context_chars=_int_env("RANGE_UNIT_CONTEXT_CHARS", cls.range_unit_context_chars, 1000),
            overview_context_chars=_int_env("OVERVIEW_CONTEXT_CHARS", cls.overview_context_chars, 2000),
            max_answer_tokens=_int_env("MAX_ANSWER_TOKENS", cls.max_answer_tokens, 128),
            llm_workers=_int_env("LLM_WORKERS", cls.llm_workers),
            max_units_per_request=_int_env("MAX_UNITS_PER_REQUEST", cls.max_units_per_request),
            temperature=_float_env("TEMPERATURE", cls.temperature),
            max_pdf_bytes=_int_env("MAX_PDF_BYTES", cls.max_pdf_bytes),
        )

    def ensure_directories(self) -> None:
        self.threads_file.parent.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.structure_dir.mkdir(parents=True, exist_ok=True)
