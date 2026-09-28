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
    asr_model: str = "openai/whisper-large-v3-turbo"
    asr_url: str = "https://router.huggingface.co/hf-inference"
    hf_provider: str = ""
    chroma_dir: Path = Path("data/chroma")
    threads_file: Path = Path("data/threads.json")
    documents_dir: Path = Path("data/documents")
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    temperature: float = 0.0
    max_pdf_bytes: int = 25 * 1024 * 1024

    @classmethod
    def from_env(cls) -> "BackendSettings":
        return cls(
            hf_token=os.getenv("HF_TOKEN", ""),
            chat_model=os.getenv("HF_CHAT_MODEL", cls.chat_model),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            asr_model=os.getenv("HF_ASR_MODEL", cls.asr_model),
            asr_url=os.getenv("HF_ASR_URL", cls.asr_url),
            hf_provider=os.getenv("HF_PROVIDER", ""),
            chroma_dir=Path(os.getenv("CHROMA_DIR", str(cls.chroma_dir))),
            threads_file=Path(os.getenv("THREADS_FILE", str(cls.threads_file))),
            documents_dir=Path(os.getenv("DOCUMENTS_DIR", str(cls.documents_dir))),
            chunk_size=_int_env("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_int_env("CHUNK_OVERLAP", cls.chunk_overlap, 0),
            top_k=_int_env("TOP_K", cls.top_k),
            temperature=_float_env("TEMPERATURE", cls.temperature),
            max_pdf_bytes=_int_env("MAX_PDF_BYTES", cls.max_pdf_bytes),
        )

    def ensure_directories(self) -> None:
        self.threads_file.parent.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
