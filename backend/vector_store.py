from __future__ import annotations

import math
import os
from pathlib import Path
from numbers import Real

from .errors import BackendError


class HFEmbeddingFunction:
    EMBEDDING_DIMENSION = 384

    def __init__(self, model_name: str, token: str):
        if not token:
            raise BackendError(
                "HF_TOKEN is required for document embeddings. Add it to .env.",
                503,
                "embedding_not_configured",
            )
        self.model_name = model_name
        try:
            from huggingface_hub import InferenceClient
        except ImportError as error:
            raise RuntimeError("huggingface-hub is required for document embeddings.") from error
        self.client = InferenceClient(token=token)

    @staticmethod
    def _normalize(value: list) -> list[float]:
        if not value or not all(isinstance(item, Real) for item in value):
            raise RuntimeError("Hugging Face returned an invalid embedding shape.")
        vector = [float(item) for item in value]
        if not all(math.isfinite(item) for item in vector):
            raise RuntimeError("Hugging Face returned non-finite embedding values.")
        norm = math.sqrt(sum(item * item for item in vector))
        return [item / norm for item in vector] if norm else vector

    @classmethod
    def _convert_output(cls, result, count: int) -> list[list[float]]:
        data = result.tolist() if hasattr(result, "tolist") else result
        if not isinstance(data, list) or not data:
            raise RuntimeError("Hugging Face returned an invalid embedding response.")
        if count == 1 and all(isinstance(item, Real) for item in data):
            vectors = [data]
        elif count == 1 and len(data) == 1 and isinstance(data[0], list) and all(
            isinstance(value, Real) for value in data[0]
        ):
            vectors = data
        elif count > 1 and len(data) == count and all(
            isinstance(item, list) and all(isinstance(value, Real) for value in item)
            for item in data
        ):
            vectors = data
        else:
            raise RuntimeError(
                "Hugging Face returned token-level or otherwise invalid embeddings; "
                "expected one sentence embedding per input."
            )
        vectors = [cls._normalize(vector) for vector in vectors]
        if any(len(vector) != cls.EMBEDDING_DIMENSION for vector in vectors):
            raise RuntimeError(
                f"Hugging Face returned an invalid embedding dimension; "
                f"expected {cls.EMBEDDING_DIMENSION}."
            )
        return vectors

    def __call__(self, values: str | list[str]) -> list[list[float]]:
        texts = [values] if isinstance(values, str) else values
        if not texts:
            return []
        try:
            result = self.client.feature_extraction(texts, model=self.model_name)
            return self._convert_output(result, len(texts))
        except BackendError:
            raise
        except RuntimeError as error:
            raise BackendError(str(error), 502, "embedding_error") from error
        except Exception as error:
            raise BackendError("Hugging Face embedding generation failed.", 502, "embedding_error") from error


class ThreadVectorStore:
    def __init__(
        self,
        root: Path,
        thread_id: str,
        embedding_model: str,
        client=None,
        embedding_function=None,
        token: str | None = None,
    ):
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError("chromadb is required for document search.") from error
        root.mkdir(parents=True, exist_ok=True)
        self.client = client or chromadb.PersistentClient(path=str(root))
        self.embedding_function = embedding_function or HFEmbeddingFunction(
            embedding_model,
            token or os.getenv("HF_TOKEN", ""),
        )
        self.collection = self.client.get_or_create_collection(name=f"thread_{thread_id.replace('-', '')}")

    def replace(self, chunks: list[dict]) -> None:
        existing = self.collection.get()
        if existing.get("ids"):
            self.collection.delete(ids=existing["ids"])
        if not chunks:
            return
        self.collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[{"page": chunk["page"]} for chunk in chunks],
            embeddings=self.embedding_function([chunk["text"] for chunk in chunks]),
        )

    def search(self, query: str, top_k: int) -> list[dict]:
        count = self.collection.count()
        if not count:
            return []
        result = self.collection.query(
            query_embeddings=self.embedding_function([query]),
            n_results=min(max(1, top_k), count),
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [{"text": text, "page": int(metadata["page"])} for text, metadata in zip(documents, metadatas)]

    def delete(self) -> None:
        self.client.delete_collection(name=self.collection.name)
