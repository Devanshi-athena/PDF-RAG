from __future__ import annotations

import threading
from pathlib import Path


class LocalEmbeddingFunction:
    _models: dict[str, object] = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    def model(self):
        with self._lock:
            if self.model_name not in self._models:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as error:
                    raise RuntimeError("sentence-transformers is required for local embeddings.") from error
                self._models[self.model_name] = SentenceTransformer(self.model_name)
            return self._models[self.model_name]

    def __call__(self, values: list[str]) -> list[list[float]]:
        return self.model.encode(values, normalize_embeddings=True).tolist()


class ThreadVectorStore:
    def __init__(self, root: Path, thread_id: str, embedding_model: str, client=None, embedding_function=None):
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError("chromadb is required for document search.") from error
        root.mkdir(parents=True, exist_ok=True)
        self.client = client or chromadb.PersistentClient(path=str(root))
        self.embedding_function = embedding_function or LocalEmbeddingFunction(embedding_model)
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
