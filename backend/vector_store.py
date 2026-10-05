from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
from pathlib import Path
from uuid import uuid4

import numpy as np

from .document_structure import load_structure, save_structure
from .errors import BackendError

SCHEMA_VERSION = 3
_EMBEDDING_FUNCTIONS = OrderedDict()
_EMBEDDING_FUNCTIONS_LOCK = threading.Lock()
_QUERY_EMBEDDINGS = OrderedDict()
_QUERY_EMBEDDINGS_LOCK = threading.Lock()
_QUERY_EMBEDDING_CACHE_LIMIT = 512
_EMBEDDING_FUNCTION_CACHE_LIMIT = 8
_RRF_CONSTANT = 60
_TOKEN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
STOP_WORDS = frozenset("""
a about above after again all also an and any are as at be been being below between both but by can could
did do does doing during each explain explained few for from further give had has have having he her here
hers him his how i if in into is it its itself just me mention mentioned more most my no nor not of on once
only or other our out over own pdf document file please same she should so some such tell than that the their
them then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your describe description detail details detailed summary summarize
summarise say says said stated state states according list show shown provide discuss information text
""".split())


def stem(word: str) -> str:
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(token) for token in _TOKEN.findall(text.casefold())]


def query_terms(text: str) -> list[str]:
    return list(dict.fromkeys(
        stem(token) for token in _TOKEN.findall(text.casefold())
        if token not in STOP_WORDS and len(token) > 1
    ))


class HFEmbeddingFunction:
    def __init__(self, model_name: str, token: str, provider: str = ""):
        if not token:
            raise BackendError("HF_TOKEN is required for document embeddings. Add it to .env.", 503, "embedding_not_configured")
        self.model_name = model_name
        try:
            from huggingface_hub import InferenceClient
        except ImportError as error:
            raise RuntimeError("huggingface-hub is required for document embeddings.") from error
        self.client = InferenceClient(token=token, **({"provider": provider} if provider else {}))

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
        elif len(data) == count and all(
            isinstance(item, list) and all(isinstance(value, Real) for value in item) for item in data
        ):
            vectors = data
        else:
            raise RuntimeError(
                "Hugging Face returned token-level or otherwise invalid embeddings; "
                "expected one sentence embedding per input."
            )
        vectors = [cls._normalize(vector) for vector in vectors]
        if len({len(vector) for vector in vectors}) != 1:
            raise RuntimeError("Hugging Face returned embeddings with inconsistent dimensions.")
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


def embedding_text(chunk: dict) -> str:
    parts = [chunk.get("section_path") or "", chunk.get("table_header") or "", chunk["text"]]
    return "\n".join(part for part in parts if part)


class ThreadVectorStore:
    """Per-thread Chroma collection plus an in-memory hybrid (vector + BM25) index and outline."""

    def __init__(
        self,
        root: Path,
        thread_id: str,
        embedding_model: str,
        client=None,
        embedding_function=None,
        token: str | None = None,
        embedding_batch_size: int = 32,
        embedding_workers: int = 4,
        structure_dir: Path | None = None,
        embedding_backend: str = "hf",
        reranker_model: str = "",
        embedding_provider: str = "",
    ):
        root = Path(root)
        if client is None:
            try:
                import chromadb
            except ImportError as error:
                raise RuntimeError("chromadb is required for document search.") from error
            root.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(root))
        self.client = client
        self.embedding_model = embedding_model
        self._embedding_function = embedding_function
        self._token = token
        self.embedding_backend = embedding_backend
        self.reranker_model = reranker_model
        self.embedding_provider = embedding_provider
        self.embedding_batch_size = max(1, embedding_batch_size)
        self.embedding_workers = max(1, embedding_workers)
        self.collection = self.client.get_or_create_collection(name=f"thread_{thread_id.replace('-', '')}")
        structure_root = Path(structure_dir) if structure_dir else root.parent / "structure"
        self.structure_path = structure_root / f"{thread_id}.json"
        self.digest_path = structure_root / f"{thread_id}.digests.json"
        self._index = None
        self._index_lock = threading.Lock()
        self._structure = None
        self._digests = None
        self._digest_lock = threading.Lock()

    @property
    def embedding_function(self):
        if self._embedding_function is None:
            backend = "hf" if self.embedding_backend == "hybrid" else self.embedding_backend
            self._embedding_function = _shared_embedding_function(
                self.embedding_model, self._token or os.getenv("HF_TOKEN", ""), backend, self.embedding_provider
            )
        return self._embedding_function

    @property
    def query_embedding_function(self):
        """"hybrid": documents are embedded through the API (parallel batches), questions locally (no round-trip)."""
        if self.embedding_backend != "hybrid":
            return self.embedding_function
        if self._embedding_function is not None and not isinstance(self._embedding_function, HFEmbeddingFunction):
            return self._embedding_function  # an injected function (tests) serves both roles
        try:
            return _shared_embedding_function(self.embedding_model, "", "local")
        except Exception as error:
            print(f"[WARN] local query embeddings unavailable, using the API: {error}")
            return self.embedding_function

    # ---------------------------------------------------------------- indexing
    def replace(self, chunks: list[dict], structure: dict | None = None) -> None:
        started = time.perf_counter()
        batches = [chunks[i:i + self.embedding_batch_size] for i in range(0, len(chunks), self.embedding_batch_size)]
        embed = lambda batch: self.embedding_function([embedding_text(chunk) for chunk in batch])
        if len(batches) <= 1 or self.embedding_workers == 1:
            embedded = [embed(batch) for batch in batches]
        else:
            with ThreadPoolExecutor(max_workers=min(self.embedding_workers, len(batches))) as executor:
                embedded = list(executor.map(embed, batches))
        embeddings = [vector for batch in embedded for vector in batch]
        print(f"[PERF] embedding_generation_seconds={time.perf_counter() - started:.4f} embeddings_created={len(embeddings)}")

        existing_ids = self.collection.get(include=[]).get("ids") or []
        generation = uuid4().hex
        ids = [f"{chunk['id']}-{generation}" for chunk in chunks]
        metadatas = [_metadata(chunk, position) for position, chunk in enumerate(chunks)]
        write_batch = max(1, self.embedding_batch_size * 4)
        written = []
        try:
            for i in range(0, len(chunks), write_batch):
                self.collection.add(
                    ids=ids[i:i + write_batch],
                    documents=[chunk["text"] for chunk in chunks[i:i + write_batch]],
                    metadatas=metadatas[i:i + write_batch],
                    embeddings=embeddings[i:i + write_batch],
                )
                written.extend(ids[i:i + write_batch])
        except Exception:
            if written:
                self.collection.delete(ids=written)
            raise
        if existing_ids:
            self.collection.delete(ids=existing_ids)
        if structure is not None:
            save_structure(self.structure_path, structure)
        self._structure = structure
        with self._digest_lock:
            self._digests = {}
            self.digest_path.unlink(missing_ok=True)
        self._index = _build_index(ids, [chunk["text"] for chunk in chunks], metadatas, embeddings)
        print(f"[PERF] embedding_indexing_seconds={time.perf_counter() - started:.4f}")

    def delete(self) -> None:
        self.client.delete_collection(name=self.collection.name)
        self.structure_path.unlink(missing_ok=True)
        self.digest_path.unlink(missing_ok=True)
        self._index = None
        self._structure = None

    def needs_reindex(self) -> bool:
        index = self.index()
        return bool(index["entries"]) and (not index["current"] or self.structure() is None)

    def structure(self) -> dict | None:
        if self._structure is None:
            self._structure = load_structure(self.structure_path)
        return self._structure

    def index(self) -> dict:
        if self._index is None:
            with self._index_lock:
                if self._index is None:
                    stored = self.collection.get(include=["documents", "metadatas", "embeddings"])
                    self._index = _build_index(
                        stored.get("ids") or [],
                        stored.get("documents") or [],
                        stored.get("metadatas") or [],
                        stored.get("embeddings"),
                    )
        return self._index

    # ------------------------------------------------------------ lookups
    def entries(self) -> list[dict]:
        return self.index()["entries"]

    def positions_for_sections(self, sections) -> list[int]:
        index = self.index()
        return sorted({p for s in sections for p in index["by_section"].get(int(s), [])})

    def positions_for_units(self, units) -> list[int]:
        index = self.index()
        return sorted({p for u in units for p in index["by_unit"].get(int(u), [])})

    def positions_for_pages(self, pages) -> list[int]:
        index = self.index()
        return sorted({p for page in pages for p in index["by_page"].get(int(page), [])})

    # ------------------------------------------------------------- search
    def search(self, query: str, limit: int, positions: list[int] | None = None) -> list[dict]:
        """Hybrid search. Each hit has pos, similarity (cosine or None), coverage and score."""
        index = self.index()
        entries = index["entries"]
        if not entries:
            return []
        allowed = None if positions is None else set(positions)
        if allowed is not None and not allowed:
            return []
        candidates = max(limit * 4, 30)
        terms = query_terms(query)

        lexical = _bm25(index, terms, allowed)[:candidates]
        similarities = None
        if index["matrix"] is not None:
            try:
                started = time.perf_counter()
                vector = np.asarray(self._cached_query_embedding(query)[0], dtype=np.float32)
                if vector.shape[0] == index["matrix"].shape[1]:
                    similarities = index["matrix"] @ vector
                print(f"[PERF] query_embedding_seconds={time.perf_counter() - started:.4f}")
            except BackendError as error:
                print(f"[WARN] query embedding failed, using lexical search only: {error.message}")
        vector_ranked = []
        if similarities is not None:
            scores = similarities if allowed is None else np.where(
                np.isin(np.arange(len(entries)), list(allowed)), similarities, -np.inf
            )
            count = min(candidates, len(entries) if allowed is None else len(allowed))
            top = np.argpartition(-scores, count - 1)[:count] if count < len(entries) else np.arange(len(entries))
            vector_ranked = [int(p) for p in sorted(top, key=lambda p: -scores[p]) if np.isfinite(scores[p])]

        fused = Counter()
        for rank, pos in enumerate(vector_ranked):
            fused[pos] += 1 / (_RRF_CONSTANT + rank + 1)
        for rank, (pos, _) in enumerate(lexical):
            fused[pos] += 1 / (_RRF_CONSTANT + rank + 1)
        term_set = set(terms)
        hits = []
        for pos, score in fused.most_common(candidates):
            entry = entries[pos]
            coverage = len(term_set & entry["term_set"]) / len(term_set) if term_set else 0.0
            hits.append({
                "pos": pos,
                "score": score,
                "similarity": float(similarities[pos]) if similarities is not None else None,
                "coverage": coverage,
            })
        hits = self._rerank(query, hits, entries)
        return hits[:max(limit, 1) * 3]

    def search_many(self, queries: list[str], limit: int, positions: list[int] | None = None) -> list[list[dict]]:
        """Search several sub-queries; their embeddings are fetched in a single request."""
        queries = list(dict.fromkeys(query for query in queries if query.strip()))
        if self.index()["matrix"] is not None:
            try:
                self._prime_query_embeddings(queries)
            except BackendError as error:
                print(f"[WARN] batch query embedding failed: {error.message}")
        return [self.search(query, limit, positions) for query in queries]

    def _rerank(self, query: str, hits: list[dict], entries: list[dict]) -> list[dict]:
        if not self.reranker_model or len(hits) < 2:
            return hits
        reranker = _shared_reranker(self.reranker_model)
        if reranker is None:
            return hits
        started = time.perf_counter()
        head = hits[:24]
        texts = [embedding_text(entries[hit["pos"]])[:1500] for hit in head]
        try:
            scores = reranker(query, texts)
        except Exception as error:  # a broken optional reranker must never break answering
            print(f"[WARN] reranker failed, keeping fused order: {error}")
            return hits
        for hit, score in zip(head, scores):
            hit["rerank"] = float(score)
        head.sort(key=lambda hit: -hit["rerank"])
        print(f"[PERF] reranker_seconds={time.perf_counter() - started:.4f}")
        return head + hits[24:]

    def _query_key(self, query: str) -> tuple[str, str]:
        return (self.embedding_model, re.sub(r"\s+", " ", query).strip().casefold())

    def _prime_query_embeddings(self, queries: list[str]) -> None:
        with _QUERY_EMBEDDINGS_LOCK:
            missing = [query for query in queries if self._query_key(query) not in _QUERY_EMBEDDINGS]
        if len(missing) < 2:
            return
        started = time.perf_counter()
        vectors = self.query_embedding_function(missing)
        with _QUERY_EMBEDDINGS_LOCK:
            for query, vector in zip(missing, vectors):
                _QUERY_EMBEDDINGS[self._query_key(query)] = (tuple(vector),)
            while len(_QUERY_EMBEDDINGS) > _QUERY_EMBEDDING_CACHE_LIMIT:
                _QUERY_EMBEDDINGS.popitem(last=False)
        print(f"[PERF] batch_query_embedding_seconds={time.perf_counter() - started:.4f} queries={len(missing)}")

    def _cached_query_embedding(self, query: str) -> list[list[float]]:
        key = self._query_key(query)
        with _QUERY_EMBEDDINGS_LOCK:
            cached = _QUERY_EMBEDDINGS.get(key)
            if cached is not None:
                _QUERY_EMBEDDINGS.move_to_end(key)
                return [list(vector) for vector in cached]
        vectors = self.query_embedding_function([query])
        with _QUERY_EMBEDDINGS_LOCK:
            _QUERY_EMBEDDINGS[key] = tuple(tuple(vector) for vector in vectors)
            while len(_QUERY_EMBEDDINGS) > _QUERY_EMBEDDING_CACHE_LIMIT:
                _QUERY_EMBEDDINGS.popitem(last=False)
        return vectors

    # ----------------------------------------------------- generated digests
    def digest(self, key: str) -> str | None:
        with self._digest_lock:
            if self._digests is None:
                try:
                    self._digests = json.loads(self.digest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self._digests = {}
            return self._digests.get(key)

    def save_digest(self, key: str, value: str) -> None:
        self.digest(key)
        with self._digest_lock:
            self._digests[key] = value
            self.digest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.digest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._digests, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.digest_path)


def _metadata(chunk: dict, position: int) -> dict:
    return {
        "page": int(chunk["page"]),
        "chunk_index": position,
        "section": int(chunk.get("section", -1)),
        "unit": int(chunk.get("unit", -1)),
        "start": int(chunk.get("start", 0)),
        "end": int(chunk.get("end", len(chunk["text"]))),
        "section_path": str(chunk.get("section_path") or ""),
        "table_header": str(chunk.get("table_header") or ""),
        "schema": SCHEMA_VERSION,
    }


def _build_index(ids, documents, metadatas, embeddings) -> dict:
    records = list(zip(ids, documents, metadatas, range(len(ids))))
    records.sort(key=lambda r: (int(r[2].get("chunk_index", r[3])), int(r[2].get("page", 0))))
    entries, postings = [], {}
    by_section, by_unit, by_page = {}, {}, {}
    for pos, (item_id, text, metadata, _) in enumerate(records):
        tokens = [t for t in tokenize(f"{metadata.get('section_path', '')} {metadata.get('table_header', '')} {text}") if t not in STOP_WORDS]
        frequencies = Counter(tokens)
        entry = {
            "id": item_id,
            "pos": pos,
            "text": text,
            "page": int(metadata.get("page", 1)),
            "section": int(metadata.get("section", -1)),
            "unit": int(metadata.get("unit", -1)),
            "start": int(metadata.get("start", 0)),
            "end": int(metadata.get("end", len(text))),
            "section_path": str(metadata.get("section_path") or ""),
            "table_header": str(metadata.get("table_header") or ""),
            "length": max(1, len(tokens)),
            "term_set": set(frequencies),
        }
        entries.append(entry)
        for term, frequency in frequencies.items():
            postings.setdefault(term, []).append((pos, frequency))
        by_section.setdefault(entry["section"], []).append(pos)
        by_unit.setdefault(entry["unit"], []).append(pos)
        by_page.setdefault(entry["page"], []).append(pos)
    matrix = None
    if embeddings is not None and len(embeddings) == len(records) and len(records):
        ordered = [embeddings[r[3]] for r in records]
        try:
            matrix = np.asarray(ordered, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0, 1, norms)
            if matrix.ndim != 2:
                matrix = None
        except (TypeError, ValueError):
            matrix = None
    return {
        "entries": entries,
        "postings": postings,
        "by_section": by_section,
        "by_unit": by_unit,
        "by_page": by_page,
        "matrix": matrix,
        "average_length": sum(e["length"] for e in entries) / max(1, len(entries)),
        "current": all(m.get("schema") == SCHEMA_VERSION for m in metadatas) if metadatas else True,
    }


def _bm25(index: dict, terms: list[str], allowed: set[int] | None) -> list[tuple[int, float]]:
    entries = index["entries"]
    total = len(entries)
    scores = Counter()
    for term in terms:
        matches = index["postings"].get(term, [])
        if not matches:
            continue
        idf = math.log(1 + (total - len(matches) + 0.5) / (len(matches) + 0.5))
        for pos, frequency in matches:
            if allowed is not None and pos not in allowed:
                continue
            length_ratio = entries[pos]["length"] / index["average_length"]
            scores[pos] += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length_ratio))
    return scores.most_common()


class LocalEmbeddingFunction:
    """Runs the same sentence-transformers model on this machine: no network round-trip or API credits."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise BackendError(
                "EMBEDDING_BACKEND=local needs the sentence-transformers package.", 503, "embedding_not_configured"
            ) from error
        self.model = SentenceTransformer(model_name, device="cpu")
        self._lock = threading.Lock()

    def __call__(self, values: str | list[str]) -> list[list[float]]:
        texts = [values] if isinstance(values, str) else values
        if not texts:
            return []
        with self._lock:
            vectors = self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, vector)) for vector in vectors]


_RERANKERS: dict[str, object] = {}
_RERANKERS_LOCK = threading.Lock()


def _shared_reranker(model_name: str):
    """Optional local cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2); None when unavailable."""
    with _RERANKERS_LOCK:
        if model_name not in _RERANKERS:
            try:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(model_name, device="cpu", max_length=512)
                lock = threading.Lock()

                def score(query, texts, model=model, lock=lock):
                    with lock:
                        return model.predict([(query, text) for text in texts], show_progress_bar=False)

                _RERANKERS[model_name] = score
            except Exception as error:
                print(f"[WARN] reranker {model_name!r} unavailable: {error}")
                _RERANKERS[model_name] = None
        return _RERANKERS[model_name]


def _shared_embedding_function(model_name: str, token: str, backend: str = "hf", provider: str = ""):
    key = (model_name, token, backend, provider)
    with _EMBEDDING_FUNCTIONS_LOCK:
        function = _EMBEDDING_FUNCTIONS.get(key)
        if function is None:
            function = LocalEmbeddingFunction(model_name) if backend == "local" else HFEmbeddingFunction(model_name, token, provider)
            _EMBEDDING_FUNCTIONS[key] = function
            while len(_EMBEDDING_FUNCTIONS) > _EMBEDDING_FUNCTION_CACHE_LIMIT:
                _EMBEDDING_FUNCTIONS.popitem(last=False)
        else:
            _EMBEDDING_FUNCTIONS.move_to_end(key)
        return function
