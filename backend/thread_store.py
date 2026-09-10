from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .errors import NotFoundError
from .models import Thread


class ThreadStore:
    """Small JSON repository; metadata is kept separate from document bytes."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read(self) -> list[Thread]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return [Thread.model_validate(item) if hasattr(Thread, "model_validate") else Thread.parse_obj(item) for item in value]
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(f"Could not read thread metadata: {error}") from error

    def _write(self, threads: list[Thread]) -> None:
        payload = [thread.model_dump() if hasattr(thread, "model_dump") else thread.dict() for thread in threads]
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def list(self) -> list[Thread]:
        with self._lock:
            return self._read()

    def get(self, thread_id: str) -> Thread:
        with self._lock:
            for thread in self._read():
                if thread.id == thread_id:
                    return thread
        raise NotFoundError()

    def create(self, name: str | None = None) -> Thread:
        now = datetime.now(timezone.utc).isoformat()
        thread = Thread(id=str(uuid.uuid4()), name=(name or "New thread").strip()[:80] or "New thread", created_at=now, updated_at=now)
        with self._lock:
            threads = self._read()
            threads.insert(0, thread)
            self._write(threads)
        return thread

    def save(self, updated: Thread) -> Thread:
        updated.updated_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            threads = self._read()
            for index, thread in enumerate(threads):
                if thread.id == updated.id:
                    threads[index] = updated
                    self._write(threads)
                    return updated
        raise NotFoundError()

    def delete(self, thread_id: str) -> Thread:
        with self._lock:
            threads = self._read()
            for index, thread in enumerate(threads):
                if thread.id == thread_id:
                    removed = threads.pop(index)
                    self._write(threads)
                    return removed
        raise NotFoundError()
