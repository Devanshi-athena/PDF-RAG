from pathlib import Path

from backend.chat_service import ChatService
from backend.config import BackendSettings


def test_store_is_reused_and_can_be_discarded():
    created = []

    def factory(*args, **kwargs):
        value = object()
        created.append(value)
        return value

    settings = BackendSettings(
        chroma_dir=Path("chroma"),
        embedding_model="embedding-model",
        hf_token="token",
    )
    service = ChatService(settings, vector_store_factory=factory)

    first = service.store("thread-1")
    assert service.store("thread-1") is first
    assert len(created) == 1

    service.discard_store("thread-1")
    assert service.store("thread-1") is not first
    assert len(created) == 2
