from __future__ import annotations

from .errors import BackendError
from .models import Source, Thread
from .vector_store import ThreadVectorStore

NOT_FOUND_ANSWER = "I couldn't find that in the uploaded PDF."

SYSTEM_PROMPT = """You answer questions using only the supplied PDF excerpts.
If the excerpts do not contain the answer, say exactly: "I couldn't find that in the uploaded PDF."
Do not use outside knowledge or invent details. Cite supporting pages inline like [Page 3].
Keep answers concise and distinguish uncertainty clearly."""


class ChatService:
    def __init__(self, settings, vector_store_factory=ThreadVectorStore, client=None):
        self.settings = settings
        self.vector_store_factory = vector_store_factory
        self.client = client
        self._client_initialized = client is not None

    def store(self, thread_id: str):
        return self.vector_store_factory(
            self.settings.chroma_dir,
            thread_id,
            self.settings.embedding_model,
            token=self.settings.hf_token,
        )

    def _hf_client(self):
        if self._client_initialized:
            return self.client
        if not self.settings.hf_token:
            raise BackendError("HF_TOKEN is required for chat. Add it to .env.", 503, "llm_not_configured")
        try:
            from huggingface_hub import InferenceClient
            kwargs = {"token": self.settings.hf_token}
            if self.settings.hf_provider:
                kwargs["provider"] = self.settings.hf_provider
            self.client = InferenceClient(**kwargs)
        except ImportError as error:
            raise BackendError("huggingface-hub is required for chat.", 503, "llm_unavailable") from error
        self._client_initialized = True
        return self.client

    def ask(self, thread: Thread, question: str) -> tuple[str, list[Source]]:
        excerpts = self.store(thread.id).search(question, self.settings.top_k)
        sources = [Source(page=item["page"], text=item["text"], pdf_name=thread.pdf_name or "Uploaded PDF") for item in excerpts]
        context = "\n\n".join(f"[Page {item['page']}]\n{item['text']}" for item in excerpts)
        history = [{"role": message.role, "content": message.content} for message in thread.messages[-8:]]
        if history and history[-1]["role"] == "user" and history[-1]["content"] == question:
            history = history[:-1]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {
            "role": "user", "content": f"PDF excerpts:\n{context}\n\nQuestion: {question}"
        }]
        try:
            response = self._hf_client().chat_completion(
                model=self.settings.chat_model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=700,
            )
            answer = (response.choices[0].message.content or NOT_FOUND_ANSWER).strip()
            if answer == NOT_FOUND_ANSWER:
                return answer, []
            return answer, sources
        except BackendError:
            raise
        except Exception as error:
            raise BackendError(f"Chat generation failed: {error}", 502, "llm_error") from error
