from __future__ import annotations

import time
from collections.abc import Iterator

from .errors import BackendError
from .models import Source, Thread
from .vector_store import ThreadVectorStore

NOT_FOUND_ANSWER = "I couldn't find that in the uploaded PDF."

SYSTEM_PROMPT = """You answer questions using only the supplied PDF excerpts.

Rules:
1. Use only information explicitly supported by the retrieved excerpts. Do not use outside knowledge or invent details.
2. If the retrieved excerpts do not directly support the answer, say exactly:
"I couldn't find that in the uploaded PDF."
3. For table questions, inspect all relevant retrieved rows and return the exact values from the table. Do not guess or select a value without checking the available rows.
4. For image/diagram questions, answer only if the retrieved excerpts contain the relevant image-derived text or description. Do not infer what an image contains from surrounding text.
5. For questions requiring multiple pieces of evidence, use all relevant retrieved excerpts before answering.
6. Cite every factual answer inline using the page number, e.g. [Page 3].
7. Do not cite a page merely because it is related to the topic. The cited page must actually support the claim.
8. If the evidence is incomplete or conflicting, clearly state the uncertainty rather than guessing.
9. Keep answers concise."""


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
        total_started = time.perf_counter()
        messages, sources = self._prompt(thread, question)
        try:
            llm_started = time.perf_counter()
            response = self._hf_client().chat_completion(
                model=self.settings.chat_model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=700,
            )
            print(f"[PERF] llm_generation_seconds={time.perf_counter() - llm_started:.4f}")
            answer = (response.choices[0].message.content or NOT_FOUND_ANSWER).strip()
            print(f"[PERF] total_chat_request_seconds={time.perf_counter() - total_started:.4f}")
            return answer, [] if answer == NOT_FOUND_ANSWER else sources
        except BackendError:
            print(f"[PERF] total_chat_request_seconds={time.perf_counter() - total_started:.4f}")
            raise
        except Exception as error:
            print(f"[PERF] total_chat_request_seconds={time.perf_counter() - total_started:.4f}")
            raise BackendError(f"Chat generation failed: {error}", 502, "llm_error") from error

    def stream(self, thread: Thread, question: str) -> tuple[Iterator[str], list[Source]]:
        total_started = time.perf_counter()
        messages, sources = self._prompt(thread, question)

        def generate() -> Iterator[str]:
            try:
                llm_started = time.perf_counter()
                response = self._hf_client().chat_completion(
                    model=self.settings.chat_model,
                    messages=messages,
                    temperature=self.settings.temperature,
                    max_tokens=700,
                    stream=True,
                )
                emitted = False
                for chunk in response:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        emitted = True
                        yield delta
                if not emitted:
                    yield NOT_FOUND_ANSWER
                print(f"[PERF] llm_generation_seconds={time.perf_counter() - llm_started:.4f}")
                print(f"[PERF] total_chat_request_seconds={time.perf_counter() - total_started:.4f}")
            except BackendError:
                print(f"[PERF] total_chat_request_seconds={time.perf_counter() - total_started:.4f}")
                raise
            except Exception as error:
                print(f"[PERF] total_chat_request_seconds={time.perf_counter() - total_started:.4f}")
                raise BackendError(f"Chat generation failed: {error}", 502, "llm_error") from error

        return generate(), sources

    def _prompt(self, thread: Thread, question: str) -> tuple[list[dict], list[Source]]:
        total_started = time.perf_counter()
        excerpts = self.store(thread.id).search(question, self.settings.top_k)
        sources = [Source(page=item["page"], text=item["text"], pdf_name=thread.pdf_name or "Uploaded PDF") for item in excerpts]
        context = "\n\n".join(f"[Page {item['page']}]\n{item['text']}" for item in excerpts)
        history = [{"role": message.role, "content": message.content} for message in thread.messages[-8:]]
        if history and history[-1]["role"] == "user" and history[-1]["content"] == question:
            history = history[:-1]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {
            "role": "user", "content": f"PDF excerpts:\n{context}\n\nQuestion: {question}"
        }]
        print(f"[PERF] retrieval_and_prompt_seconds={time.perf_counter() - total_started:.4f}")
        return messages, sources
