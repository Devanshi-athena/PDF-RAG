from pathlib import Path

import pytest

from backend.chat_service import NOT_FOUND_ANSWER, ChatService
from backend.config import BackendSettings
from backend.ingestion import ingest_pdf
from backend.models import Message, Source, Thread
from backend.vector_store import ThreadVectorStore

from .helpers import HashEmbeddings, MemoryClient, RecordingLLM, build_pdf

TOPICS = [
    ("Distributed Systems", "Technology", "replication consensus partition tolerance failure recovery", "latency and partial failure"),
    ("Database Indexing", "Technology", "b-trees hash indexes composite keys", "write amplification when indexes multiply"),
    ("Computer Vision", "Technology", "convolution feature maps object detection", "labelled image scarcity"),
    ("Vaccination Logistics", "Health Science", "cold chain storage distribution scheduling", "last mile refrigeration"),
    ("Sleep Architecture", "Health Science", "rem cycles slow wave sleep circadian timing", "measuring sleep outside laboratories"),
]


def _topics_pdf() -> bytes:
    pages = []
    for number, (name, domain, mechanisms, difficulty) in enumerate(TOPICS, start=1):
        pages.append([
            (f"{number:03}. {name}", 19, True),
            (f"Domain: {domain}  |  Page {number} of {len(TOPICS)}", 8, False),
            (f"{name} is the study of coordinated practice in its field.", 11, False),
            (f"The central mechanisms in {name} include {mechanisms}.", 11, False),
        ])
        pages.append([
            (f"A major practical difficulty in {name} is {difficulty}.", 11, False),
            (f"Practitioners of {name} document assumptions before drawing conclusions.", 11, False),
        ])
    return build_pdf(pages)


@pytest.fixture
def service(tmp_path):
    settings = BackendSettings(
        chroma_dir=tmp_path / "chroma", structure_dir=tmp_path / "structure", documents_dir=tmp_path / "documents",
        hf_token="test", chunk_size=400, chunk_overlap=60, llm_workers=2,
    )
    embeddings = HashEmbeddings()
    llm = RecordingLLM()
    client = MemoryClient()

    def factory(root, thread_id, model, **kwargs):
        return ThreadVectorStore(root, thread_id, model, client=client, embedding_function=embeddings, **kwargs)

    chat = ChatService(settings, vector_store_factory=factory, client=llm)
    thread = Thread(id="thread-topics", name="Topics", pdf_name="topics.pdf")
    ingest_pdf(_topics_pdf(), chat.store(thread.id), settings.chunk_size, settings.chunk_overlap)
    return chat, thread, llm, embeddings


def ask(chat, thread, question):
    thread.messages.append(Message(role="user", content=question))
    answer, sources = chat.ask(thread, question)
    thread.messages.append(Message(role="assistant", content=answer, sources=sources))
    return answer, sources


def test_structural_questions_are_answered_from_outline_without_llm(service):
    chat, thread, llm, _ = service

    count, count_sources = ask(chat, thread, "How many chapters are there in this document?")
    assert count.startswith("The document contains 5 chapters. They are numbered 001 to 005.")
    assert len(count_sources) == 5
    listing, _ = ask(chat, thread, "List all topics")
    assert "5. Topic 005: Sleep Architecture — Domain: Health Science [Pages 9–10]" in listing
    domain, _ = ask(chat, thread, "How many topics belong to the Health Science domain?")
    assert domain.startswith("2 topics in the uploaded PDF have Domain: Health Science.")
    lookup, _ = ask(chat, thread, "What is the title of topic 3?")
    assert lookup == 'Topic 003 is "Computer Vision"; Domain: Technology [Pages 5–6].'
    missing, _ = ask(chat, thread, "Explain topic 99")
    assert missing.startswith("I couldn't find topic 99 in the uploaded PDF. It contains 5 top-level topics.")
    pages, _ = ask(chat, thread, "How many pages are in the document?")
    assert pages == "The document has 10 pages."
    assert llm.calls == []


def test_chapter_questions_receive_the_whole_chapter_not_just_the_first_match(service):
    chat, thread, llm, _ = service

    answer, sources = ask(chat, thread, "What are the central mechanisms of Computer Vision?")

    prompt = llm.prompt()
    assert "convolution feature maps object detection" in prompt
    assert "labelled image scarcity" in prompt  # the rest of the chapter, from the next page
    assert "[Page 5]" in prompt and "[Page 6]" in prompt
    assert answer == "Grounded answer [Page 5]."
    assert [source.page for source in sources] == [5]


def test_chapter_range_explains_each_unit_in_parallel_and_caches(service):
    chat, thread, llm, _ = service

    answer, _ = ask(chat, thread, "explain chapter 2-4")

    assert answer.startswith("3 chapters:")
    for heading in ["Chapter 002: Database Indexing", "Chapter 003: Computer Vision", "Chapter 004: Vaccination Logistics"]:
        assert heading in answer
    assert len(llm.calls) == 3
    assert all(not call.get("stream") for call in llm.calls)
    ask(chat, thread, "summarize topics 2 to 4")
    assert len(llm.calls) == 3  # served from the digest cache


def test_single_chapter_explanation_streams_with_heading(service):
    chat, thread, llm, _ = service

    tokens, _ = chat.stream(thread, "explain chapter 1")
    parts = list(tokens)

    assert parts[0] == "Chapter 001: Distributed Systems (Pages 1–2)\n\n"
    assert llm.calls[0]["stream"] is True
    assert "latency and partial failure" in llm.prompt()


def test_off_topic_question_abstains_without_llm(service):
    chat, thread, llm, _ = service

    answer, sources = ask(chat, thread, "Who built the Taj Mahal?")

    assert answer == NOT_FOUND_ANSWER and sources == []
    assert llm.calls == []


def test_general_question_uses_hybrid_retrieval_and_neighbouring_context(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "Which field deals with cold chain storage?")

    prompt = llm.prompt()
    assert "cold chain storage" in prompt
    assert "last mile refrigeration" in prompt
    assert prompt.index("--- From: 004. Vaccination Logistics") < prompt.index("--- From: 001. Distributed Systems")


def test_overview_uses_outline_and_is_cached(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "What does the document contain?")
    prompt = llm.prompt()
    assert "Structure: 5 top-level sections" in prompt
    assert "Domain values: Technology (3), Health Science (2)" in prompt
    assert "Sleep Architecture" in prompt
    ask(chat, thread, "what does this document contain?")
    assert len(llm.calls) == 1


def test_follow_up_is_scoped_to_previous_chapter(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "Which topic is 2? Summarize it.")
    ask(chat, thread, "what is its practical difficulty?")

    assert "write amplification" in llm.prompt()
    assert "Sleep Architecture" not in llm.prompt()


def test_sources_are_limited_to_cited_pages():
    sources = [Source(page=page, text="t") for page in (1, 2, 3, 4)]

    assert [s.page for s in ChatService.finalize_sources("A [Page 2] and B [Pages 3–4].", sources)] == [2, 3, 4]
    assert ChatService.finalize_sources("No citations.", sources) == sources
    assert ChatService.finalize_sources(NOT_FOUND_ANSWER, sources) == []


def test_legacy_index_is_rebuilt_from_saved_pdf(tmp_path):
    settings = BackendSettings(chroma_dir=tmp_path / "chroma", structure_dir=tmp_path / "structure", documents_dir=tmp_path / "docs", hf_token="t")
    settings.ensure_directories()
    client = MemoryClient()
    client.collection.add(ids=["old"], documents=["old chunk"], metadatas=[{"page": 1}], embeddings=[[1.0, 0.0]])
    (settings.documents_dir / "legacy.pdf").write_bytes(_topics_pdf())
    chat = ChatService(
        settings,
        vector_store_factory=lambda root, tid, model, **kw: ThreadVectorStore(root, tid, model, client=client, embedding_function=HashEmbeddings(), **kw),
        client=RecordingLLM(),
    )

    answer, _ = chat.ask(Thread(id="legacy", name="Legacy", pdf_name="legacy.pdf"), "How many topics are there?")

    assert answer.startswith("The document contains 5 topics.")
    assert "old" not in client.collection.records
    assert Path(settings.structure_dir / "legacy.json").exists()
