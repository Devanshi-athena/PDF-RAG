"""End-to-end answers on a generated policy/report PDF: structure, locate, captions, multi-part, comparison, long sections."""
from dataclasses import replace

import pytest

from backend.chat_service import NOT_FOUND_ANSWER, ChatService, PlainTextFilter
from backend.config import BackendSettings
from backend.ingestion import ingest_pdf
from backend.models import Message, Thread
from backend.vector_store import ThreadVectorStore

from .helpers import HashEmbeddings, MemoryClient, RecordingLLM, build_layout_pdf, build_pdf


def _report_pdf() -> bytes:
    def lines(sentence, count=4):
        return [(f"{sentence} sentence {n}.", 10, False) for n in range(1, count + 1)]

    return build_pdf([
        [("Executive summary", 16, True), *lines("The programme expands rural clinics and trains nurses")],
        [("1 Introduction", 16, True), *lines("Rural clinics lack trained nurses"),
         ("1.1 Purpose and scope", 13, True), *lines("The purpose is to expand rural clinic coverage")],
        [("2 Eligibility and funding", 16, True), *lines("Districts apply for programme grants"),
         ("2.1 Eligibility criteria", 13, True), *lines("Eligible districts have fewer than ten clinics per million people"),
         ("2.2 Funding allocation", 13, True), *lines("Funding is allocated by population weighted formula")],
        [("Table 1: Grant budget by year", 10, False), *lines("Budget grows each year"),
         ("3 Monitoring", 16, True), *lines("Quarterly indicators track clinic openings and nurse retention")],
        [("Annex A: Glossary", 16, True), *lines("A clinic is a primary care facility")],
    ])


@pytest.fixture
def service(tmp_path):
    settings = BackendSettings(
        chroma_dir=tmp_path / "chroma", structure_dir=tmp_path / "structure", documents_dir=tmp_path / "documents",
        hf_token="test", chunk_size=300, chunk_overlap=40, llm_workers=2,
    )
    client = MemoryClient()
    embeddings = HashEmbeddings()
    llm = RecordingLLM()

    def factory(root, thread_id, model, **kwargs):
        return ThreadVectorStore(root, thread_id, model, client=client, embedding_function=embeddings, **kwargs)

    chat = ChatService(settings, vector_store_factory=factory, client=llm)
    thread = Thread(id="thread-report", name="Report", pdf_name="report.pdf")
    ingest_pdf(_report_pdf(), chat.store(thread.id), settings.chunk_size, settings.chunk_overlap)
    return chat, thread, llm, settings


def ask(chat, thread, question):
    thread.messages.append(Message(role="user", content=question))
    answer, sources = chat.ask(thread, question)
    thread.messages.append(Message(role="assistant", content=answer, sources=sources))
    return answer, sources


def test_outline_children_annexes_and_captions_need_no_llm(service):
    chat, thread, llm, _ = service

    outline, _ = ask(chat, thread, "What is the structure of the document?")
    assert "- 2 Eligibility and funding [Pages 3–4]\n  - 2.1 Eligibility criteria [Page 3]" in outline
    children, _ = ask(chat, thread, "What are the sections in chapter 2?")
    assert "has 2 sub-sections" in children and "2.2 Funding allocation" in children
    annexes, _ = ask(chat, thread, "How many annexes are there?")
    assert annexes.startswith("The document has 1 annex:")
    tables, _ = ask(chat, thread, "How many tables are there?")
    assert "Table 1: Grant budget by year [Page 4]" in tables
    assert llm.calls == []


def test_locate_lists_sections_with_quotes_without_llm(service):
    chat, thread, llm, _ = service

    answer, sources = ask(chat, thread, "Which section discusses the funding formula?")

    assert answer.startswith("The topic is discussed in these parts of the document")
    assert "2 Eligibility and funding > 2.2 Funding allocation" in answer
    assert '"Funding is allocated by population weighted formula' in answer
    assert sources and llm.calls == []


def test_multi_part_question_retrieves_evidence_for_each_part(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "Who is eligible and how is funding allocated?")

    prompt = llm.prompt()
    assert "fewer than ten clinics" in prompt and "population weighted formula" in prompt
    assert "The question has several parts" in prompt
    assert len(llm.calls) == 1


def test_concept_comparison_groups_evidence_per_side(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "Compare district grant applications with nurse retention indicators")

    prompt = llm.prompt()
    assert "=== Evidence about: district grant applications" in prompt
    assert "=== Evidence about: nurse retention indicators" in prompt
    assert prompt.index("Districts apply for programme grants") < prompt.index("=== Evidence about: nurse retention indicators")
    assert prompt.index("nurse retention sentence") > prompt.index("=== Evidence about: nurse retention indicators")


def test_section_comparison_when_both_sides_are_section_titles(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "Compare eligibility criteria and funding allocation")

    assert "=== 2.1 Eligibility criteria (Page 3)" in llm.prompt()
    assert "=== 2.2 Funding allocation" in llm.prompt()


def test_caption_question_uses_the_caption_page(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "What does Table 1 show?")

    assert "Budget grows each year" in llm.prompt()
    assert "Table 1: Grant budget by year (page 4)" in llm.prompt()


def test_scoped_question_includes_the_section_outline(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "What does chapter 2 say about districts?")

    assert "Outline of 2 Eligibility and funding" in llm.prompt()


def test_long_section_is_explained_from_subsection_summaries(service):
    chat, thread, llm, settings = service
    chat.settings = replace(settings, explain_context_chars=200, range_unit_context_chars=600)

    concise, _ = ask(chat, thread, "Explain chapter 2")

    # Normal length: the exact sub-section list from the outline plus one concise model call.
    assert len(llm.calls) == 1 and llm.calls[0]["stream"]
    assert concise.startswith("Chapter 2: Eligibility and funding (Pages 3–4)\n\nSub-sections:\n- 2.1 Eligibility criteria [Page 3]\n- 2.2 Funding allocation")
    assert "Length: only as long as the question needs" in llm.prompt()

    detailed, _ = ask(chat, thread, "Explain chapter 2 in detail")

    # Detailed + long: one summary per sub-section streamed in order, no extra combining call.
    assert len(llm.calls) == 3 and not any(call.get("stream") for call in llm.calls[1:])
    assert detailed.startswith("Chapter 2: Eligibility and funding (Pages 3–4)\n\nIt has 2 sub-sections:")
    assert detailed.index("2.1 Eligibility criteria (Page 3)") < detailed.index("2.2 Funding allocation")


def test_repetition_guard_stops_looping_output(service):
    chat, thread, llm, _ = service
    looping = "Why sandboxes?\nThey allow safe testing [Page 4].\nWhy sandboxes? (continued)\nThey allow safe testing [Page 4].\n"
    llm.answer = looping

    answer, _ = ask(chat, thread, "Why are districts allowed to apply for grants?")

    assert answer == "Why sandboxes?\nThey allow safe testing [Page 4]."


def test_brief_questions_get_a_small_output_budget(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "Briefly, how is funding allocated?")

    assert llm.calls[-1]["max_tokens"] < 500
    assert "Length: brief" in llm.prompt()


def test_off_topic_question_abstains(service):
    chat, thread, llm, _ = service

    answer, _ = ask(chat, thread, "Who won the football world cup?")

    assert answer == NOT_FOUND_ANSWER and llm.calls == []


def test_table_header_is_restored_for_excerpts_starting_mid_table(tmp_path):
    rows = [("ID", "Student", "Attendance")] + [(f"S-{n:03}", f"Learner {n}", f"{60 + n}%") for n in range(35)]
    data = build_layout_pdf([[(x, 50 + r * 12, cell, 9, False) for r, row in enumerate(rows) for x, cell in zip((60, 200, 380), row)]])
    settings = BackendSettings(chroma_dir=tmp_path / "c", structure_dir=tmp_path / "s", hf_token="t", chunk_size=250, chunk_overlap=30)
    llm = RecordingLLM()
    chat = ChatService(
        settings,
        vector_store_factory=lambda root, tid, model, **kw: ThreadVectorStore(root, tid, model, client=MemoryClient(), embedding_function=HashEmbeddings(), **kw),
        client=llm,
    )
    thread = Thread(id="table", name="t", pdf_name="t.pdf")
    store = chat.store(thread.id)
    ingest_pdf(data, store, settings.chunk_size, settings.chunk_overlap)

    chat.ask(thread, "What is the attendance of Learner 30?")

    excerpt = llm.prompt().split("[Page 1]\n", 1)[1]
    assert excerpt.startswith("ID | Student | Attendance\n")


def test_plain_text_filter_removes_markdown_across_token_boundaries():
    plain = PlainTextFilter()
    chunks = ["**Key", " points**\n", "* first", "\n* second\n", "## Next", "\n"]
    assert "".join(plain.feed(c) for c in chunks) + plain.flush() == "Key points\n- first\n- second\nNext\n"


def test_comparison_answers_are_requested_as_tables(service):
    chat, thread, llm, _ = service

    ask(chat, thread, "Compare district grant applications with nurse retention indicators")

    prompt = llm.prompt()
    assert "markdown comparison table" in prompt and '"Aspect"' in prompt
    assert "Use a table when the answer compares items" in llm.calls[-1]["messages"][0]["content"]


def test_tables_stream_through_unchanged(service):
    chat, thread, llm, _ = service
    table = (
        "| Aspect | Eligibility | Funding |\n| --- | --- | --- |\n"
        "| Rule | Fewer than ten clinics [Page 3] | Population weighted formula [Page 3] |\n"
        "| Review | Not stated in the PDF | Not stated in the PDF |"
    )
    llm.answer = table

    answer, sources = ask(chat, thread, "Compare district grant applications with nurse retention indicators")

    assert answer == table
    assert [source.page for source in sources] == [3]
