from backend.document_structure import annotate_chunks, build_structure, parse_number, unit_heading
from backend.pdf_service import chunk_document, chunk_pages, extract_document

from .helpers import body, build_pdf


def _book():
    pages = []
    for number, title in enumerate(["Beginnings", "The Long Road", "Homecoming"], start=1):
        pages.append([(f"Chapter {number}: {title}", 20, True), *body(f"Chapter {number} narrative about {title.lower()}")])
        pages.append(body(f"More of chapter {number} continues here", 8))
    return build_pdf(pages, header="My Sample Book", page_numbers=True)


def test_font_headings_become_chapters_and_running_headers_are_removed():
    parsed = extract_document(_book())
    structure = build_structure(parsed)

    assert parsed.outline_source == "fonts"
    assert [s["title"] for s in structure["sections"]] == [
        "Chapter 1: Beginnings", "Chapter 2: The Long Road", "Chapter 3: Homecoming",
    ]
    assert structure["unit_noun"] == "chapter"
    assert [structure["sections"][i]["number"] for i in structure["units"]] == [1, 2, 3]
    assert structure["sections"][0]["page_start"] == 1 and structure["sections"][0]["page_end"] == 2
    assert all("My Sample Book" not in text for _, text in parsed.pages)


def test_story_labels_under_titles_provide_noun_and_number():
    pages = [
        [(title, 20, True), (f"Story {n} of 2", 8, False), *body(f"{title} tells a tale")]
        for n, title in enumerate(["The Lantern at Platform Seven", "The Orchard of Glass"], start=1)
    ]
    structure = build_structure(extract_document(build_pdf(pages)))

    assert structure["unit_noun"] == "story"
    second = structure["sections"][structure["units"][1]]
    assert second["number"] == 2
    assert unit_heading(structure, second) == "Story 2: The Orchard of Glass"


def test_numbered_topics_keep_padded_numbers_and_common_fields():
    pages = [
        [(f"{n:03}. Topic Title {n}", 19, True), (f"Domain: {'Health Science' if n % 2 else 'Technology'}  |  Page {n} of 4", 8, False), *body("Topic text")]
        for n in range(1, 5)
    ]
    structure = build_structure(extract_document(build_pdf(pages)))
    first = structure["sections"][0]

    assert structure["fields"] == ["Domain"]
    assert first["fields"] == {"Domain": "Health Science"}
    assert first["number"] == 1 and first["display_number"] == "001"
    assert unit_heading(structure, first) == "001. Topic Title 1"


def test_label_only_heading_merges_with_following_title():
    pages = [[("Chapter 3", 16, True), ("The Return", 20, True), *body("Body")], [("Chapter 4", 16, True), ("Endings", 20, True), *body("Body")]]
    structure = build_structure(extract_document(build_pdf(pages)))

    assert [s["title"] for s in structure["sections"]] == ["Chapter 3: The Return", "Chapter 4: Endings"]
    assert [s["number"] for s in structure["sections"]] == [3, 4]


def test_plain_text_pdf_falls_back_to_chapter_patterns():
    pages = [[("Chapter One", 11, False), *body("Opening text")], [("Chapter Two", 11, False), *body("Second text")]]
    parsed = extract_document(build_pdf(pages))
    structure = build_structure(parsed)

    assert parsed.outline_source == "text"
    assert [structure["sections"][i]["number"] for i in structure["units"]] == [1, 2]


def test_chunks_stay_inside_sections_and_carry_unit_and_path():
    parsed = extract_document(_book())
    structure = build_structure(parsed)
    chunks = chunk_document(parsed, chunk_size=200, overlap=40)
    annotate_chunks(chunks, structure)

    assert {chunk["unit"] for chunk in chunks} == set(structure["units"])
    for chunk in chunks:
        assert len(chunk["text"]) <= 200
        section = structure["sections"][chunk["section"]]
        assert section["page_start"] <= chunk["page"] <= section["page_end"]
        assert chunk["section_path"] == section["title"]


def test_chunking_prefers_sentence_boundaries_and_rejects_bad_overlap():
    chunks = chunk_pages([(7, "First sentence. Second sentence. Third sentence. Fourth sentence.")], chunk_size=36, overlap=8)

    assert len(chunks) > 1 and all(chunk["page"] == 7 for chunk in chunks)
    assert chunks[0]["text"].endswith(".")
    try:
        chunk_pages([(1, "text")], chunk_size=10, overlap=10)
    except ValueError:
        pass
    else:
        raise AssertionError("overlap >= chunk size must be rejected")


def test_parse_number_handles_digits_words_ordinals_and_roman():
    assert [parse_number(v) for v in ["12", "twelve", "twenty-one", "third", "IV", "xii"]] == [12, 12, 21, 3, 4, 12]
