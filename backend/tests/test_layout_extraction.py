"""Real-world layout handling: columns, tables, captions, contents pages, numbering depth, running titles."""
from backend.document_structure import build_structure
from backend.pdf_service import chunk_document, extract_document

from .helpers import body, build_layout_pdf, build_pdf

FILLER = "word " * 8


def _column(x, start_y, label, lines=5):
    # Equal-length lines reach the column's right edge, so they read as one wrapped paragraph.
    return [(x, start_y + i * 13, f"{label}{i} {FILLER}".strip(), 10, False) for i in range(lines)]


def test_two_column_page_reads_left_column_first_and_joins_wrapped_lines():
    page = _column(50, 100, "left") + _column(320, 100, "right")
    parsed = extract_document(build_layout_pdf([page]))
    text = parsed.pages[0][1]

    assert text.index("left0") < text.index("left4") < text.index("right0")
    assert "\n" not in text[text.index("left0"):text.index("left4")]


def test_table_cells_on_a_shared_baseline_become_rows():
    rows = [("Name", "Dept", "Score", "Year"), ("Asha", "AI", "91", "2023"), ("Ravi", "Data", "84", "2024")]
    page = [(x, 120 + r * 16, cell, 10, False) for r, row in enumerate(rows) for x, cell in zip((60, 180, 300, 420), row)]
    parsed = extract_document(build_layout_pdf([page + _column(60, 300, "prose", 3)]))
    text = parsed.pages[0][1]

    assert "Name | Dept | Score | Year\nAsha | AI | 91 | 2023\nRavi | Data | 84 | 2024" in text
    assert parsed.tables and parsed.tables[0]["header"] == "Name | Dept | Score | Year"


def test_chunks_starting_inside_a_table_keep_its_header_row():
    rows = [("ID", "Student", "Attendance")] + [(f"S-{n:03}", f"Student number {n}", f"{80 + n}%") for n in range(30)]
    page = [(x, 60 + r * 12, cell, 9, False) for r, row in enumerate(rows) for x, cell in zip((60, 200, 380), row)]
    parsed = extract_document(build_layout_pdf([page]))
    chunks = chunk_document(parsed, chunk_size=300, overlap=40)

    continuation = [chunk for chunk in chunks[1:] if " | " in chunk["text"]]
    assert continuation and all(chunk.get("table_header") == "ID | Student | Attendance" for chunk in continuation)


def test_numbering_depth_sets_levels_when_fonts_are_equal():
    pages = [
        [("1 Introduction", 14, True), *body("Intro text"), ("1.1 Scope", 14, True), *body("Scope text")],
        [("2 Methods", 14, True), *body("Method text"), ("2.1 Data", 14, True), *body("Data text")],
    ]
    structure = build_structure(extract_document(build_pdf(pages)))
    levels = {s["title"]: s["level"] for s in structure["sections"]}

    assert levels == {"1 Introduction": 1, "1.1 Scope": 2, "2 Methods": 1, "2.1 Data": 2}
    assert [structure["sections"][i]["title"] for i in structure["units"]] == ["1 Introduction", "2 Methods"]


def test_contents_page_is_not_read_as_headings_and_supplies_chapter_numbers():
    contents = [("Contents", 16, True)] + [
        (f"{n}. {title} .......... {page}", 11, True) for n, title, page in [(1, "Introduction", 2), (2, "Findings", 3), (3, "Next steps", 3)]
    ] + [("Annex A ......... 4", 11, True), ("Glossary ......... 4", 11, True), ("Acronyms ......... 4", 11, True)]
    pages = [
        contents,
        [("Introduction", 16, True), *body("Opening")],
        [("Findings", 16, True), *body("Results text"), ("Next steps", 16, True), *body("Plans")],
    ]
    structure = build_structure(extract_document(build_pdf(pages)))
    titles = [s["title"] for s in structure["sections"]]

    assert not any("....." in title for title in titles)
    numbers = {s["title"]: s["number"] for s in structure["sections"]}
    assert numbers["Introduction"] == 1 and numbers["Findings"] == 2 and numbers["Next steps"] == 3


def test_captions_are_indexed_but_sentences_about_figures_are_not():
    pages = [[
        ("Results", 16, True),
        ("Table 1: Survey response rates by region", 10, False),
        *body("Response rates differ"),
        ("Figure 2 shows the trend over time and it continues", 10, False),
        ("Figure 3. Trend of enrolment", 10, False),
    ]]
    parsed = extract_document(build_pdf(pages))

    assert [(c["kind"], c["number"]) for c in parsed.captions] == [("table", "1"), ("figure", "3")]


def test_running_title_repeated_on_every_page_is_not_a_section():
    pages = [[("Annual Programme Report", 18, True), (f"Chapter {n}: Topic {n}", 14, True), *body(f"Text {n}")] for n in (1, 2, 3)]
    parsed = extract_document(build_pdf(pages))
    structure = build_structure(parsed)

    assert parsed.title == "Annual Programme Report"
    assert [s["title"] for s in structure["sections"]] == ["Chapter 1: Topic 1", "Chapter 2: Topic 2", "Chapter 3: Topic 3"]


def test_labelled_annexes_recommendations_and_parent_numbers():
    pages = [
        [("INTRODUCTION", 16, True), *body("Intro"), ("1.1 Purpose", 12, True), *body("Purpose"), ("1.2 Audience", 12, True), *body("Audience")],
        [("Recommendation 1: Train supervisors", 12, True), *body("Rec one")],
        [("Annex A: Glossary of terms", 16, True), *body("Terms"), ("Annex B: Survey questions", 16, True), *body("Questions")],
    ]
    structure = build_structure(extract_document(build_pdf(pages)))
    by_title = {s["title"]: s for s in structure["sections"]}

    assert by_title["INTRODUCTION"]["number"] == 1  # inferred from 1.1 / 1.2
    assert (by_title["Annex B: Survey questions"]["noun"], by_title["Annex B: Survey questions"]["number"]) == ("annex", 2)
    assert by_title["Recommendation 1: Train supervisors"]["noun"] == "recommendation"
    assert "objectives" in by_title["1.1 Purpose"]["roles"]
