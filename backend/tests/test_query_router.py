import pytest

from backend.query_router import route_query


def _structure(count=30, noun=None, fields=True):
    sections = []
    for index in range(count):
        number = index + 1
        sections.append({
            "index": index, "title": f"{number:03}. Title {number}", "name": ["Computer Vision", "Speech Recognition", "Distributed Systems"][index] if index < 3 else f"Subject {number}",
            "level": 1, "page_start": number * 2 - 1, "page_end": number * 2, "offset": 0, "noun": noun,
            "number": number, "number_path": str(number), "display_number": f"{number:03}", "label": "",
            "fields": {"Domain": "Health Science" if number > 20 else "Technology"} if fields else {},
            "matter": False, "parent": None, "ordinal": number,
        })
    return {"sections": sections, "units": list(range(count)), "unit_noun": noun, "fields": ["Domain"] if fields else [], "page_count": count * 2}


@pytest.mark.parametrize(
    "question, intent, target",
    [
        ("How many chapters are there in this document?", "count", "units"),
        ("how many total headings/topics the document contains?", "count", "units"),
        ("total headings?", "count", "headings"),
        ("How many pages does it have?", "count", "pages"),
        ("List all 30 topics in numerical order.", "list", "units"),
        ("list all the headlines?", "list", "headings"),
        ("What are the chapters?", "list", "units"),
        ("Show me the table of contents", "list", "outline"),
        ("What does the document contain?", "overview", "units"),
        ("what is this pdf about?", "overview", "units"),
        ("Give me an overview of the whole book", "overview", "units"),
        ("What does the document say about quantum teleportation?", "qa", "units"),
        ("Who was Lina?", "qa", "units"),
        ("Which stories mention a lantern?", "locate", "units"),
    ],
)
def test_structural_and_general_intents(question, intent, target):
    route = route_query(question, _structure())
    assert (route.intent, route.target) == (intent, target)


def test_chapter_range_is_explained_unit_by_unit():
    route = route_query("explain chapter 10-20", _structure())
    assert route.intent == "explain"
    assert route.units == list(range(9, 20))
    assert route.number_range == (10, 20)


@pytest.mark.parametrize("question", ["Summarize topics 5 to 7", "explain chapters 5 through 7", "Describe chapter 5–7"])
def test_range_separators(question):
    assert route_query(question, _structure()).units == [4, 5, 6]


def test_list_range_stays_structural_and_reports_missing_numbers():
    route = route_query("List topics 25-35", _structure())
    assert route.intent == "list"
    assert route.number_range == (25, 35)
    assert route.missing_numbers == [31, 32, 33, 34, 35]


@pytest.mark.parametrize(
    "question, intent, units",
    [
        ("What is the title of topic 2?", "lookup", [1]),
        ("name of the 20th story?", "lookup", [19]),
        ("which is the story 12?", "lookup", [11]),
        ("Which topic is 7? Summarize it.", "explain", [6]),
        ("chapter 12?", "explain", [11]),
        ("What happens in the last chapter?", "explain", [29]),
        ("Compare topic 3 with topic 7", "compare", [2, 6]),
        ("What does chapter 4 say about latency?", "scoped_qa", [3]),
        ("What are the central mechanisms of Computer Vision?", "scoped_qa", [0]),
        ("Compare Computer Vision and Speech Recognition", "compare", [0, 1]),
        ("Give a detailed summary of Distributed Systems.", "explain", [2]),
    ],
)
def test_unit_references(question, intent, units):
    route = route_query(question, _structure())
    assert (route.intent, route.units) == (intent, units)


def test_unknown_unit_number_is_reported_missing():
    route = route_query("Explain topic 999", _structure())
    assert route.intent == "explain" and route.units == [] and route.missing_numbers == [999]


def test_field_filters_for_count_and_list():
    count = route_query("How many chapters/topics have domain as Health sciences written ?", _structure())
    listing = route_query("There are 10 chapters that has Domain: Health Science. List all of them", _structure())
    assert count.intent == "count" and count.field_filter == ("Domain", "Health Science")
    assert listing.intent == "list" and listing.field_filter == ("Domain", "Health Science")


def test_page_questions_and_unit_on_page():
    assert route_query("what is on page 7?", _structure()).pages == [7]
    route = route_query("summarize the story on page 7", _structure())
    assert route.intent == "explain" and route.units == [3]


def test_follow_up_inherits_previous_unit():
    route = route_query("what is its main difficulty?", _structure(), previous_question="Explain topic 5")
    assert route.intent == "scoped_qa" and route.units == [4]
    assert "topic 5" in route.search_query
