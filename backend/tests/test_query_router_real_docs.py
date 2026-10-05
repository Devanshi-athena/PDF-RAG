"""Routing for policy documents, reports and papers: references, structural questions and open-question modes."""
import pytest

from backend.query_router import route_query


def _section(index, title, level=1, parent=None, noun=None, number=None, path=None, roles=(), ident=None, name=None, children=()):
    return {
        "index": index, "title": title, "name": name or title, "level": level, "page_start": index + 1,
        "page_end": index + 2, "offset": 0, "noun": noun, "number": number, "number_path": path,
        "display_number": path or (str(number) if number else None), "ident": ident, "label": "", "fields": {},
        "matter": False, "roles": list(roles), "parent": parent, "children": list(children),
    }


@pytest.fixture
def report():
    sections = [
        _section(0, "Executive summary", roles=["summary"]),
        _section(1, "1 Introduction", number=1, path="1", name="Introduction", roles=["introduction"], children=[2]),
        _section(2, "1.1 Purpose and scope", level=2, parent=1, number=1, path="1.1", name="Purpose and scope", roles=["objectives"]),
        _section(3, "2 Regulatory framework", number=2, path="2", name="Regulatory framework", children=[4, 5]),
        _section(4, "2.1 Central functions", level=2, parent=3, number=2, path="2.1", name="Central functions"),
        _section(5, "2.2 Role of regulators", level=2, parent=3, number=2, path="2.2", name="Role of regulators"),
        _section(6, "3 Recommendations", number=3, path="3", name="Recommendations", roles=["recommendations"], children=[7, 8]),
        _section(7, "Recommendation 1: Train supervisors", level=2, parent=6, noun="recommendation", number=1, name="Train supervisors"),
        _section(8, "Recommendation 2: Fund data systems", level=2, parent=6, noun="recommendation", number=2, name="Fund data systems"),
        _section(9, "AC-2 ACCOUNT MANAGEMENT", level=2, parent=3, ident="AC-2"),
        _section(10, "Annex A: Glossary", noun="annex", number=1, name="Glossary"),
        _section(11, "Annex B: Stakeholder engagement", noun="annex", number=2, name="Stakeholder engagement"),
    ]
    return {
        "sections": sections, "units": [0, 1, 3, 6, 10, 11], "unit_noun": None, "explicit_numbers": True,
        "fields": [], "page_count": 40,
        "captions": [{"kind": "table", "number": "1", "text": "Table 1: Budget by year", "page": 9, "section": 6},
                     {"kind": "figure", "number": "2", "text": "Figure 2: Timeline", "page": 12, "section": 3}],
    }


@pytest.mark.parametrize(
    "question, intent, target",
    [
        ("What is the structure of the report?", "list", "outline"),
        ("list the contents of this document", "list", "outline"),
        ("How many tables are there?", "count", "captions"),
        ("List all figures", "list", "captions"),
        ("How many annexes are there?", "count", "units"),
        ("How many recommendations are there?", "count", "units"),
        ("What are the sections in chapter 2?", "list", "children"),
        ("How many subsections does section 1 have?", "count", "children"),
        ("Which section discusses data protection?", "locate", "units"),
        ("Where are penalties mentioned?", "locate", "units"),
        ("On which page is the budget described?", "locate", "units"),
    ],
)
def test_structural_routes(report, question, intent, target):
    route = route_query(question, report)
    assert (route.intent, route.target) == (intent, target)


@pytest.mark.parametrize(
    "question, intent, units",
    [
        ("Summarize Annex B", "explain", [11]),
        ("Explain section 2.2", "explain", [5]),
        ("What does 2.1 say about funding?", "scoped_qa", [4]),
        ("What is AC-2?", "explain", [9]),
        ("What is recommendation 2?", "explain", [8]),
        ("What is the role of regulater?", "explain", [5]),  # typo-tolerant title match
        ("Compare section 2.1 and section 2.2", "compare", [4, 5]),
    ],
)
def test_references(report, question, intent, units):
    route = route_query(question, report)
    assert (route.intent, route.units) == (intent, units)


def test_caption_reference_is_not_read_as_a_section_number(report):
    route = route_query("What does Table 1 show?", report)
    assert route.intent == "caption" and route.units == [] and route.captions[0]["page"] == 9


def test_open_question_modes(report):
    multi = route_query("What are the eligibility criteria and how is funding allocated?", report)
    assert multi.mode == "multi" and len(multi.subqueries) == 2

    facets = route_query("What are the objectives, budget and timeline of the programme?", report)
    assert facets.mode == "multi"
    assert facets.subqueries[0] == "objectives of the programme"

    compare = route_query("What is the difference between a baseline and an overlay?", report)
    assert compare.mode == "compare" and compare.entities == ["baseline", "overlay"]

    assert route_query("Why did the government avoid new legislation?", report).mode == "analysis"
    assert route_query("sandboxes", report).mode == "ambiguous"
    assert route_query("What is the reporting deadline?", report).mode == "direct"


def test_role_sections_scope_findings_and_recommendation_questions(report):
    route = route_query("What do the recommendations say about funding?", report)
    assert route.intent == "qa" and 6 in route.role_sections
    # The PDF labels its recommendations, so asking for the key ones lists them.
    assert route_query("What are the key recommendations?", report).intent == "list"


def test_document_name_words_do_not_block_structural_questions(report):
    route = route_query("What are the main chapters of the National Digital Health Blueprint? List them in order.", report,
                        document_name="IND_India_National-Digital-Health-Blueprint.pdf")
    assert (route.intent, route.target) == ("list", "units")


def test_whole_document_summary_wins_over_section_titles(report):
    route = route_query("Summarize the entire report in 8-10 concise points covering the regulatory framework", report)
    assert route.intent == "overview" and route.detail == "brief"
    blueprint = route_query("Give a summary of the Blueprint", report, document_name="National-Digital-Health-Blueprint.pdf")
    assert blueprint.intent == "overview"


def test_comparison_sides_are_cleaned_and_lists_split(report):
    pair = route_query("Compare the roles of UHID and Health Locker in the Blueprint.", report)
    assert pair.entities == ["UHID", "Health Locker"] and pair.aspect == "roles"
    layers = route_query("Compare the Infrastructure, Data, Technology and Application layers of the architecture.", report)
    assert layers.entities == ["Infrastructure layer", "Data layer", "Technology layer", "Application layer"]


def test_detail_level(report):
    assert route_query("Explain the funding allocation in detail", report).detail == "detailed"
    assert route_query("Briefly, what is the reporting deadline?", report).detail == "brief"
    assert route_query("What is the reporting deadline?", report).detail == "normal"


def test_content_nouns_are_structural_only_when_the_pdf_labels_them(report):
    unlabelled = dict(report, sections=[s for s in report["sections"] if s["noun"] != "recommendation"])
    assert route_query("How many recommendations are there?", report).intent == "count"
    assert route_query("How many recommendations are there?", unlabelled).intent == "qa"
    assert route_query("What is recommendation 7?", unlabelled).intent == "qa"  # left to retrieval, not "not found"


def test_off_topic_and_unknown_references(report):
    assert route_query("Explain Annex F", report).missing_numbers == ["F"]
    assert route_query("Explain section 9.9", report).missing_numbers == ["9.9"]
