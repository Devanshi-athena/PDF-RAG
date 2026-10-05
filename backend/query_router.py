"""Rule-based query routing.

Structural questions (counts, lists, outline, captions, sub-sections, "which section...") are answered from the
outline. References to chapters/sections/annexes/articles/pages/tables scope the context. Open questions get a
mode (direct, multi-part, comparison, analysis, enumeration, ambiguous) that controls retrieval breadth and the prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import difflib

from .document_structure import CONTENT_NOUNS, NUMBER_WORD_PATTERN, is_content_noun, normalize_title, parse_number, singular_noun
from .vector_store import query_terms, stem

_CONTENT_NOUN_WORDS = "|".join(
    sorted({word for pair in CONTENT_NOUNS.items() for word in pair}, key=len, reverse=True)
)
_UNIT_NOUN = (
    r"(?:chapters?|sections?|sub-?sections?|topics?|stor(?:y|ies)|parts?|units?|lessons?|modules?|articles?|episodes?|"
    rf"cases?|annex(?:es|ures?)?|appendix|appendices|schedules?|clauses?|ch\.|sec\.|art\.|{_CONTENT_NOUN_WORDS})"
)
_HEADING_NOUN = r"(?:headings?|headlines?|titles?|sub-?headings?)"
_CAPTION_NOUN = r"(?:tables?|figures?|figs?\.?|boxes|box|exhibits?|charts?|graphs?|diagrams?|illustrations?)"
_NUM = rf"(?:\d{{1,4}}|[ivxlcdm]{{1,7}}\b|{NUMBER_WORD_PATTERN})"
_ORDINAL_WORDS = (
    r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|"
    r"fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|last|final)"
)
_SEPARATOR = r"\s*(?:-|–|—|to|through|thru|till|until)\s*"
_RANGE = re.compile(
    rf"\b(?:from\s+)?(?P<noun>{_UNIT_NOUN})\s+(?P<a>{_NUM}){_SEPARATOR}(?:(?:{_UNIT_NOUN})\s+)?(?P<b>{_NUM})\b",
    re.IGNORECASE,
)
_REF_LIST = re.compile(
    rf"\b(?P<noun>{_UNIT_NOUN})\s+(?:(?:number|no\.?|#)\s*)?(?P<nums>\d{{1,4}}(?:\s*(?:,|&|\band\b|\bor\b)\s*\d{{1,4}})*|{_NUM})\b(?![.\-]\d)(?!\s*(?:st|nd|rd|th)\b)",
    re.IGNORECASE,
)
_LETTER_REF = re.compile(r"\b(?P<noun>annex(?:es|ures?)?|appendix|appendices|schedules?)\s+(?P<letters>[a-z](?:\s*(?:,|&|and|or)\s*[a-z])*)\b(?!\.\d)", re.IGNORECASE)
_REVERSED_REF = re.compile(rf"\b(?:which|what)\s+(?P<noun>{_UNIT_NOUN})\s+is\s+(?:number\s+|no\.?\s*|#)?(?P<num>{_NUM})\b", re.IGNORECASE)
_ORDINAL_REF = re.compile(
    rf"\b(?:the\s+)?(?P<ord>\d{{1,4}}(?:st|nd|rd|th)|{_ORDINAL_WORDS})\s+(?P<noun>{_UNIT_NOUN})\b", re.IGNORECASE
)
_DOTTED = re.compile(r"(?<![\w.])((?:\d{1,3}|[A-Z])(?:\.\d{1,3})+)(?![\w.]|\.\d)", re.IGNORECASE)
_PAGES = re.compile(r"\b(?:pages?|pg\.?|pp?\.)\s*(\d{1,4})(?:\s*(?:-|–|—|to|through|and)\s*(\d{1,4}))?\b", re.IGNORECASE)
_CAPTION_REF = re.compile(
    r"\b(?P<kind>tables?|figures?|figs?\.?|box(?:es)?|exhibits?|charts?)\s+(?P<number>(?:[a-z][.\-])?\d{1,3}(?:[.\-]\d{1,3})*|[a-z])\b(?!\s*(?:of|%))",
    re.IGNORECASE,
)
_DOC = r"(?:document|pdf|file|book|report|paper|text|doc|material|manual|guide|content|upload|policy|guideline|guidelines|framework|programme|program|strategy|plan|publication|study)"
_OVERVIEW = [
    re.compile(rf"\bwhat\b.*\b(?:does|do|is|are)\b.*\b(?:this|the|uploaded|given|my)\s+{_DOC}s?\b.*\b(?:contains?|covers?|includes?|discuss(?:es)?|talks? about|consists? of|about|deals? with|explains?)\s*[?.!]*$", re.I),
    re.compile(rf"\bwhat(?:'s| is| are)\s+(?:inside|in)\s+(?:this|the|my)\s+{_DOC}", re.I),
    re.compile(rf"\b(?:overview|summary|summari[sz]e|gist|synopsis|outline|main (?:ideas?|topics?|themes?|points?|messages?)|key (?:topics?|themes?|points?|ideas?|messages?|takeaways?))\b.*\b(?:this|the|whole|entire|uploaded|full|complete)\s+{_DOC}s?\b", re.I),
    re.compile(rf"\bsummari[sz]e\s+(?:this|the|it all|everything|all of it)(?:\s+(?:whole|entire))?(?:\s+{_DOC})?\s*[?.!]*$", re.I),
    re.compile(r"^(?:please\s+)?(?:give me |provide |write )?(?:an?\s+)?(?:overview|summary|gist|synopsis|tl;?dr)\s*[?.!]*$", re.I),
    re.compile(r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:main|key|major|central|overall)\s+(?:topics?|themes?|ideas?|points?|subjects?|content|messages?|takeaways?)\b\s*(?:of\s+(?:this|the)\s+\w+)?\s*[?.!]*$", re.I),
    re.compile(rf"\btell me about\s+(?:this|the)\s+{_DOC}\s*[?.!]*$", re.I),
    re.compile(rf"\bwhat\s+(?:kind|type|sort)\s+of\s+{_DOC}", re.I),
    re.compile(rf"\bwhat\s+(?:is|'s)\s+(?:this|the)\s+{_DOC}\s+(?:all\s+)?about\b", re.I),
]
_LOCATE = [
    re.compile(rf"\b(?:which|what)\s+(?:{_UNIT_NOUN}|pages?|headings?)\b.*\b(?:discuss|cover|talk|mention|describ|explain|deal|address|defin|contain|refer|about|regarding|include|has|have|focus|list|set out|specif|relat|concern|apply|applies|involv)", re.I),
    re.compile(r"\bwhere\b.*\b(?:mentioned|discussed|described|defined|covered|explained|stated|found|located|written|talked about|addressed|referenced|specified|set out|listed|introduced)\b", re.I),
    re.compile(rf"\bin\s+(?:which|what)\s+(?:{_UNIT_NOUN}|pages?)\b", re.I),
    re.compile(r"\bon\s+(?:which|what)\s+pages?\b", re.I),
    re.compile(r"\bwhere\s+(?:can|do|could|would)\s+i\s+(?:find|read|look)\b", re.I),
]
_LOCATE_WORDS = {
    stem(word) for word in """which what where section sections chapter chapters part parts page pages annex appendix article
    clause heading discuss discusses discussed cover covers covered talk talks mention mentions mentioned describe describes
    described explain explains explained deal deals address addresses addressed define defines defined contain contains refer
    refers referenced about regarding include includes stated found located written listed specified introduced find read
    look can could would there document pdf""".split()
}
_COUNT = re.compile(r"\b(?:how many|number of|count|total)\b", re.I)
_LIST = re.compile(
    r"\b(?:list|enumerate|show|name|names|what are|which are|which|give me|tell me|table of contents|toc|outline|"
    r"structure|index of|all|every|each)\b",
    re.I,
)
_OUTLINE = re.compile(r"\b(?:table of contents|contents|toc|outline|structure|structured|organi[sz]ed|organi[sz]ation|layout)\b", re.I)
_EXPLAIN = re.compile(
    r"\b(?:explain|explanation|summari[sz]e|summary|summaries|describe|overview|elaborate|discuss|walk me through|"
    r"tell me about|what happens|what is (?:it|this|that|[\w\s]{0,60}) about|what's (?:it|this|[\w\s]{0,60}) about|"
    r"go through|details? (?:of|about)|break ?down|recap|key points|main points|in detail|cover(?:s|ed)?|brief)\b",
    re.I,
)
_LOOKUP = re.compile(r"\b(?:name|title|heading|headline|called|titled|which|what is the name)\b", re.I)
_COMPARE = re.compile(r"\b(?:compare|comparison|comparing|contrast|difference|differences|differ|differs|versus|vs\.?|similarit(?:y|ies)|distinguish)\b", re.I)
_ANALYSIS = re.compile(
    r"\b(?:why|analy[sz]e|analysis|evaluate|evaluation of|assess|assessment of|implications?|impacts?|effects? of|"
    r"strengths?|weaknesses?|pros and cons|advantages?|disadvantages?|trade-?offs?|critically|critique|justify|"
    r"justification|rationale|reasons?|how effective|how well|to what extent|relationship between|relate to|"
    r"consequences?|risks? of|benefits? of|gaps?|interpret|significance|what can be concluded|infer)\b",
    re.I,
)
_ENUMERATE = re.compile(
    r"\b(?:list (?:all|every|the)|all (?:the )?|every|each of the|what are all|enumerate|name (?:all|the)|"
    r"what are the (?:\w+ )?(?:types|kinds|categories|steps|requirements|principles|criteria|components|elements|"
    r"functions|characteristics|stages|phases|objectives|pillars|priorities|measures|actions|options|features|"
    r"conditions|obligations|rights|duties|roles|areas|recommendations|findings))\b",
    re.I,
)
_QUESTION_WORD = re.compile(r"\b(?:what|how|why|when|where|which|who|whom|whose|is|are|does|do|can|could|should|will|explain|describe|list|summari[sz]e|compare)\b", re.I)
_FOLLOW_UP = re.compile(r"\b(?:it|its|this|that|these|those|they|them|their|he|she|his|her|him|same|above|previous|more|further|else|former|latter)\b", re.I)
_SUBQUESTION_SPLIT = re.compile(
    r"\?\s+|;\s+|\n+|\s+(?:and|also|plus|as well as)\s+(?=(?:what|how|why|when|where|which|who|whom|whose|is|are|does|do|can|could|should|list|describe|explain|summari[sz]e|give|tell)\b)",
    re.I,
)
_FACET_SPLIT = re.compile(r"\s*,\s*(?:and\s+|as well as\s+)?|\s+(?:and|as well as|along with|plus)\s+", re.I)
_LEADING_ASK = re.compile(
    r"^(?:(?:please\s+)?(?:tell me|explain|describe|list|summari[sz]e|give me|provide|outline|show me)\s+(?:about\s+)?|"
    r"what\s+(?:are|is|were|was)\s+|how\s+(?:are|is|do|does)\s+|what\s+does\s+the\s+\w+\s+say\s+about\s+)(?:the\s+)?",
    re.I,
)
_SHARED_TAIL = re.compile(r"\s+((?:of|for|in|under|within|across|regarding|on)\s+(?:the|this|its)?\s*[\w\s-]{3,60})$", re.I)
_COMPARE_PATTERNS = [
    re.compile(r"\bcompar(?:e|ing|ison of)\s+(?:the\s+)?(?P<a>.+?)\s+(?:and|with|to|against|vs\.?|versus)\s+(?:the\s+)?(?P<b>.+?)(?:\s+(?:in terms of|regarding|with respect to|based on|on|for|across|in relation to)\s+(?P<aspect>.+?))?\s*[?.!]*$", re.I),
    re.compile(r"\bdifferences?\s+between\s+(?:the\s+)?(?P<a>.+?)\s+and\s+(?:the\s+)?(?P<b>.+?)(?:\s+(?:in terms of|regarding|with respect to|based on|on|for|in relation to)\s+(?P<aspect>.+?))?\s*[?.!]*$", re.I),
    re.compile(r"\bhow\s+(?:does|do|is|are)\s+(?:the\s+)?(?P<a>.+?)\s+(?:differ|compare|contrast|relate)\w*\s*(?:from|with|to|against)?\s+(?:the\s+)?(?P<b>.+?)\s*[?.!]*$", re.I),
    re.compile(r"^(?:the\s+)?(?P<a>[^?]+?)\s+(?:vs\.?|versus)\s+(?:the\s+)?(?P<b>[^?]+?)\s*[?.!]*$", re.I),
]
_PLURAL_NOUNS = (
    r"\b(?:chapters|sections|subsections|sub-sections|topics|stories|parts|units|lessons|modules|articles|episodes|cases|"
    r"annexes|annexures|appendices|schedules|clauses|headings|headlines|titles|subheadings|sub-headings)\b"
)
_STRUCTURE_WORDS = {
    stem(word) for word in """many much total count number numbers numbered numerical numeric order ordered sequence
    sequential complete full entire whole name names title titles heading headings headline headlines subheading
    subheadings subsection subsections there present available contain contains include includes belong belongs altogether
    overall exist exists find found give get all every each one list enumerate show which have has written labelled
    labeled called table tables contents toc outline structure structured organised organized organisation organization
    layout page pages chapter chapters section sections topic topics story stories part parts unit units lesson lessons
    module modules article articles episode episodes case cases annex annexes appendix appendices schedule schedules
    clause clauses figure figures fig figs box boxes exhibit exhibits chart charts graph graphs diagram diagrams
    illustration illustrations wise sub inside under within document documents paper report publication book file
    doc text manual guide pdf main major principal key important""".split()
}
_DETAILED = re.compile(
    r"\b(?:in detail|detailed|elaborate|elaborately|comprehensive|comprehensively|thorough|thoroughly|in depth|"
    r"in-depth|explain fully|step by step|all (?:the )?details|everything about)\b",
    re.I,
)
_BRIEF = re.compile(
    r"\b(?:brief|briefly|short|shortly|concise|concisely|in one (?:line|sentence|word)|in (?:a )?few (?:words|lines|sentences)|"
    r"tl;?dr|quick|quickly|gist|in \d+(?:\s*(?:-|–|to)\s*\d+)? (?:points|bullets|bullet points|lines|sentences|words))\b",
    re.I,
)
_ENTITY_LEAD = re.compile(
    r"^(?:the\s+)?(?:roles?|functions?|purposes?|uses?|importance|significance|definitions?|meaning|features?|"
    r"responsibilities|objectives?|scope)\s+of\s+(?:the\s+)?",
    re.I,
)
_ENTITY_TAIL = re.compile(r"\s+(?:in|within|under|according to|as described in|as defined in|from)\s+(?:the|this)\s+[\w\s-]{2,60}$", re.I)
_ENTITY_LIST_NOUN = re.compile(r"^(?P<item>.+?)\s+(?P<noun>[A-Za-z]{3,}s)(?P<rest>\s+(?:of|in|for|under|within)\b.*)?$", re.I)
_SUMMARIZE_WHOLE = re.compile(
    r"\b(?:summari[sz]e|summary of|overview of|gist of|key points of|main points of|recap of)\s+(?:the\s+|this\s+)?"
    r"(?:entire|whole|full|complete|overall)\b",
    re.I,
)
_GENERIC_TITLES = {"summary", "overview", "contents", "index", "notes", "outline", "table of contents", "abstract", "introduction", "conclusion", "background"}
_ROLE_QUERIES = [
    (re.compile(r"\b(?:main|key|major|principal|overall)\s+(?:findings?|results?|outcomes?)\b|\bfindings\b|\bwhat (?:did|does) (?:the )?(?:study|paper|report|research|evaluation) find\b", re.I), ["results", "summary", "conclusion"]),
    (re.compile(r"\bconclu(?:sion|sions|de|ded|ding)\b", re.I), ["conclusion", "summary"]),
    (re.compile(r"\brecommend(?:s|ed|ation|ations)?\b", re.I), ["recommendations", "conclusion"]),
    (re.compile(r"\b(?:objectives?|aims?|goals?|purpose)\b", re.I), ["objectives", "introduction", "summary"]),
    (re.compile(r"\b(?:methodology|methods?|research design|how (?:was|were) (?:the )?(?:study|data|research|experiments?) (?:conducted|done|carried out|collected))\b", re.I), ["methods"]),
    (re.compile(r"\b(?:limitations?|weaknesses?|shortcomings?|constraints?)\b", re.I), ["limitations", "discussion"]),
    (re.compile(r"\b(?:budget|funding|financing|costs?|financial|resources? required)\b", re.I), ["budget"]),
    (re.compile(r"\b(?:definitions?|defined|meaning of|what is meant by|terminology|stands? for|acronyms?|abbreviations?)\b", re.I), ["definitions"]),
    (re.compile(r"\b(?:governance|roles? and responsibilit\w+|who is responsible|accountab\w+|oversight|institutional arrangements?)\b", re.I), ["governance"]),
    (re.compile(r"\b(?:monitoring|indicators?|kpis?|targets?|reporting requirements?|results framework|m\s*&\s*e)\b", re.I), ["monitoring"]),
    (re.compile(r"\b(?:implementation|timeline|timeframe|roadmap|work ?plan|phases?|milestones?|next steps)\b", re.I), ["implementation", "conclusion"]),
    (re.compile(r"\b(?:background|context|rationale|problem statement|why (?:was|is) (?:this|the) (?:policy|programme|program|framework|study))\b", re.I), ["introduction"]),
    (re.compile(r"\b(?:key points|key messages|takeaways|highlights|in short|tl;?dr|main points)\b", re.I), ["summary", "conclusion"]),
]


@dataclass
class Route:
    intent: str  # count | list | lookup | explain | compare | scoped_qa | page | caption | locate | overview | qa
    units: list[int] = field(default_factory=list)  # section indices
    explicit: bool = False  # sections came from numbers/labels/ids rather than a fuzzy title match
    pages: list[int] = field(default_factory=list)
    noun: str | None = None
    target: str = "units"  # units | headings | outline | pages | captions | children (for count/list)
    caption_kind: str | None = None
    captions: list[dict] = field(default_factory=list)
    field_filter: tuple[str, str] | None = None
    number_range: tuple[int, int] | None = None
    missing_numbers: list = field(default_factory=list)
    search_query: str = ""
    mode: str = "direct"  # direct | multi | compare | analysis | enumerate | ambiguous
    subqueries: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    role_sections: list[int] = field(default_factory=list)
    analysis: bool = False
    aspect: str = ""
    detail: str = "normal"  # brief | normal | detailed - how long the answer should be


def _norm(text: str) -> str:
    return normalize_title(text)


def _resolve(structure: dict, noun: str | None, number: int) -> list[int]:
    """Find the section a "<noun> <number>" reference means, preferring sections labelled with that noun."""
    sections = structure["sections"]
    if noun:
        labelled = [s["index"] for s in sections if s.get("noun") == noun and s.get("number") == number]
        if labelled:
            # A label can appear twice (summary list and full section): the longer section is the real one.
            return [max(labelled, key=lambda i: (sections[i].get("page_end", sections[i]["page_start"]) - sections[i]["page_start"], i))]
        if is_content_noun(noun):
            return []
    units = structure["units"]
    explicit = structure.get("explicit_numbers")
    if explicit or any(sections[i].get("number") is not None for i in units):
        matches = [i for i in units if sections[i].get("number") == number]
        if matches:
            return matches[:1]
    elif 0 < number <= len(units):
        return [units[number - 1]]
    if noun in (None, "section", "chapter", "part", "subsection"):
        paths = [s["index"] for s in sections if s.get("number_path") == str(number)]
        if paths:
            return paths[:1]
    return []


def _first_noun(question: str) -> str | None:
    match = re.search(rf"\b{_UNIT_NOUN}\b", question, re.I)
    return singular_noun(re.sub(r"^sub-?", "", match.group(0))) if match else None


def _numeric_refs(question: str, structure: dict, route: Route) -> list[int]:
    units = structure.get("units", [])
    sections = structure.get("sections", [])
    found: list[int] = []
    consumed = []
    range_match = _RANGE.search(question)
    if range_match:
        a, b = parse_number(range_match.group("a")), parse_number(range_match.group("b"))
        if a is not None and b is not None:
            start, end = sorted((a, b))
            noun = singular_noun(range_match.group("noun"))
            route.number_range = (start, end)
            labelled = [s["index"] for s in sections if noun and s.get("noun") == noun and s.get("number") is not None]
            pool = labelled or units
            if not labelled and not any(sections[i].get("number") is not None for i in units):
                numbered = {i: n for n, i in enumerate(units, start=1)}
            else:
                numbered = {i: sections[i].get("number") for i in pool}
            found.extend(i for i in pool if start <= (numbered.get(i) or -1) <= end)
            present = set(numbered.values())
            route.missing_numbers = [n for n in range(start, end + 1) if n not in present][:50]
            route.noun = route.noun or noun
            consumed.append(range_match.span())
    for match in _REF_LIST.finditer(question):
        if any(s <= match.start() < e for s, e in consumed):
            continue
        noun = singular_noun(re.sub(r"^sub-?", "", match.group("noun")))
        route.noun = route.noun or noun
        for raw in re.split(r"\s*(?:,|&|\band\b|\bor\b)\s*", match.group("nums")):
            number = parse_number(raw)
            if number is None:
                continue
            hits = _resolve(structure, noun, number)
            found.extend(hits)
            # "Recommendation 11" that isn't a heading may still be in the text: leave it to retrieval.
            if not hits and not is_content_noun(noun):
                route.missing_numbers.append(number)
    for match in _LETTER_REF.finditer(question):
        noun = singular_noun(match.group("noun"))
        route.noun = route.noun or noun
        for letter in re.findall(r"\b[a-z]\b", match.group("letters"), re.I):
            hits = [s["index"] for s in sections if s.get("noun") == noun and s.get("number") == ord(letter.lower()) - 96]
            hits = hits or [s["index"] for s in sections if s.get("number_path") == letter.upper()]
            found.extend(hits[:1])
            if not hits:
                route.missing_numbers.append(letter.upper())
    for match in _REVERSED_REF.finditer(question):
        noun = singular_noun(match.group("noun"))
        route.noun = route.noun or noun
        number = parse_number(match.group("num"))
        hits = _resolve(structure, noun, number) if number is not None else []
        found.extend(hits)
        if number is not None and not hits:
            route.missing_numbers.append(number)
    for match in _ORDINAL_REF.finditer(question):
        route.noun = route.noun or singular_noun(match.group("noun"))
        word = match.group("ord").casefold()
        if word in ("last", "final"):
            found.extend(units[-1:])
            continue
        position = int(re.sub(r"\D", "", word)) if word[0].isdigit() else parse_number(word)
        if position and 0 < position <= len(units):
            found.append(units[position - 1])
        elif position:
            route.missing_numbers.append(position)
    paths = {s.get("number_path"): s["index"] for s in sections if s.get("number_path") and "." in s["number_path"]}
    for match in _DOTTED.finditer(question):
        key = match.group(1).upper() if match.group(1)[0].isalpha() else match.group(1)
        if key in paths:
            found.append(paths[key])
        elif re.search(rf"\b(?:section|clause|sub-?section|paragraph|para)\s+{re.escape(match.group(1))}", question, re.I):
            route.missing_numbers.append(match.group(1))
    idents = {s["ident"].casefold(): s["index"] for s in sections if s.get("ident")}
    if idents:
        for token in re.findall(r"\b[A-Za-z]{1,6}-\d{1,4}(?:\(\d{1,3}\))?|\b[A-Za-z]{1,6}\s\d{1,2}\.\d{1,2}\b", question):
            if token.casefold() in idents:
                found.append(idents[token.casefold()])
    return list(dict.fromkeys(found))


def _title_refs(question: str, structure: dict) -> list[int]:
    normalized = f" {_norm(question)} "
    candidates = []
    for section in structure.get("sections", []):
        name = _norm(section["name"])
        name = re.sub(r"^(?:the|a|an) ", "", name)
        if len(name) < 4 or name in _GENERIC_TITLES or singular_noun(name) or section.get("matter"):
            continue
        match = re.search(rf" {re.escape(name)}s? ", normalized)
        if match:
            candidates.append((match.start(), match.end(), section["index"]))
    # Prefer the most specific section: drop a match whose descendant also matched ("THE CONTROLS" vs "INCIDENT RESPONSE").
    sections = structure.get("sections", [])
    matched = {index for _, _, index in candidates}

    def has_matched_descendant(index):
        return any(_is_ancestor(sections, index, other) for other in matched if other != index)

    candidates = [item for item in candidates if not has_matched_descendant(item[2])]
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
    taken, result = [], []
    for start, end, index in candidates:
        if any(start < e and s < end for s, e in taken):
            continue
        taken.append((start, end))
        result.append((start, index))
    return [index for _, index in sorted(result)]


def _residual_terms(text: str, structure: dict, units: list[int]) -> list[str]:
    """Question words that are not part of (a possibly misspelt) referenced section title."""
    names = " ".join(structure["sections"][u]["name"] for u in units)
    name_words = {stem(word) for word in query_terms(names)}
    residual = []
    for word in query_words(text):
        word = stem(word)
        if word in name_words or any(difflib.SequenceMatcher(None, word, name).ratio() >= 0.8 for name in name_words):
            continue
        residual.append(word)
    return residual


def _fuzzy_title_refs(question: str, structure: dict) -> list[int]:
    """Tolerate typos ("guidline use" -> "Guideline use") by comparing question n-grams with section names."""
    words = _norm(question).split()
    if not words:
        return []
    best: tuple[float, int] | None = None
    for section in structure.get("sections", []):
        if section.get("matter"):
            continue
        name = re.sub(r"^(?:the|a|an) ", "", _norm(section["name"]))
        size = len(name.split())
        if len(name) < 8 or not 2 <= size <= 6 or name in _GENERIC_TITLES:
            continue
        for start in range(0, max(1, len(words) - size + 1)):
            candidate = " ".join(words[start:start + size])
            if abs(len(candidate) - len(name)) > 3:
                continue
            ratio = difflib.SequenceMatcher(None, candidate, name).ratio()
            if ratio >= 0.88 and (best is None or ratio > best[0]):
                best = (ratio, section["index"])
    return [best[1]] if best else []


def _is_ancestor(sections: list[dict], ancestor: int, index: int) -> bool:
    parent = sections[index].get("parent")
    while parent is not None:
        if parent == ancestor:
            return True
        parent = sections[parent].get("parent")
    return False


def _role_refs(question: str, structure: dict) -> list[int]:
    sections = structure.get("sections", [])
    wanted: list[str] = []
    for pattern, roles in _ROLE_QUERIES:
        if pattern.search(question):
            wanted.extend(role for role in roles if role not in wanted)
    if not wanted:
        return []
    found = []
    for role in wanted:
        found.extend(s["index"] for s in sections if role in s.get("roles", []) and s["index"] not in found)
        if len(found) >= 6:
            break
    return found[:6]


def _field_filter(question: str, structure: dict) -> tuple[str, str] | None:
    fields = structure.get("fields") or []
    if not fields:
        return None
    question_stems = {stem(t) for t in _norm(question).split()}
    best = None
    for key in fields:
        key_present = all(stem(t) in question_stems for t in _norm(key).split())
        values = {structure["sections"][i]["fields"].get(key) for i in structure["units"]} - {None, ""}
        for value in values:
            value_stems = {stem(t) for t in _norm(value).split()}
            if value_stems and value_stems <= question_stems:
                score = (key_present, len(value_stems))
                if best is None or score > best[0]:
                    best = (score, key, value)
    return (best[1], best[2]) if best else None


def _caption_refs(question: str, structure: dict, route: Route) -> list[dict]:
    captions = structure.get("captions") or []
    found = []
    for match in _CAPTION_REF.finditer(question):
        kind = _caption_kind(match.group("kind"))
        number = match.group("number").casefold()
        hits = [c for c in captions if c["kind"] == kind and c["number"].casefold() == number]
        if hits:
            found.extend(hits[:1])
        else:
            route.missing_numbers.append(f"{kind} {match.group('number')}")
    return found


def _caption_kind(value: str) -> str:
    value = value.casefold().rstrip(".")
    singular = {
        "tables": "table", "figures": "figure", "figs": "fig", "boxes": "box", "exhibits": "exhibit",
        "charts": "chart", "graphs": "graph", "diagrams": "diagram", "illustrations": "illustration",
    }
    value = singular.get(value, value)
    return {"fig": "figure", "chart": "figure", "graph": "figure", "diagram": "figure", "illustration": "figure"}.get(value, value)


def _content_terms(text: str, route: Route | None = None, extra: set[str] | None = None) -> list[str]:
    ignored = set(_STRUCTURE_WORDS) | (extra or set())
    if route and route.field_filter:
        ignored |= {stem(t) for t in _norm(" ".join(route.field_filter)).split()}
    parts = [stem(part) for term in query_terms(text) for part in re.split(r"[-./]", term) if part]

    def division_noun(term):  # chapter(s), annexure(s), appendices ... in any form
        noun = singular_noun(term) or singular_noun(term + "s")
        return noun is not None and not is_content_noun(noun)

    return [
        t for t in parts
        if t not in ignored and not t.isdigit() and not re.fullmatch(r"\d+(?:st|nd|rd|th)", t) and not division_noun(t)
    ]


def query_words(text: str) -> list[str]:
    """Content words left after removing references such as "chapter 3", "Annex B", "3.2.4" or "AC-2"."""
    stripped = re.sub(rf"\b{_UNIT_NOUN}\s+(?:{_NUM}|[a-z]\b)(?:\.\d+)*", " ", text, flags=re.I)
    stripped = re.sub(r"\b[A-Za-z]{1,6}-\d{1,4}(?:\(\d{1,3}\))?|\b(?:\d{1,3}|[A-Z])(?:\.\d{1,3})+\b", " ", stripped)
    return [word for word in query_terms(stripped) if stem(word) not in _STRUCTURE_WORDS]


def _compare_entities(text: str) -> tuple[list[str], str]:
    """The things being compared: "roles of UHID" vs "Health Locker in the Blueprint" -> UHID, Health Locker (aspect:
    roles); "the Infrastructure, Data, Technology and Application layers" -> four layers."""
    for pattern in _COMPARE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        a, b = (re.sub(r"^(?:the|a|an)\s+", "", match.group(name).strip(" ,"), flags=re.I) for name in ("a", "b"))
        aspect = (match.groupdict().get("aspect") or "").strip()
        lead = _ENTITY_LEAD.match(a)
        if lead:
            aspect = aspect or re.sub(r"^(?:the\s+)?|\s+of(?:\s+the)?$", "", lead.group(0).strip(), flags=re.I)
            a = a[lead.end():]
        b = _ENTITY_TAIL.sub("", b)
        entities = [a, b]
        if "," in a:  # "X, Y, Z and W layers of the architecture"
            listed = _ENTITY_LIST_NOUN.match(b)
            noun, rest = (listed.group("noun"), (listed.group("rest") or "").strip()) if listed else ("", "")
            items = [item.strip() for item in re.split(r"\s*,\s*", a) if item.strip()] + [listed.group("item") if listed else b]
            entities = [f"{item} {noun[:-1] if noun.lower().endswith('s') else noun}".strip() for item in items][:5]
            aspect = " ".join(part for part in (aspect, rest) if part)
        entities = [entity for entity in entities if query_terms(entity) and len(entity) <= 120]
        if len(entities) >= 2:
            return entities, aspect
    return [], ""


def _subqueries(text: str) -> list[str]:
    """Split multi-part questions into independently retrievable parts (the full question is always searched too).
    "What responsibilities are assigned to the National, State and Facility levels?" ->
    "...assigned to the National levels", "...the State levels", "...the Facility levels"."""
    pieces = [p.strip(" ,.") for p in _SUBQUESTION_SPLIT.split(text) if p and query_terms(p)]
    if len(pieces) >= 2:
        return pieces[:4]
    core = _LEADING_ASK.sub("", text.strip().rstrip("?.! "))
    tail_match = _SHARED_TAIL.search(core)
    tail = tail_match.group(1) if tail_match else ""
    body = core[: tail_match.start()] if tail_match else core
    facets = [f.strip() for f in _FACET_SPLIT.split(body) if f.strip()]
    facets = [f for f in facets if query_terms(f)]
    if len(facets) < 2 or len(facets) > 5:
        return []
    # The words before the first item and after the last one belong to every item.
    prefix, suffix = "", ""
    first, last = facets[0].split(), facets[-1].split()
    if len(first) >= 3:
        prefix, facets[0] = " ".join(first[:-1]), first[-1]
    if len(last) >= 3:
        facets[-1], suffix = last[0], " ".join(last[1:])
    return [re.sub(r"\s+", " ", f"{prefix} {facet} {suffix} {tail}").strip() for facet in facets][:4]


def _document_words(structure: dict, document_name: str | None) -> tuple[set[str], set[str]]:
    """Words of the document's own name/title (not content words in a question), and its type word ("Blueprint")."""
    names = [structure.get("title") or "", re.sub(r"\.pdf$", "", document_name or "", flags=re.I)]
    words, type_words = set(), set()
    for name in names:
        terms = [stem(t) for t in query_terms(re.sub(r"[_\-]+", " ", name)) if not t.isdigit()]
        words.update(terms)
        if terms:
            type_words.add(terms[-1])
    return words, type_words


def _detail(text: str) -> str:
    if _DETAILED.search(text):
        return "detailed"
    if _BRIEF.search(text):
        return "brief"
    return "normal"


def route_query(
    question: str, structure: dict | None, previous_question: str | None = None, document_name: str | None = None
) -> Route:
    structure = structure or {"sections": [], "units": []}
    text = question.strip().strip("“”\"'")
    route = Route(intent="qa", search_query=question, detail=_detail(text))
    lowered = text.casefold()
    document_words, type_words = _document_words(structure, document_name)
    heading_noun = re.search(rf"\b{_HEADING_NOUN}\b", lowered)
    caption_noun = re.search(rf"\b{_CAPTION_NOUN}\b", lowered)
    pages_word = re.search(r"\bpages\b", lowered)

    route.captions = _caption_refs(text, structure, route)
    # "Box 1.2" / "Table 3" must not also be read as section 1.2 / section 3.
    units = _numeric_refs(_CAPTION_REF.sub(" ", text), structure, route)
    route.explicit = bool(units) or bool(route.missing_numbers)
    if not units and not route.missing_numbers:
        units = _title_refs(text, structure) or _fuzzy_title_refs(text, structure)
    route.noun = route.noun or _first_noun(text)
    labelled_nouns = {s.get("noun") for s in structure.get("sections", []) if s.get("noun")}
    if is_content_noun(route.noun) and route.noun not in labelled_nouns:
        route.noun = None  # "objectives", "recommendations" ... are content here, not headings to count or list
    for match in _PAGES.finditer(text):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        start, end = sorted((start, end))
        route.pages.extend(range(start, min(end, start + 9) + 1))
    route.field_filter = _field_filter(text, structure)
    labelled_words = {stem(w) for noun in labelled_nouns if is_content_noun(noun) for w in (noun, CONTENT_NOUNS[noun])}
    structural = route.field_filter is not None or not _content_terms(text, route, labelled_words | document_words)
    plural = re.search(_PLURAL_NOUNS, lowered) or (
        route.noun and is_content_noun(route.noun) and re.search(rf"\b{re.escape(CONTENT_NOUNS[route.noun])}\b", lowered)
    )
    sub_word = re.search(r"\bsub-?(?:sections?|headings?|topics?|parts?)\b", lowered)

    # ---- sub-sections of a referenced section: "what sections are in chapter 3", "how many subsections in 2.1"
    if units and route.explicit and (_COUNT.search(lowered) or _LIST.search(lowered)) and structural and not _EXPLAIN.search(lowered):
        ref_noun = route.noun
        plural_noun_found = singular_noun(re.sub(r"^sub-?", "", plural.group(0))) if plural else None
        if sub_word or (plural and (plural_noun_found != ref_noun or heading_noun)) or heading_noun:
            route.units = units
            route.intent = "count" if _COUNT.search(lowered) else "list"
            route.target = "children"
            return route

    # ---- counting / listing the document's structure
    if not route.explicit and structural and not route.captions:
        if _COUNT.search(lowered) and (route.noun or heading_noun or pages_word or caption_noun):
            route.intent = "count"
            if caption_noun and not route.noun:
                route.target, route.caption_kind = "captions", _caption_kind(caption_noun.group(0))
            elif pages_word and not (route.noun or heading_noun):
                route.target = "pages"
            elif (heading_noun or sub_word) and not (route.noun and not sub_word):
                route.target = "headings"
            return route
        if _LIST.search(lowered) and not _EXPLAIN.search(lowered) and not _COMPARE.search(lowered) and (route.number_range or not units):
            if _OUTLINE.search(lowered):
                route.intent, route.target = "list", "outline"
                return route
            if caption_noun and not route.noun and re.search(r"\b(?:tables|figures|figs|boxes|exhibits|charts|graphs|diagrams|illustrations)\b", lowered):
                route.intent, route.target, route.caption_kind = "list", "captions", _caption_kind(caption_noun.group(0))
                return route
            if plural:
                route.intent = "list"
                route.target = "headings" if heading_noun or sub_word else "units"
                return route
    if route.number_range and _LIST.search(lowered) and structural and not _EXPLAIN.search(lowered):
        route.intent = "list"
        return route

    # ---- "which section discusses X?" / "where is X mentioned?"
    if any(pattern.search(text) for pattern in _LOCATE) and _content_terms(text, extra=_LOCATE_WORDS):
        route.intent = "locate"
        route.search_query = " ".join(t for t in re.findall(r"[\w-]+", text) if stem(t.casefold()) not in _LOCATE_WORDS) or text
        return route

    # ---- follow-ups such as "summarize it" inherit the section referenced by the previous question
    if not units and not route.missing_numbers and not route.captions and previous_question and _FOLLOW_UP.search(lowered) and len(text.split()) <= 14:
        previous = route_query(previous_question, structure)
        if previous.units:
            units, route.explicit = previous.units, previous.explicit
            route.noun = route.noun or previous.noun
        elif previous.captions:
            route.captions = previous.captions
        route.search_query = f"{previous_question} {question}"

    if route.pages and not units and route.noun and not route.explicit:
        sections = structure["sections"]
        units = [i for i in structure["units"] if any(sections[i]["page_start"] <= p <= sections[i]["page_end"] for p in route.pages)]
        route.explicit = bool(units)

    route.analysis = bool(_ANALYSIS.search(lowered))
    entities, aspect = _compare_entities(text) if _COMPARE.search(lowered) else ([], "")

    if route.captions and not route.explicit:
        route.intent = "caption"  # "Figure 2.1" beats a fuzzy title match on words around it
        return route
    # "Summarize the entire Blueprint ..." is about the whole document, even if it names some of its topics.
    whole_document = _SUMMARIZE_WHOLE.search(text) or any(
        re.search(rf"\b(?:summari[sz]e|summary of|overview of|gist of)\s+(?:the\s+|this\s+)?{re.escape(word)}\w*\b", lowered)
        for word in type_words
    )
    if not route.explicit and not route.pages and (whole_document or any(p.search(text) for p in _OVERVIEW)):
        route.intent = "overview"
        if route.detail == "normal" and re.search(r"\b\d+\s*(?:-|–|to)?\s*\d*\s*(?:points|bullets)\b", lowered):
            route.detail = "brief"
        return route
    if units or route.missing_numbers:
        route.units = units
        if not units:
            route.intent = "explain"  # reference to a number that does not exist -> not found
        elif _COMPARE.search(lowered) and len(units) >= 2:
            route.intent = "compare"
        elif _COMPARE.search(lowered) and entities:
            route.units = [] if not route.explicit else units
            if not route.explicit:
                return _open_question(route, text, structure, entities, aspect)
            route.intent = "scoped_qa"
            route.mode = "compare"
            route.entities = entities
        elif _EXPLAIN.search(lowered) or len(units) > 1 and not _LOOKUP.search(lowered):
            route.intent = "explain"
        elif _LOOKUP.search(lowered) and len(re.findall(r"\w+", lowered)) <= 14 and route.explicit:
            route.intent = "lookup"
        elif not _residual_terms(text, structure, units):
            route.intent = "explain"  # "chapter 10?", "What is AC-2?", "what is guideline use?" on their own
        else:
            route.intent = "scoped_qa"
            if route.analysis:
                route.mode = "analysis"
        return route
    if route.pages:
        route.intent = "page"
        return route
    return _open_question(route, text, structure, entities, aspect)


def _open_question(route: Route, text: str, structure: dict, entities: list[str], aspect: str) -> Route:
    route.intent = "qa"
    route.role_sections = _role_refs(text, structure)
    content = _content_terms(text)
    if entities:
        route.mode = "compare"
        route.entities = entities
        route.aspect = aspect
        route.subqueries = [f"{entity} {aspect}".strip() for entity in entities]
    else:
        parts = _subqueries(text)
        if len(parts) >= 2:
            route.mode = "multi"
            route.subqueries = parts
        elif route.analysis:
            route.mode = "analysis"
        elif _ENUMERATE.search(text):
            route.mode = "enumerate"
        elif len(content) <= 2 and not _QUESTION_WORD.search(text) and len(text.split()) <= 5:
            route.mode = "ambiguous"
    if route.mode in ("multi", "compare") and route.analysis:
        route.analysis = True
    return route
