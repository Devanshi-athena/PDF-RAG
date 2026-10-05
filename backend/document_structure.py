"""Document outline model: sections, top-level units (chapters/parts/topics), numbering, roles, fields and captions."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .pdf_service import ParsedDocument

STRUCTURE_VERSION = 4

UNIT_NOUNS = {
    "chapter": "chapters", "section": "sections", "topic": "topics", "story": "stories",
    "part": "parts", "unit": "units", "lesson": "lessons", "module": "modules",
    "article": "articles", "book": "books", "episode": "episodes", "case": "cases",
    "annex": "annexes", "appendix": "appendices", "schedule": "schedules", "clause": "clauses",
}
# Labels that number items of content rather than divisions of the document ("Recommendation 11", "Pillar 2").
CONTENT_NOUNS = {
    "recommendation": "recommendations", "principle": "principles", "objective": "objectives", "outcome": "outcomes",
    "output": "outputs", "pillar": "pillars", "component": "components", "goal": "goals", "priority": "priorities",
    "step": "steps", "phase": "phases", "stage": "stages", "requirement": "requirements", "standard": "standards",
    "guideline": "guidelines", "rule": "rules", "measure": "measures", "action": "actions", "activity": "activities",
    "strategy": "strategies", "theme": "themes", "finding": "findings", "indicator": "indicators", "target": "targets",
    "criterion": "criteria", "question": "questions", "regulation": "regulations", "policy": "policies",
}
UNIT_NOUNS.update(CONTENT_NOUNS)
_NOUN_ALIASES = {"ch": "chapter", "chap": "chapter", "sec": "section", "art": "article", "annexure": "annex", "annexures": "annex", "appendixes": "appendix"}

_UNITS_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30, "fortieth": 40, "fiftieth": 50,
}
NUMBER_WORD_PATTERN = (
    r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen"
    r"|sixteen|seventeen|eighteen|nineteen)"
)
CONTENT_NOUN_PATTERN = (
    r"(?:recommendation|principle|objective|outcome|output|pillar|component|goal|priority|step|phase|stage|"
    r"requirement|standard|guideline|rule|measure|action|activity|strategy|theme|finding|indicator|target|"
    r"criterion|question|regulation|policy)"
)
NOUN_PATTERN = (
    r"(?:chapter|part|section|unit|lesson|story|topic|module|article|book|episode|case|annex(?:ure)?|appendix|"
    rf"schedule|clause|ch\.?|chap\.?|sec\.?|art\.?|{CONTENT_NOUN_PATTERN})"
)
_TITLE_NUMBER = re.compile(
    rf"^(?:(?P<noun>{NOUN_PATTERN})\s+)?"
    rf"(?P<num>\d{{1,4}}(?:\.\d{{1,3}})*|[A-Z](?:\.\d{{1,3}})+|[ivxlcdm]{{1,7}}|[a-z]|{NUMBER_WORD_PATTERN})"
    r"(?P<sep>[.:)\]]|\s*[-–—]|\s|$)\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_LABEL_LINE = re.compile(
    rf"^(?P<noun>{NOUN_PATTERN})\s+(?P<num>\d{{1,4}}|[ivxlcdm]{{1,7}}|{NUMBER_WORD_PATTERN})"
    r"(?:\s*(?:of|/)\s*(?P<total>\d{1,4}))?\b",
    re.IGNORECASE,
)
_IDENT = re.compile(r"^([A-Z]{1,6}-\d{1,4}(?:\(\d{1,3}\))?|[A-Z]{1,6}\s\d{1,2}\.\d{1,2})\b")
_FIELD = re.compile(r"^([A-Za-z][A-Za-z /&-]{0,30}?)\s*:\s*(.*)$")
_MATTER = re.compile(
    r"^(?:table of contents|contents|index|preface|foreword|acknowledg\w*|bibliography|references|works cited|"
    r"about the authors?|copyright|dedication|notes|endnotes|list of (?:figures|tables|abbreviations|acronyms|boxes)|"
    r"abbreviations|acronyms|abbreviations and acronyms|title page|cover|keywords|key words|errata|correction slip)$",
    re.IGNORECASE,
)
_CONTENTS_TITLE = re.compile(r"^\s*(?:table of contents|contents|list of (?:figures|tables|boxes|abbreviations|acronyms))\s*$", re.IGNORECASE)
ROLE_KEYWORDS = {
    "summary": ["abstract", "executive summary", "summary", "key messages", "key findings", "highlights", "at a glance", "overview", "synopsis"],
    "introduction": ["introduction", "background", "context", "rationale", "problem statement"],
    "objectives": ["objectives", "objective", "aims", "aim", "goals", "goal", "purpose", "scope", "mission", "vision"],
    "methods": ["methodology", "methods", "method", "approach", "materials and methods", "research design", "data and methods", "experimental setup"],
    "results": ["results", "findings", "outcomes", "evaluation", "experiments", "performance", "achievements"],
    "discussion": ["discussion", "analysis", "implications", "interpretation"],
    "conclusion": ["conclusion", "conclusions", "concluding remarks", "next steps", "way forward", "final remarks"],
    "recommendations": ["recommendations", "recommendation", "proposals", "actions", "policy recommendations", "action plan"],
    "limitations": ["limitations", "limitation", "challenges", "constraints", "risks", "risk", "lessons learned"],
    "definitions": ["definitions", "glossary", "terminology", "terms", "abbreviations", "acronyms", "a note on terminology"],
    "budget": ["budget", "financing", "funding", "costs", "cost", "financial", "resources", "resource requirements"],
    "governance": ["governance", "roles and responsibilities", "institutional arrangements", "implementation arrangements", "management arrangements", "oversight", "accountability"],
    "monitoring": ["monitoring", "evaluation", "monitoring and evaluation", "indicators", "reporting", "results framework", "performance measurement"],
    "implementation": ["implementation", "delivery", "timeline", "work plan", "workplan", "roadmap", "schedule", "phasing"],
    "references": ["references", "bibliography", "works cited"],
}


def parse_number(value: str) -> int | None:
    value = value.strip().casefold().rstrip(".")
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value in _ORDINALS:
        return _ORDINALS[value]
    parts = re.split(r"[- ]", value)
    if parts[0] in _TENS_WORDS:
        tail = _UNITS_WORDS.index(parts[1]) if len(parts) > 1 and parts[1] in _UNITS_WORDS else 0
        return _TENS_WORDS[parts[0]] + tail
    if value in _UNITS_WORDS:
        return _UNITS_WORDS.index(value)
    if re.fullmatch(r"[ivxlcdm]+", value) and value not in ("d", "c", "l", "m"):
        return _roman(value)
    if re.fullmatch(r"[a-z]", value):
        return ord(value) - ord("a") + 1
    return None


def _roman(value: str) -> int | None:
    numerals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total, previous = 0, 0
    for character in reversed(value):
        current = numerals[character]
        total += -current if current < previous else current
        previous = max(previous, current)
    return total if 0 < total < 4000 else None


def singular_noun(value: str | None) -> str | None:
    if not value:
        return None
    value = value.casefold().rstrip(".")
    value = _NOUN_ALIASES.get(value, value)
    if value in UNIT_NOUNS:
        return value
    for singular, plural in UNIT_NOUNS.items():
        if value == plural:
            return singular
    return None


def plural_noun(value: str) -> str:
    return UNIT_NOUNS.get(value, value + "s")


def is_content_noun(value: str | None) -> bool:
    return bool(value) and value in CONTENT_NOUNS


def normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", value.casefold())).strip()


def _parse_title(title: str) -> dict:
    title = title.strip()
    match = _TITLE_NUMBER.match(title)
    if not match:
        return {"name": title}
    noun = singular_noun(match.group("noun"))
    raw = match.group("num")
    rest = match.group("rest").strip(" :.-–—")
    is_path = bool(re.fullmatch(r"\d+(?:\.\d+)*|[A-Z](?:\.\d+)+", raw))
    single_letter = bool(re.fullmatch(r"[a-z]", raw, re.IGNORECASE))
    if not noun:
        # Without a label word only accept "12. Title", "3.2 Title" or "A.1 Title" style numbering.
        if not is_path or len(raw.split(".")[0]) > 3 or not rest:
            return {"name": title}
        if match.group("sep").strip() == "" and "." not in raw and not rest[:1].isupper():
            return {"name": title}
    elif single_letter and not raw.isupper() and not rest:
        return {"name": title}
    first = raw.split(".")[0]
    number = parse_number(first) if is_path else parse_number(raw)
    if number is None:
        return {"name": title}
    display = raw
    if noun and re.fullmatch(r"[ivxlcdm]+", raw, re.IGNORECASE) and not single_letter:
        display = raw.upper()
    elif single_letter:
        display = raw.upper()
    return {
        "name": rest or title,
        "noun": noun,
        "number": number,
        "number_path": raw.upper() if is_path and raw[0].isalpha() else raw if is_path else None,
        "display_number": display,
    }


def _roles(title: str) -> list[str]:
    name = normalize_title(re.sub(r"^\s*(?:[\dA-Z]+(?:\.\d+)*[.)]?\s+)", "", title))
    name = re.sub(rf"^{NOUN_PATTERN}\s+\S+\s*", "", name)
    roles = []
    for role, keywords in ROLE_KEYWORDS.items():
        if any(name == keyword or name.startswith(keyword + " ") or name.endswith(" " + keyword) or f" {keyword} " in f" {name} " for keyword in keywords):
            roles.append(role)
    return roles


def _section_opening(parsed: ParsedDocument, entries: list[dict], index: int, limit: int = 600) -> str:
    """Text right after a heading (up to the next heading), used for labels and fields."""
    entry = entries[index]
    following = entries[index + 1] if index + 1 < len(entries) else None
    pieces, remaining = [], limit
    for page, text in parsed.pages:
        if page < entry["page"]:
            continue
        if remaining <= 0 or (following and page > following["page"]):
            break
        start = entry["offset"] if page == entry["page"] else 0
        if page == entry["page"]:
            heading_end = text.find("\n\n", start)
            start = heading_end + 2 if heading_end >= 0 else len(text)
        stop = len(text)
        if following and following["page"] == page:
            stop = max(start, following["offset"])
        piece = text[start:stop][:remaining]
        pieces.append(piece)
        remaining -= len(piece)
        if following and following["page"] == page:
            break
    return "\n\n".join(piece for piece in pieces if piece.strip())


def _fields_from_opening(opening: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lines = [line.strip() for line in re.split(r"\n+", opening) if line.strip()][:4]
    pending = None
    for line in lines:
        if len(line) > 200:
            break
        if pending:
            fields[pending] = line.split("|")[0].strip()
            pending = None
            continue
        for part in line.split("|"):
            match = _FIELD.match(part.strip())
            if match and len(match.group(1).split()) <= 3:
                key = match.group(1).strip()
                value = match.group(2).strip()
                if value:
                    fields[key] = value
                else:
                    pending = key
    return fields


def build_structure(parsed: ParsedDocument) -> dict:
    entries = parsed.outline
    sections = []
    contents_numbers = getattr(parsed, "contents_numbers", {}) or {}
    for index, entry in enumerate(entries):
        info = _parse_title(entry["title"])
        if info.get("number") is None and not info.get("noun"):
            # The heading has no number but the contents page gives one ("Introduction" -> "1").
            path = contents_numbers.get(normalize_title(entry["title"]))
            if path:
                info.update(number=int(path.split(".")[0]), number_path=path, display_number=path)
        opening = _section_opening(parsed, entries, index)
        first_line = next((line.strip() for line in opening.splitlines() if line.strip()), "")
        label_match = _LABEL_LINE.match(first_line) if len(first_line) <= 80 else None
        label = ""
        if label_match:
            label = label_match.group(0)
            info["noun"] = info.get("noun") or singular_noun(label_match.group("noun"))
            number = parse_number(label_match.group("num"))
            if info.get("number") is None and number is not None:
                info["number"] = number
                info["display_number"] = label_match.group("num")
        ident = _IDENT.match(entry["title"].strip())
        sections.append({
            "index": index,
            "title": entry["title"],
            "name": info.get("name") or entry["title"],
            "level": int(entry["level"]),
            "page_start": int(entry["page"]),
            "offset": int(entry["offset"]),
            "noun": info.get("noun"),
            "number": info.get("number"),
            "number_path": info.get("number_path"),
            "display_number": info.get("display_number"),
            "ident": ident.group(1) if ident else None,
            "label": label,
            "fields": _fields_from_opening(opening),
            "matter": bool(_MATTER.match(normalize_title(entry["title"]))),
            "roles": _roles(entry["title"]),
            "parent": None,
            "children": [],
        })
    _repair_levels_from_numbering(sections)
    _demote_headings_inside_numbered_chapters(sections)
    last_page = parsed.pages[-1][0] if parsed.pages else 0
    for index, section in enumerate(sections):
        parent = _numbered_parent(sections, index)
        if parent is None:
            for previous in range(index - 1, -1, -1):
                if sections[previous]["level"] < section["level"]:
                    if _CONTENTS_TITLE.match(sections[previous]["title"]):
                        continue  # "Contents" heads a list of titles, not the sections after it
                    parent = previous
                    break
        if parent is not None:
            section["parent"] = parent
            sections[parent]["children"].append(index)
        end = last_page
        for following in sections[index + 1:]:
            if following["level"] <= section["level"]:
                end = following["page_start"] - (1 if following["offset"] == 0 else 0)
                break
        section["page_end"] = max(section["page_start"], end)

    title = parsed.title
    # A long, unnumbered first bookmark on page 1 is the document's own title, not a chapter.
    if sections and not title:
        first = sections[0]
        if first["page_start"] == 1 and first["number"] is None and len(first["title"].split()) >= 6 and len(sections) >= 3:
            title = first["title"]
            first["matter"] = True
    _infer_parent_numbers(sections)
    _adopt_part_members(sections, last_page)
    _mark_front_matter(sections, parsed.page_count or last_page)

    unit_level = _choose_unit_level(sections)
    units = [s["index"] for s in sections if s["level"] == unit_level and not s["matter"]] if unit_level else []
    # Annexes/appendices are back matter: they don't decide what the main parts are called.
    body_units = [i for i in units if sections[i]["noun"] not in _BACK_MATTER_NOUNS]
    nouns = Counter(sections[i]["noun"] for i in body_units if sections[i]["noun"])
    unit_noun = nouns.most_common(1)[0][0] if nouns and nouns.most_common(1)[0][1] * 2 >= len(body_units) else None
    _name_label_only_sections(sections)
    numbered = sum(1 for i in units if sections[i]["number"] is not None)
    explicit_numbers = bool(units) and numbered * 2 >= len(units)
    for ordinal, index in enumerate(units, start=1):
        sections[index]["ordinal"] = ordinal
    field_counts = Counter(key for i in units for key in sections[i]["fields"])
    common_fields = [key for key, count in field_counts.items() if count * 10 >= len(units) * 3]
    for section in sections:
        section["fields"] = {k: v for k, v in section["fields"].items() if k in common_fields}
    return {
        "version": STRUCTURE_VERSION,
        "title": title,
        "page_count": parsed.page_count or len(parsed.pages),
        "outline_source": parsed.outline_source,
        "sections": sections,
        "unit_level": unit_level,
        "units": units,
        "unit_noun": unit_noun,
        "explicit_numbers": explicit_numbers,
        "fields": common_fields,
        "captions": _place_captions(parsed.captions, sections),
        "tables": _place_captions(getattr(parsed, "tables", []), sections),
    }


def _repair_levels_from_numbering(sections: list[dict]) -> None:
    """Bookmarks are sometimes nested wrongly (3.2.5 under 3.2.4); numbered headings of equal depth share a level."""
    depths = [len(s["number_path"].split(".")) if s["number_path"] else None for s in sections]
    numbered = [d for d in depths if d]
    if len(numbered) < 3 or len(numbered) * 2 < len(sections):
        return
    level_for: dict[int, int] = {}
    for depth in set(numbered):
        levels = Counter(s["level"] for s, d in zip(sections, depths) if d == depth)
        level_for[depth] = levels.most_common(1)[0][0]
    for section, depth in zip(sections, depths):
        if depth and section["level"] != level_for[depth]:
            section["level"] = level_for[depth]


_BACK_MATTER_NOUNS = {"annex", "appendix", "schedule"}
_LABEL_ONLY_TITLE = re.compile(rf"^\s*{NOUN_PATTERN}\s+[\w.]+\s*[:.\-–—]?\s*$", re.IGNORECASE)


def _name_label_only_sections(sections: list[dict]) -> None:
    """"Annexure I" whose descriptive title is the heading right under it takes that title as its name."""
    for section in sections:
        if not section["noun"] or not _LABEL_ONLY_TITLE.match(section["title"]) or not section["children"]:
            continue
        first = sections[section["children"][0]]
        if first["page_start"] == section["page_start"] and not first["noun"] and not first["number_path"]:
            section["name"] = first["title"]


def _demote_headings_inside_numbered_chapters(sections: list[dict]) -> None:
    """Unnumbered headings at chapter level that sit between "4.5" and "4.6" (table headers, call-outs) belong to
    chapter 4, not beside it."""
    def chapter(path):
        return path.split(".")[0] if path and "." in path else None

    for index, section in enumerate(sections):
        if section["number_path"] or section["noun"] or section["matter"]:
            continue
        before = next((chapter(s["number_path"]) for s in reversed(sections[:index]) if chapter(s["number_path"])), None)
        after = None
        for following in sections[index + 1:]:
            if following["level"] <= section["level"] and (following["noun"] or (following["number_path"] and "." not in following["number_path"])):
                break
            if chapter(following["number_path"]):
                after = chapter(following["number_path"])
                break
        dotted_level = next((s["level"] for s in sections[index + 1:] if chapter(s["number_path"])), None)
        if before and before == after and dotted_level and section["level"] < dotted_level:
            section["level"] = dotted_level


def _numbered_parent(sections: list[dict], index: int) -> int | None:
    """"4.6" belongs under the nearest earlier "4", whatever headings came in between."""
    path = sections[index]["number_path"]
    if not path or "." not in path:
        return None
    prefix = path.rsplit(".", 1)[0]
    for previous in range(index - 1, -1, -1):
        candidate = sections[previous]
        if candidate["number_path"] == prefix and candidate["level"] < sections[index]["level"]:
            return previous
        if candidate["number_path"] and "." not in candidate["number_path"] and candidate["number_path"] != prefix.split(".")[0]:
            return None  # crossed into another chapter
    return None


def _mark_front_matter(sections: list[dict], page_count: int) -> None:
    """Cover pages, messages and forewords before the table of contents are not chapters."""
    contents = next((s for s in sections if _CONTENTS_TITLE.match(s["title"])), None)
    if contents is None or contents["page_start"] > max(5, page_count * 0.25):
        return
    for section in sections:
        if section is contents:
            break
        if section["number"] is None and "summary" not in section["roles"]:
            section["matter"] = True


def _adopt_part_members(sections: list[dict], last_page: int) -> None:
    """Flat outlines list "Part 1" and its chapters as siblings; let the part own the chapters up to the next part."""
    grouping = ("part", "book")
    for index, section in enumerate(sections):
        if section["noun"] not in grouping or section["children"]:
            continue
        members = []
        for following in sections[index + 1:]:
            if following["level"] < section["level"]:
                break
            if following["level"] == section["level"] and (
                following["noun"] in grouping + ("annex", "appendix", "schedule") or following["matter"]
            ):
                break
            if following["level"] == section["level"]:
                members.append(following["index"])
        if members:
            section["children"] = members
            last = sections[members[-1]]
            section["page_end"] = max(section["page_end"], last.get("page_end", last["page_start"]))


def _infer_parent_numbers(sections: list[dict]) -> None:
    """"INTRODUCTION" whose sub-sections are 1.1, 1.2 ... is section 1; "A Extra" with A.1, A.2 is appendix A."""
    for section in sections:
        paths = [sections[c]["number_path"] for c in section["children"] if sections[c]["number_path"]]
        if not paths:
            continue
        prefixes = {path.split(".")[0] for path in paths if "." in path}
        if len(prefixes) != 1 or len(paths) < 2:
            continue
        prefix = prefixes.pop()
        if section["number_path"] and section["number_path"] != prefix:
            continue
        section["number_path"] = prefix
        if section["number"] is None:
            section["number"] = int(prefix) if prefix.isdigit() else ord(prefix.upper()) - ord("A") + 1
            section["display_number"] = prefix
        if section["title"].startswith(prefix + " ") and section["name"] == section["title"]:
            section["name"] = section["title"][len(prefix) + 1:].strip()


def _choose_unit_level(sections: list[dict]) -> int | None:
    levels = sorted({section["level"] for section in sections})
    for level in levels:
        if sum(1 for s in sections if s["level"] == level and s["noun"] == "chapter") >= 2:
            return level
    for level in levels:
        if sum(1 for s in sections if s["level"] == level and not s["matter"]) >= 2:
            return level
    return levels[0] if levels else None


def _place_captions(captions: list[dict], sections: list[dict]) -> list[dict]:
    placed = []
    for caption in captions:
        section = -1
        for candidate in sections:
            if (candidate["page_start"], candidate["offset"]) <= (caption["page"], caption.get("offset", 0)):
                section = candidate["index"]
            else:
                break
        placed.append({**caption, "section": section})
    return placed


def annotate_chunks(chunks: list[dict], structure: dict) -> None:
    """Attach unit index and a readable section path to every chunk."""
    sections = structure.get("sections", [])
    unit_level = structure.get("unit_level")
    for chunk in chunks:
        index = chunk.get("section", -1)
        unit, path = -1, []
        while index is not None and 0 <= index < len(sections):
            section = sections[index]
            path.append(section["title"])
            if section["level"] == unit_level and not section["matter"]:
                unit = index
            index = section["parent"]
        chunk["unit"] = unit
        chunk["section_path"] = " > ".join(reversed(path))


def descendants(structure: dict, index: int) -> list[int]:
    sections = structure["sections"]
    result, stack = [], [index]
    while stack:
        current = stack.pop()
        result.append(current)
        stack.extend(reversed(sections[current].get("children", [])))
    return sorted(result)


def unit_noun_for(structure: dict, section: dict, requested: str | None = None) -> str:
    return section.get("noun") or requested or structure.get("unit_noun") or "section"


def unit_label(structure: dict, section: dict, requested: str | None = None) -> str:
    """"Story 12", "Annex B", "Topic 073"; the PDF's own title when the section carries no number."""
    number = section.get("display_number") or section.get("number")
    if number is None:
        return section["title"]
    noun = section.get("noun")
    if not noun:
        if not (requested or structure.get("unit_noun")):
            return section["title"]
        noun = requested or structure.get("unit_noun")
    elif singular_noun(section["title"].split()[0]) == noun and len(section["title"].split()) >= 2:
        return " ".join(section["title"].split()[:2]).rstrip(":.")  # the PDF's own wording: "Annexure I", "Part Two"
    return f"{noun.title()} {number}"


def unit_heading(structure: dict, section: dict, requested: str | None = None) -> str:
    label = unit_label(structure, section, requested)
    if label == section["title"] and section["name"] != section["title"] and _LABEL_ONLY_TITLE.match(section["title"]):
        return f"{section['title']}: {section['name']}"  # "Annexure I: Composition of the Committee on NHS"
    if label == section["title"] or section["name"] == section["title"] and section["title"].startswith(label):
        return section["title"]
    return f"{label}: {section['name']}" if section["name"] != section["title"] else f"{label}: {section['title']}"


def section_path(structure: dict, index: int) -> str:
    sections = structure["sections"]
    parts = []
    while index is not None and 0 <= index < len(sections):
        parts.append(sections[index]["title"])
        index = sections[index]["parent"]
    return " > ".join(reversed(parts))


def pages_text(section: dict) -> str:
    start, end = section["page_start"], section.get("page_end", section["page_start"])
    return f"Page {start}" if start == end else f"Pages {start}–{end}"


def save_structure(path: Path, structure: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(structure, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_structure(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if value.get("version") == STRUCTURE_VERSION else None
