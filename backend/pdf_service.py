from __future__ import annotations

import io
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .errors import BackendError

_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty"
)
_LABEL_NOUNS = r"(?:chapter|part|section|unit|lesson|story|topic|module|article|book|annex|appendix|schedule|clause)"
_LABEL_ONLY = re.compile(
    rf"^{_LABEL_NOUNS}\s+"
    rf"(?:\d{{1,4}}(?:\.\d{{1,3}})*|[ivxlcdm]{{1,7}}|[a-z]|(?:{_NUMBER_WORDS})(?:[- ](?:{_NUMBER_WORDS}))?)[.:]?$",
    re.IGNORECASE,
)
_NUMBER_ONLY = re.compile(r"^(?:\d{1,3}(?:\.\d{1,3})*|[A-Z](?:\.\d{1,3})*|[IVXLC]{1,6})[.)]?$")
_TEXT_CHAPTER = re.compile(
    rf"^(?:chapter|part|annex|appendix)\s+(?:\d{{1,4}}|[ivxlcdm]{{1,7}}|[a-z]|(?:{_NUMBER_WORDS}))\b.{{0,90}}$",
    re.IGNORECASE,
)
_TEXT_NUMBERED = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,3})[.)]?\s+([A-Z][^.!?]{2,90})$")
_HEADING_NUMBER = re.compile(r"^(\d{1,3}(?:\.\d{1,3}){0,4}|[A-Z](?:\.\d{1,3}){1,3})[.)]?\s+\S")
_TOC_LINE = re.compile(r"(?:\.{3,}|…{2,}|\s{2,}|\t)\s*\d{1,4}\s*$")
_CAPTION = re.compile(
    r"^(?P<kind>table|figure|fig\.|exhibit|box|chart)\s+"
    r"(?P<number>(?:[A-Z][.\-])?\d{1,3}(?:[.\-]\d{1,3})*|[A-Z])(?P<sep>\s*[:.\-–—|]\s*|\s+)(?P<text>\S.*)$",
    re.IGNORECASE,
)
_PAGE_NUMBER_LINE = re.compile(r"^(?:page\s+)?[-–—\s]*(?:\d{1,4}|[ivxlcdm]{1,6})(?:\s*(?:of|/)\s*\d{1,4})?[-–—\s]*$", re.IGNORECASE)
_BULLET_START = re.compile(r"^(?:[-•*▪◦·●■–]|\(?\d{1,3}[.)]|\(?[a-z][.)]|\([ivx]+\))\s", re.IGNORECASE)
_SUMMARY_TITLE = re.compile(r"^(?:abstract|executive summary|summary)\b", re.IGNORECASE)
_BAND = 0.07


@dataclass
class ParsedDocument:
    """Clean page text plus a document outline and figure/table captions located inside that text."""

    pages: list[tuple[int, str]]
    outline: list[dict] = field(default_factory=list)
    page_count: int = 0
    title: str = ""
    outline_source: str = "none"
    captions: list[dict] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    contents_numbers: dict[str, str] = field(default_factory=dict)  # normalized title -> "4" / "4.2" from the contents page


def extract_document(pdf_bytes: bytes) -> ParsedDocument:
    """Read the PDF's text layer. Pages without one (scanned/image-only) are skipped: OCR is not used."""
    started = time.perf_counter()
    try:
        import pymupdf
        document = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
    except Exception as error:
        raise BackendError("The uploaded file is not a readable PDF.", 422, "invalid_pdf") from error
    raw_pages = []
    try:
        page_count = document.page_count
        try:
            toc = document.get_toc(simple=True)
        except Exception:
            toc = []
        flags = pymupdf.TEXTFLAGS_TEXT
        skipped = []
        for number, page in enumerate(document, start=1):
            lines = _page_lines(page.get_text("dict", flags=flags), page.rect.width)
            if not lines:
                skipped.append(number)
            raw_pages.append((number, page.rect.height, lines))
    finally:
        document.close()
    if skipped:
        print(f"[PERF] pages_without_text={len(skipped)} skipped={skipped[:20]}{'...' if len(skipped) > 20 else ''}")

    body_size = _body_font_size(raw_pages)
    _remove_running_lines(raw_pages)
    toc_pages = {number for number, _, lines in raw_pages if _is_contents_page(lines)}
    pages: list[tuple[int, str]] = []
    font_headings: list[dict] = []
    captions: list[dict] = []
    tables: list[dict] = []
    for number, _, lines in raw_pages:
        text, headings, page_captions, page_tables = _page_text(lines, body_size)
        if text:
            pages.append((number, text))
            if number not in toc_pages:
                tables.extend({**table, "page": number} for table in page_tables)
            if number in toc_pages:
                # Contents / list-of-figures pages repeat every title; keep only their own heading.
                headings = [h for h in headings if re.match(r"^(?:table of )?contents|list of", h["title"], re.I)]
                page_captions = []
            font_headings.extend({**heading, "page": number} for heading in headings)
            captions.extend({**caption, "page": number} for caption in page_captions)
    if not pages:
        raise BackendError(
            "No text could be extracted from this PDF. It looks scanned (images only); scanned PDFs are not supported.",
            422,
            "empty_document",
        )

    parsed = ParsedDocument(
        pages=pages, page_count=page_count, captions=_dedupe_captions(captions), tables=tables,
        contents_numbers=_contents_numbers(pages, toc_pages),
    )
    outline = _outline_from_toc(toc, pages)
    if outline:
        parsed.outline, parsed.outline_source = outline, "bookmarks"
    else:
        outline, title = _outline_from_fonts(font_headings, len(pages))
        if outline:
            parsed.outline, parsed.title, parsed.outline_source = outline, title, "fonts"
        else:
            parsed.outline = _outline_from_text(pages)
            parsed.outline_source = "text" if parsed.outline else "none"
    print(
        f"[PERF] pdf_extraction_internal_seconds={time.perf_counter() - started:.4f} "
        f"outline_source={parsed.outline_source} outline_entries={len(parsed.outline)} captions={len(parsed.captions)}"
    )
    return parsed


def extract_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    return extract_document(pdf_bytes).pages


# --------------------------------------------------------------------------- layout
def _page_lines(data: dict, page_width: float) -> list[dict]:
    lines = []
    for block_index, block in enumerate(data.get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        block_lines = []
        for line in block.get("lines", []):
            direction = line.get("dir", (1, 0))
            if abs(direction[1]) > 0.2:
                continue  # rotated margin text (arXiv stamps, vertical labels)
            spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
            if not spans:
                continue
            text = unicodedata.normalize("NFKC", "".join(span["text"] for span in line["spans"]))
            text = re.sub(r"\s+", " ", text).strip()
            if not text or not any(character.isalnum() for character in text):
                continue
            dominant = max(spans, key=lambda span: len(span["text"].strip()))
            bold = all(
                (span.get("flags", 0) & 16) or "bold" in span.get("font", "").casefold()
                for span in spans
            )
            block_lines.append({
                "text": text,
                "size": round(float(dominant.get("size", 0.0)) * 2) / 2,
                "bold": bool(bold),
                "block": block_index,
                "x0": float(line["bbox"][0]),
                "y0": float(line["bbox"][1]),
                "x1": float(line["bbox"][2]),
                "y1": float(line["bbox"][3]),
                "table": False,
            })
        for line in block_lines:
            line["block_lines"] = len(block_lines)
        lines.extend(block_lines)
    return _rebuild_table_rows(lines, page_width)


def _rebuild_table_rows(lines: list[dict], page_width: float) -> list[dict]:
    """Merge cells that share a baseline into "a | b | c" rows when 2+ consecutive rows have 3+ narrow cells."""
    if len(lines) < 6 or not page_width:
        return lines
    ordered = sorted(range(len(lines)), key=lambda i: (lines[i]["y0"] + lines[i]["y1"]) / 2)
    bands: list[list[int]] = []
    for index in ordered:
        line = lines[index]
        center = (line["y0"] + line["y1"]) / 2
        tolerance = max(2.0, (line["y1"] - line["y0"]) * 0.35)
        if bands:
            last = lines[bands[-1][0]]
            if abs((last["y0"] + last["y1"]) / 2 - center) <= tolerance:
                bands[-1].append(index)
                continue
        bands.append([index])
    widths: dict[int, list[float]] = {}
    for line in lines:
        widths.setdefault(line["block"], []).append(line["x1"] - line["x0"])
    prose_blocks = {
        block for block, values in widths.items()
        if len(values) >= 3 and sorted(values)[len(values) // 2] >= page_width * 0.25
    }

    def is_cell(line):
        return (
            line["block"] not in prose_blocks
            and line["x1"] - line["x0"] < page_width * 0.3
            and len(line["text"]) <= 60
        )

    is_row = [
        len(band) >= 3 and all(is_cell(lines[i]) for i in band)
        and sum(1 for i in band if len(lines[i]["text"]) > 40) <= 1
        for band in bands
    ]
    tables: list[list[list[int]]] = []
    current: list[list[int]] = []
    for band, row in zip(bands, is_row):
        if row:
            current.append(band)
        else:
            if len(current) >= 2:
                tables.append(current)
            current = []
    if len(current) >= 2:
        tables.append(current)
    if not tables:
        return lines
    replacement: dict[int, list[dict]] = {}
    consumed: set[int] = set()
    for table in tables:
        rows = []
        for band in table:
            cells = sorted(band, key=lambda i: lines[i]["x0"])
            first = lines[cells[0]]
            rows.append({
                **first,
                "text": " | ".join(lines[i]["text"] for i in cells),
                "x0": min(lines[i]["x0"] for i in cells),
                "x1": max(lines[i]["x1"] for i in cells),
                "bold": False,
                "block": -1 - len(replacement),
                "table": True,
            })
            consumed.update(band)
        anchor = min(i for band in table for i in band)
        replacement[anchor] = rows
    result = []
    for index, line in enumerate(lines):
        if index in replacement:
            result.extend(replacement[index])
        elif index not in consumed:
            result.append(line)
    return result


_ENDS_WITH_PAGE = re.compile(r"[A-Za-z][^|]*?(?:\.{2,}|…|\s)\s*\d{1,3}$")


def _is_contents_page(lines: list[dict]) -> bool:
    """A table of contents / list of figures: many lines ending in a page number."""
    texts = [line["text"] for line in lines if not line["table"]]
    rows = [line["text"].split(" | ") for line in lines if line["table"]]
    # Contents rows look like "1.2 | Title | 5"; wide data-table rows ("EMP-101 | ... | 88") are not contents.
    toc_rows = sum(1 for cells in rows if len(cells) <= 3 and re.fullmatch(r"\d{1,3}", cells[-1]) and any(c.isalpha() for c in "".join(cells[:-1])))
    texts += [" ".join(cells) for cells in rows]
    if len(texts) < 6:
        return False
    toc_like = toc_rows + sum(1 for text in texts[: len(texts) - len(rows)] if _ENDS_WITH_PAGE.search(text))
    # Right-aligned page numbers are separate items on the same baseline as their title.
    numbers = [line for line in lines if re.fullmatch(r"\d{1,3}", line["text"])]
    toc_like += sum(
        1 for number in numbers
        if any(other is not number and abs(other["y0"] - number["y0"]) < 2.5 and other["x1"] < number["x0"] for other in lines)
    )
    titled = any(re.match(r"^(?:table of )?contents$|^list of (?:figures|tables|boxes)", text, re.I) for text in texts)
    return toc_like >= 6 and (toc_like >= len(texts) * 0.35 or (titled and toc_like >= len(texts) * 0.2))


_CONTENTS_ENTRY = re.compile(
    r"(?:^|(?<=\s)|(?<=\+))\s*(?P<number>\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+(?P<title>[A-Z][^+\n|]*?)\s*(?:\.{2,}|…)?\s*\|?\s*(?P<page>\d{1,3})(?=\s|$)"
)


def _contents_numbers(pages: list[tuple[int, str]], toc_pages: set[int]) -> dict[str, str]:
    """Chapter numbers printed only on the contents page ("1. Introduction .... 18") keyed by normalized title."""
    numbers: dict[str, str] = {}
    for page, text in pages:
        if page not in toc_pages:
            continue
        for match in _CONTENTS_ENTRY.finditer(text.replace("\n", " ")):
            title = re.sub(r"\s+", " ", match.group("title")).strip(" .:")
            key = re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", title.casefold())).strip()
            if 3 <= len(key) <= 120 and key not in numbers:
                numbers[key] = match.group("number")
    return numbers


def _body_font_size(raw_pages) -> float:
    sizes = Counter()
    for _, _, lines in raw_pages:
        for line in lines:
            if line["size"] and not line["table"]:
                sizes[line["size"]] += len(line["text"])
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _remove_running_lines(raw_pages) -> None:
    """Drop repeated headers/footers and bare page numbers in the top/bottom page bands."""
    if not raw_pages:
        return

    def in_band(line, height):
        return height and line["y1"] > 1 and (line["y1"] < height * _BAND or line["y0"] > height * (1 - _BAND))

    def key(text):
        return re.sub(r"\d+", "#", text.casefold()).strip()

    counts = Counter()
    for _, height, lines in raw_pages:
        counts.update({key(line["text"]) for line in lines if in_band(line, height)})
    threshold = max(3, int(len(raw_pages) * 0.4))
    repeated = {value for value, count in counts.items() if count >= threshold}
    for index, (number, height, lines) in enumerate(raw_pages):
        kept = [
            line for line in lines
            if not (in_band(line, height) and (key(line["text"]) in repeated or _PAGE_NUMBER_LINE.match(line["text"])))
        ]
        raw_pages[index] = (number, height, kept)


def _column_edges(lines: list[dict]) -> dict:
    """Right text edge per block (multi-line blocks) and per left margin (lines sharing an x0)."""
    by_bucket: dict[int, list[float]] = {}
    by_block: dict[int, list[float]] = {}
    for line in lines:
        by_bucket.setdefault(int(line["x0"] // 12), []).append(line["x1"])
        by_block.setdefault(line["block"], []).append(line["x1"])
    buckets = {key: sorted(values)[int(len(values) * 0.9)] if len(values) > 3 else max(values) for key, values in by_bucket.items()}
    blocks = {key: max(values) for key, values in by_block.items() if len(values) >= 2}
    return {"buckets": buckets, "blocks": blocks}


def _column_right(line: dict, edges: dict) -> float:
    if line["block"] in edges["blocks"]:
        return edges["blocks"][line["block"]]
    bucket = int(line["x0"] // 12)
    buckets = edges["buckets"]
    return max(buckets.get(bucket - 1, 0.0), buckets.get(bucket, 0.0), buckets.get(bucket + 1, 0.0))


def _reaches_margin(line: dict | None, right_edge: float) -> bool:
    return bool(line and right_edge and line["x1"] >= right_edge - max(18.0, (right_edge - line["x0"]) * 0.12))


def _same_baseline(previous: dict, line: dict) -> bool:
    return abs(previous["y0"] - line["y0"]) < 2.0 and line["x0"] > previous["x1"] - 1


def _is_heading(paragraph: dict, body_size: float) -> bool:
    text = paragraph["text"]
    words = text.split()
    if paragraph["table"] or not text or len(text) > 160 or len(words) > 20 or not any(ch.isalpha() for ch in text):
        return False
    if re.search(r"[,;]$", text) or _PAGE_NUMBER_LINE.match(text) or _TOC_LINE.search(text) or _CAPTION.match(text):
        return False
    if sum(ch.isdigit() for ch in text) > len(text) * 0.4:
        return False
    size = paragraph["size"]
    if body_size and size >= body_size * 1.15:
        return True
    return bool(
        paragraph["bold"]
        and paragraph["whole_block"]
        and body_size
        and size >= body_size - 0.5
        and len(words) <= 14
        and not re.search(r"[.:]$", text)
    )


def _page_text(lines: list[dict], body_size: float) -> tuple[str, list[dict], list[dict], list[dict]]:
    edges = _column_edges([line for line in lines if not line["table"]])
    paragraphs: list[dict] = []
    for line in lines:
        previous = paragraphs[-1] if paragraphs else None
        last = previous["lines"][-1] if previous else None
        same_style = previous and (
            abs(previous["size"] - line["size"]) < 0.6
            and previous["bold"] == line["bold"]
            and previous["table"] == line["table"]
        )
        standalone = _TEXT_CHAPTER.match(line["text"]) or _LABEL_ONLY.match(line["text"]) or _CAPTION.match(line["text"])
        starts_after_heading = previous and len(previous["lines"]) == 1 and (
            _TEXT_CHAPTER.match(last["text"]) or _LABEL_ONLY.match(last["text"])
        )
        if same_style and line["table"]:
            continues = line["y0"] - last["y1"] <= max(14.0, line["size"] * 1.8)
        else:
            continues = same_style and not standalone and not starts_after_heading and (
                previous["block"] == line["block"]
                or _same_baseline(last, line)
                or (
                    _reaches_margin(last, _column_right(last, edges))
                    and 0 <= line["y0"] - last["y1"] <= max(4.0, line["size"] * 0.8)
                    and not _BULLET_START.match(line["text"])
                )
            )
        # A number on its own ("2.1") next to a heading title on the same baseline belongs to that title.
        if previous and not continues and _NUMBER_ONLY.match(last["text"]) and len(previous["lines"]) == 1 and (
            _same_baseline(last, line) or abs(last["y0"] - line["y0"]) < 3
        ):
            continues = True
            previous.update(size=line["size"], bold=line["bold"])
        if continues:
            previous["lines"].append(line)
        else:
            paragraphs.append({
                "block": line["block"], "size": line["size"], "bold": line["bold"],
                "table": line["table"], "lines": [line],
            })
    for paragraph in paragraphs:
        if paragraph["table"]:
            paragraph["text"] = "\n".join(line["text"] for line in paragraph["lines"])
        else:
            paragraph["text"] = _join_lines(paragraph["lines"], edges)
        paragraph["whole_block"] = len(paragraph["lines"]) >= paragraph["lines"][0].get("block_lines", 1)
        paragraph["heading"] = _is_heading(paragraph, body_size)
        if paragraph["heading"]:
            paragraph["text"] = re.sub(r"\s*\n\s*", " ", paragraph["text"])

    merged: list[dict] = []
    for paragraph in paragraphs:
        previous = merged[-1] if merged else None
        # A chapter number set apart in its own (often larger) font: "1" + "Introduction" -> "1. Introduction".
        if previous and paragraph["heading"] and not previous["heading"] and not previous["table"] and re.fullmatch(r"\d{1,2}\.?", previous["text"]):
            previous.update(
                text=previous["text"].rstrip(".") + ". " + paragraph["text"], heading=True,
                size=paragraph["size"], bold=paragraph["bold"],
            )
            continue
        if previous and previous["heading"] and paragraph["heading"] and (
            _LABEL_ONLY.match(previous["text"])
            or _NUMBER_ONLY.match(previous["text"])
            or (abs(previous["size"] - paragraph["size"]) < 0.3 and previous["bold"] == paragraph["bold"]
                and len(previous["text"]) + len(paragraph["text"]) <= 150
                and not _HEADING_NUMBER.match(paragraph["text"]))
        ):
            separator = ": " if _LABEL_ONLY.match(previous["text"]) else " "
            previous["text"] = previous["text"].rstrip(":. ") + separator + paragraph["text"]
            previous["size"] = max(previous["size"], paragraph["size"])
            continue
        merged.append(paragraph)

    parts, headings, captions, tables, offset = [], [], [], [], 0
    for paragraph in merged:
        if parts:
            offset += 2
        if paragraph["table"] and len(paragraph["lines"]) >= 2:
            tables.append({
                "offset": offset, "end": offset + len(paragraph["text"]),
                "header": paragraph["lines"][0]["text"], "rows": len(paragraph["lines"]),
            })
        if paragraph["heading"]:
            headings.append({
                "title": paragraph["text"], "offset": offset, "size": paragraph["size"],
                "bold": paragraph["bold"],
                "font_based": bool(body_size and paragraph["size"] >= body_size * 1.15),
            })
        elif not paragraph["table"]:
            caption = _caption(paragraph["text"])
            if caption:
                captions.append({
                    "kind": _caption_kind(caption.group("kind")),
                    "number": caption.group("number"),
                    "text": re.sub(r"\s+", " ", paragraph["text"])[:300],
                    "offset": offset,
                })
        parts.append(paragraph["text"])
        offset += len(paragraph["text"])
    return "\n\n".join(parts), headings, captions, tables


def _caption(text: str):
    """A figure/table caption, not a sentence that merely starts with "Figure 1 shows..." or a list-of-figures entry."""
    match = _CAPTION.match(text)
    if not match or _TOC_LINE.search(text.splitlines()[0]):
        return None
    if not match.group("sep").strip():
        first_word = match.group("text").split()[0]
        if not first_word[:1].isupper() or len(text) > 300 or first_word.casefold() in _CAPTION_VERBS:
            return None
    return match


_CAPTION_VERBS = {
    "illustrates", "shows", "presents", "provides", "lists", "describes", "summarizes", "summarises", "depicts",
    "contains", "gives", "highlights", "outlines", "compares", "displays", "reports", "is", "was", "and", "in",
}


def _caption_kind(value: str) -> str:
    value = value.casefold().rstrip(".")
    return {"fig": "figure", "chart": "figure", "diagram": "figure", "graph": "figure", "map": "figure"}.get(value, value)


def _dedupe_captions(captions: list[dict]) -> list[dict]:
    """Keep the first real caption per (kind, number); list-of-figures pages repeat every caption."""
    per_page = Counter(caption["page"] for caption in captions)
    best: dict[tuple[str, str], dict] = {}
    for caption in captions:
        key = (caption["kind"], caption["number"].casefold())
        if key not in best or (per_page[best[key]["page"]] > 3 and per_page[caption["page"]] <= 3):
            best[key] = caption
    return sorted(best.values(), key=lambda item: (item["page"], item["offset"]))


def _join_lines(lines: list[dict], edges: dict) -> str:
    text = lines[0]["text"]
    for previous, line in zip(lines, lines[1:]):
        if _same_baseline(previous, line):
            text += " " + line["text"]
        elif text.endswith("-") and line["text"][:1].islower():
            text = text[:-1] + line["text"]
        elif _reaches_margin(previous, _column_right(previous, edges)) and not _BULLET_START.match(line["text"]):
            text += " " + line["text"]
        else:
            text += "\n" + line["text"]
    return text


# -------------------------------------------------------------------------- outline
def _locate(title: str, text: str) -> int:
    words = title.split()
    if not words:
        return 0
    match = re.search(r"\s+".join(re.escape(word) for word in words), text, re.IGNORECASE)
    return match.start() if match else 0


def _outline_from_toc(toc, pages) -> list[dict]:
    page_text = dict(pages)
    if not page_text:
        return []
    last = max(page_text)
    entries = []
    for item in toc or []:
        if len(item) < 3:
            continue
        level = int(item[0])
        title = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(item[1]))).strip()
        page = int(item[2])
        if not title or page < 1:
            continue
        while page not in page_text and page <= last:
            page += 1
        if page not in page_text:
            continue
        entries.append({"title": title, "level": max(1, level), "page": page, "offset": _locate(title, page_text[page])})
    if len(entries) < 2:
        return []
    return _dedupe_outline(entries)


def _looks_garbled(title: str) -> bool:
    """Text from fonts without a proper encoding (e.g. legacy Hindi fonts) comes out as Latin gibberish."""
    if sum(ch.isalpha() for ch in title) < 4 or re.search(r"[{}\[\]\\<>@~^]", title):
        return True
    words = [w for w in re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", title) if len(w) > 1]
    if len(words) < 3:
        return False
    odd = sum(
        1 for w in words
        if (not re.search(r"[aeiouyAEIOUY]", w) and not w.isupper())  # no vowel and not an acronym
        or (re.search(r"[a-z][A-Z]", w) and not re.fullmatch(r"[a-z]{1,2}(?:[A-Z][a-z]+)+", w))  # "qR", not "eHealth"
    )
    return odd / len(words) >= 0.25


def _numbering_depth(title: str) -> int | None:
    match = _HEADING_NUMBER.match(title)
    if not match:
        return None
    return match.group(1).count(".") + 1


def _outline_from_fonts(headings: list[dict], page_count: int) -> tuple[list[dict], str]:
    if not headings:
        return [], ""
    headings = [h for h in headings if not _looks_garbled(h["title"])]
    if not headings:
        return [], ""
    # A title repeated at the top of many pages is a running header, not a new section each time.
    pages_per_title = Counter()
    for heading in {(h["title"].casefold(), h["page"]) for h in headings}:
        pages_per_title[heading[0]] += 1
    running = {title for title, count in pages_per_title.items() if count >= 3 and count >= page_count * 0.3}
    running_title = ""
    if running:
        first = next(h for h in headings if h["title"].casefold() in running)
        running_title = first["title"]
        headings = [h for h in headings if h["title"].casefold() not in running]
    # Front matter on the first page(s) (authors, affiliations) precedes the abstract/summary: drop it.
    summary = next((h for h in headings if h["page"] <= 3 and _SUMMARY_TITLE.match(h["title"])), None)
    title = ""
    if summary:
        before = [h for h in headings if (h["page"], h["offset"]) < (summary["page"], summary["offset"])]
        if before:
            title = max(before, key=lambda h: h["size"])["title"]
        headings = [h for h in headings if (h["page"], h["offset"]) >= (summary["page"], summary["offset"])]

    title = title or running_title
    if not headings:
        return [], title
    sized = sorted({heading["size"] for heading in headings if heading["font_based"]}, reverse=True)
    level_of = {size: min(index + 1, 4) for index, size in enumerate(sized)}
    bold_level = min(len(sized) + 1, 4)
    entries = [
        {
            "title": heading["title"],
            "level": level_of[heading["size"]] if heading["font_based"] else bold_level,
            "page": heading["page"],
            "offset": heading["offset"],
            "size": heading["size"],
            "font_based": heading["font_based"],
        }
        for heading in headings
    ]
    if not title:
        top = [entry for entry in entries if entry["level"] == 1]
        early = [entry for entry in top if entry["page"] <= 3]
        same_title = len(early) >= 1 and len({entry["title"].casefold() for entry in early}) == 1
        if top and same_title and len(early) == len(top) and len(entries) > len(early) + 1:
            title = early[0]["title"]
            entries = [entry for entry in entries if entry not in early]
    # Bold-only candidates are noisy; drop them when they dominate the page count.
    if sum(1 for entry in entries if not entry["font_based"]) > max(10, page_count * 3):
        entries = [entry for entry in entries if entry["font_based"]]
    _levels_from_numbering(entries)
    for entry in entries:
        entry.pop("size", None)
        entry.pop("font_based", None)
    return _dedupe_outline(entries), title


def _levels_from_numbering(entries: list[dict]) -> None:
    """Numbered headings ("2", "2.1", "2.1.3") define their own depth; unnumbered ones follow their font size."""
    depths = [_numbering_depth(entry["title"]) for entry in entries]
    if sum(1 for depth in depths if depth) < max(2, len(entries) // 3):
        levels = sorted({entry["level"] for entry in entries})
        remap = {level: index + 1 for index, level in enumerate(levels)}
        for entry in entries:
            entry["level"] = remap[entry["level"]]
        return
    by_size: dict[float, Counter] = {}
    for entry, depth in zip(entries, depths):
        if depth:
            by_size.setdefault(entry["size"], Counter())[depth] += 1
    for entry, depth in zip(entries, depths):
        if depth:
            entry["level"] = depth
        elif entry["size"] in by_size:
            entry["level"] = by_size[entry["size"]].most_common(1)[0][0]
        else:
            entry["level"] = min(4, 1 + sum(1 for size in by_size if size > entry["size"]))


def _outline_from_text(pages) -> list[dict]:
    candidates = []
    for page, text in pages:
        offset = 0
        for line in text.split("\n"):
            title = line.strip()
            if title and len(title) <= 100 and not _TOC_LINE.search(title):
                if _TEXT_CHAPTER.match(title):
                    candidates.append({"title": title, "level": 1, "page": page, "offset": offset})
                else:
                    match = _TEXT_NUMBERED.match(title)
                    if match and len(title.split()) <= 12:
                        level = 1 + match.group(1).count(".")
                        candidates.append({"title": title, "level": level, "page": page, "offset": offset})
            offset += len(line) + 1
    # Table-of-contents pages repeat every title; keep the occurrence on the least crowded page.
    per_page = Counter(item["page"] for item in candidates)
    best = {}
    for item in candidates:
        key = item["title"].casefold()
        if key not in best or per_page[item["page"]] < per_page[best[key]["page"]]:
            best[key] = item
    kept = sorted(best.values(), key=lambda item: (item["page"], item["offset"]))
    return _dedupe_outline(kept) if len(kept) >= 2 else []


def _dedupe_outline(entries: list[dict]) -> list[dict]:
    entries = sorted(entries, key=lambda item: (item["page"], item["offset"]))
    result = []
    for entry in entries:
        if result and result[-1]["page"] == entry["page"] and result[-1]["title"].casefold() == entry["title"].casefold():
            continue
        result.append(entry)
    return result


# ------------------------------------------------------------------------- chunking
def chunk_document(parsed: ParsedDocument, chunk_size: int = 1000, overlap: int = 150) -> list[dict]:
    """Split pages into chunks that never cross a section boundary and remember their section."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")
    boundaries: dict[int, list[tuple[int, int]]] = {}
    for index, entry in enumerate(parsed.outline):
        boundaries.setdefault(entry["page"], []).append((entry["offset"], index))
    tables_by_page: dict[int, list[dict]] = {}
    for table in parsed.tables:
        tables_by_page.setdefault(table["page"], []).append(table)
    chunks: list[dict] = []
    section = -1
    for page, text in parsed.pages:
        cuts = sorted(boundaries.get(page, []))
        segments = []
        start = 0
        for offset, index in cuts:
            offset = min(max(offset, 0), len(text))
            if offset > start:
                segments.append((start, offset, section))
                start = offset
            section = index
        segments.append((start, len(text), section))
        page_tables = tables_by_page.get(page, [])
        for segment_start, segment_end, segment_section in segments:
            for chunk_start, chunk_end in _split_text(text, segment_start, segment_end, chunk_size, overlap):
                chunk = {
                    "id": f"p{page}-{len(chunks)}",
                    "text": text[chunk_start:chunk_end],
                    "page": page,
                    "section": segment_section,
                    "start": chunk_start,
                    "end": chunk_end,
                }
                # A chunk that starts inside a table keeps that table's header row (for embedding and context).
                table = next((t for t in page_tables if t["offset"] < chunk_start < t["end"]), None)
                if table and not chunk["text"].startswith(table["header"]):
                    chunk["table_header"] = table["header"]
                chunks.append(chunk)
    return chunks


def chunk_pages(pages: list[tuple[int, str]], chunk_size: int = 1000, overlap: int = 150) -> list[dict]:
    return chunk_document(ParsedDocument(pages=pages, page_count=len(pages)), chunk_size, overlap)


def _split_text(text: str, start: int, stop: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    spans = []
    while start < stop:
        end = min(stop, start + chunk_size)
        if end < stop:
            end = _preferred_boundary(text, start + chunk_size // 2, end, start)
        value_start, value_end = start, end
        while value_start < value_end and text[value_start].isspace():
            value_start += 1
        while value_end > value_start and text[value_end - 1].isspace():
            value_end -= 1
        if value_end > value_start:
            spans.append((value_start, value_end))
        if end >= stop:
            break
        next_start = max(end - overlap, start + 1)
        sentence = max(text.rfind(". ", next_start, end), text.rfind("\n", next_start, end))
        if sentence > next_start:
            next_start = sentence + 1
        else:
            while next_start < end and not text[next_start - 1].isspace():
                next_start += 1
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start
    return spans


def _preferred_boundary(text: str, search_start: int, end: int, start: int) -> int:
    paragraph = text.rfind("\n\n", search_start, end)
    if paragraph > start:
        return paragraph + 2
    sentence = max(text.rfind(". ", search_start, end), text.rfind("? ", search_start, end), text.rfind("! ", search_start, end))
    if sentence > start:
        return sentence + 2
    line = text.rfind("\n", search_start, end)
    if line > start:
        return line + 1
    whitespace = text.rfind(" ", search_start, end)
    return whitespace + 1 if whitespace > start else end
