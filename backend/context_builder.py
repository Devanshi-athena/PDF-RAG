"""Turns chunk positions into ordered, de-duplicated, page-tagged excerpts within a character budget."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Source


@dataclass
class Excerpt:
    positions: list[int]
    text: str
    pages: list[int]
    section_path: str
    page_texts: dict[int, str]


def expand_hits(entries: list[dict], hits: list[dict], budget: int, seeds: int, section_limit: int) -> list[int]:
    """Top hits pull in their whole section when it is small, otherwise their neighbouring chunks."""
    chosen: set[int] = set()
    used = 0
    by_section: dict[int, list[int]] = {}
    for entry in entries:
        by_section.setdefault(entry["section"], []).append(entry["pos"])

    def cost(positions):
        return sum(len(entries[p]["text"]) for p in positions if p not in chosen)

    for rank, hit in enumerate(hits[:seeds]):
        pos = hit["pos"]
        section = entries[pos]["section"]
        whole = by_section.get(section, []) if section >= 0 else []
        options = []
        if rank < 2 and whole and cost(whole) <= section_limit:
            options.append(whole)
        window = [p for p in (pos - 1, pos, pos + 1) if 0 <= p < len(entries) and entries[p]["section"] == section]
        options.extend([window, [pos]])
        for option in options:
            extra = cost(option)
            if used + extra <= budget:
                chosen.update(option)
                used += extra
                break
        if used >= budget:
            break
    return sorted(chosen)


def budgeted(entries: list[dict], positions: list[int], budget: int) -> list[int]:
    """Keep every sub-section represented: take chunks round-robin across sections in reading order."""
    positions = sorted(set(positions))
    if sum(len(entries[p]["text"]) for p in positions) <= budget:
        return positions
    groups: dict[int, list[int]] = {}
    for pos in positions:
        groups.setdefault(entries[pos]["section"], []).append(pos)
    queues = list(groups.values())
    chosen, used, depth = set(), 0, 0
    while used < budget and any(depth < len(queue) for queue in queues):
        for queue in queues:
            if depth < len(queue):
                length = len(entries[queue[depth]]["text"])
                if used + length <= budget:
                    chosen.add(queue[depth])
                    used += length
        depth += 1
    return sorted(chosen)


def excerpts(entries: list[dict], positions: list[int]) -> list[Excerpt]:
    runs: list[list[int]] = []
    for pos in sorted(set(positions)):
        if runs and pos == runs[-1][-1] + 1 and entries[pos]["section"] == entries[runs[-1][-1]]["section"]:
            runs[-1].append(pos)
        else:
            runs.append([pos])
    result = []
    for run in runs:
        page_texts: dict[int, str] = {}
        previous = None
        for pos in run:
            entry = entries[pos]
            text = entry["text"]
            if previous and previous["page"] == entry["page"] and entry["start"] < previous["end"]:
                text = text[max(0, previous["end"] - entry["start"]):].lstrip()
            elif entry.get("table_header") and (previous is None or previous["page"] != entry["page"]):
                text = f"{entry['table_header']}\n{text}"  # the excerpt starts mid-table: restore its header row
            if text:
                existing = page_texts.get(entry["page"])
                page_texts[entry["page"]] = f"{existing} {text}" if existing else text
            previous = entry
        pages = list(page_texts)
        body = "\n".join(f"[Page {page}]\n{content}" for page, content in page_texts.items())
        result.append(Excerpt(run, body, pages, entries[run[0]]["section_path"], page_texts))
    return result


def interleave(hit_lists: list[list[dict]], per_list: int) -> list[dict]:
    """Round-robin the top hits of several sub-queries so every part of the question gets evidence."""
    merged, seen = [], set()
    for depth in range(per_list):
        for hits in hit_lists:
            if depth < len(hits) and hits[depth]["pos"] not in seen:
                seen.add(hits[depth]["pos"])
                merged.append(hits[depth])
    return merged


def cited_pages(items: list[Excerpt]) -> list[int]:
    return sorted({page for item in items for page in item.pages})


def best_sentence(text: str, terms: set[str], limit: int = 240) -> str:
    """The sentence of a chunk that mentions most query terms, for "which section discusses X" answers."""
    from .vector_store import tokenize

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]
    if not sentences:
        return text[:limit]

    def score(sentence):
        footnote = bool(re.match(r"^\d{1,3}\s", sentence) or "http" in sentence or re.search(r"\b(?:19|20)\d\d\.$", sentence))
        return (not footnote, len(terms & set(tokenize(sentence))), -abs(len(sentence) - 160))

    best = max(sentences, key=score)
    return best if len(best) <= limit else best[:limit].rsplit(" ", 1)[0] + "…"


def by_relevance(items: list[Excerpt], hits: list[dict]) -> list[Excerpt]:
    rank = {hit["pos"]: index for index, hit in enumerate(hits)}
    return sorted(items, key=lambda item: min((rank.get(p, len(rank)) for p in item.positions), default=len(rank)))


def render(items: list[Excerpt]) -> str:
    blocks = []
    for item in items:
        where = item.section_path or "Document text"
        blocks.append(f"--- From: {where}\n{item.text}")
    return "\n\n".join(blocks)


def sources_for(items: list[Excerpt], pdf_name: str) -> list[Source]:
    sources, seen = [], set()
    for item in items:
        for page, text in item.page_texts.items():
            if page in seen:
                continue
            seen.add(page)
            sources.append(Source(page=page, text=text[:1200], pdf_name=pdf_name))
    return sources
