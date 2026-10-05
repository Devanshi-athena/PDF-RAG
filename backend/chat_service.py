from __future__ import annotations

import re
import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from .context_builder import (
    best_sentence, budgeted, by_relevance, cited_pages, excerpts, expand_hits, interleave, render, sources_for,
)
from .document_structure import (
    descendants, pages_text, plural_noun, section_path, unit_heading, unit_label, unit_noun_for,
)
from .errors import BackendError
from .ingestion import ingest_pdf
from .models import Source, Thread
from .query_router import Route, route_query
from .vector_store import ThreadVectorStore, query_terms, tokenize

NOT_FOUND_ANSWER = "I couldn't find that in the uploaded PDF."
_DIGEST_VERSION = "v2"
_CITATION = re.compile(r"\[(?:Page|Pages|p\.|pp\.)\s*([0-9][0-9,\s–\-and]*)\]", re.IGNORECASE)

SYSTEM_PROMPT = f"""You are a document assistant. You answer strictly from the PDF text supplied in the user's message.

Grounding rules (never break these):
- Use only information stated in the supplied PDF text. Do not use outside knowledge, do not guess, and do not fill gaps with assumptions or general explanations.
- If the supplied text does not contain the answer, reply exactly: "{NOT_FOUND_ANSWER}"
- If it answers only part of the question, answer that part and say plainly which part the PDF does not cover. Do not add notes about things the question did not ask for.
- Cite pages for every factual statement using the [Page N] markers from the text, e.g. [Page 12]. Cite only pages listed as available and only where the page supports the statement.
- Earlier conversation is only for understanding follow-up references. It is never evidence.
- If the question is ambiguous or the term is used in several places with different meanings, briefly cover each relevant reading and name the section it comes from.

Answer quality:
- Answer exactly what was asked. Include every detail from the text that the question needs (definitions, requirements, conditions, steps, figures, names, exceptions), but leave out background, side topics and passages that are only loosely related.
- When the question covers several parts or subtopics, cover each of them once, in order.
- Never repeat a sentence, point or heading, and do not add a closing summary that restates the answer.
- Keep the PDF's exact numbers, names, defined terms and obligation words (must, shall, should, may).
- Table rows appear as cells separated by " | ", with the header row first; read values across the row and keep units.
- For a largest/smallest/total/average/count question over a table, check every row of the relevant table, give the result, and show the values you compared or added.
- Format as plain text: short paragraphs, "- " bullet points, and short labelled lines for sub-parts. Do not use markdown headings (#) or bold (**).
- Use a table when the answer compares items or gives several items that share the same attributes (for example levels and their responsibilities, standards and their purpose, values per year). Write it as a markdown table: a header row, a separator row such as | --- | --- |, then one row per item or aspect. Keep cells short, put [Page N] citations inside the cells, and write "Not stated in the PDF" for missing information. Never use a table for a single fact or for an explanation."""

_MODE_INSTRUCTIONS = {
    "multi": "The question has several parts: {parts}. Answer each part once, under its own short label, in order - or, when the parts are parallel items described by the same attributes, as one table with a row per part. If the PDF does not cover a part, say so for that part.",
    "compare": (
        "Compare {entities}{aspect}. Answer with a markdown comparison table: the first column is \"Aspect\" and there is "
        "one column per item; use one row per aspect the PDF describes (for example purpose, role, what it contains, who "
        "uses or manages it, how it works). Put [Page N] citations in the cells and write \"Not stated in the PDF\" where "
        "the PDF says nothing. Choose aspects the text covers for at least one item. After the table, add at most three "
        "\"- \" bullets on the key differences, and a similarity only if the PDF states or directly shows it. Every cell "
        "and bullet must follow from cited text - do not invent common features, do not rank or judge the items, and do "
        "not end with your own conclusion. Passages that mention the items together are the strongest evidence of how "
        "they relate."
    ),
    "analysis": "This is an analytical question. Build the answer by connecting statements from different excerpts. Every claim must rest on cited text; when you draw a conclusion the PDF does not state directly, mark it as an inference (for example: \"Taken together, [Page 4] and [Page 9] indicate ...\") and keep it close to the text. Do not bring in outside knowledge.",
    "enumerate": "List every item the excerpts give that directly answers the question, completely and in the PDF's order; do not stop after the first few, and leave out passages that are only loosely related. Prefer a list the PDF itself states over assembling one from scattered mentions. Only if the excerpts visibly cut a list off, say that the list may continue.",
    "ambiguous": "The request is short and may refer to several things. Explain what the PDF says about it in each distinct context where it appears, naming the section for each.",
}
# How long an answer should be: chosen from the question's wording ("briefly", "in detail", "in 5 points").
_LENGTH_INSTRUCTIONS = {
    "brief": "Length: brief - at most about 120 words, or exactly the number of points or sentences the question asks for.",
    "normal": "Length: only as long as the question needs - usually 60-250 words; a single fact needs one or two sentences. No introduction.",
    "detailed": "Length: a complete, detailed answer covering every relevant point in the text, organised under short labels.",
}
_LENGTH_TOKENS = {"brief": 0.4, "normal": 0.7, "detailed": 1.5}  # x MAX_ANSWER_TOKENS


class ChatService:
    def __init__(self, settings, vector_store_factory=ThreadVectorStore, client=None):
        self.settings = settings
        self.vector_store_factory = vector_store_factory
        self.client = client
        self._client_initialized = client is not None
        self._clients: dict[str, object] = {}
        self._stores = OrderedDict()
        self._stores_lock = threading.Lock()
        self._store_cache_limit = 8
        self._reindex_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------ plumbing
    def store(self, thread_id: str):
        with self._stores_lock:
            store = self._stores.get(thread_id)
            if store is None:
                store = self.vector_store_factory(
                    self.settings.chroma_dir,
                    thread_id,
                    self.settings.embedding_model,
                    token=self.settings.hf_token,
                    embedding_batch_size=self.settings.embedding_batch_size,
                    embedding_workers=self.settings.embedding_workers,
                    structure_dir=self.settings.structure_dir,
                    embedding_backend=getattr(self.settings, "embedding_backend", "hf"),
                    reranker_model=getattr(self.settings, "reranker_model", ""),
                    embedding_provider=getattr(self.settings, "embedding_provider", ""),
                )
                self._stores[thread_id] = store
                if len(self._stores) > self._store_cache_limit:
                    self._stores.popitem(last=False)
            else:
                self._stores.move_to_end(thread_id)
            return store

    def discard_store(self, thread_id: str) -> None:
        with self._stores_lock:
            self._stores.pop(thread_id, None)

    def _provider_for(self, model: str | None) -> str:
        """Each model can be pinned to its own inference provider (HF_PROVIDER / HF_ANALYSIS_PROVIDER)."""
        analysis_model = getattr(self.settings, "analysis_model", "")
        analysis_provider = getattr(self.settings, "analysis_provider", "")
        if model and analysis_model and model == analysis_model and analysis_provider:
            return analysis_provider
        return self.settings.hf_provider

    def _hf_client(self, model: str | None = None):
        if self._client_initialized:
            return self.client  # injected client (tests)
        if not self.settings.hf_token:
            raise BackendError("HF_TOKEN is required for chat. Add it to .env.", 503, "llm_not_configured")
        provider = self._provider_for(model)
        with self._stores_lock:
            client = self._clients.get(provider)
            if client is None:
                try:
                    from huggingface_hub import InferenceClient
                except ImportError as error:
                    raise BackendError("huggingface-hub is required for chat.", 503, "llm_unavailable") from error
                kwargs = {"token": self.settings.hf_token}
                if provider:
                    kwargs["provider"] = provider
                client = InferenceClient(**kwargs)
                self._clients[provider] = client
            return client

    def _ready_store(self, thread: Thread):
        """Return the thread's store, re-indexing indexes built by older versions from the saved PDF."""
        store = self.store(thread.id)
        needs_reindex = getattr(store, "needs_reindex", None)
        if needs_reindex and needs_reindex():
            path = self.settings.documents_dir / f"{thread.id}.pdf"
            if path.exists():
                with self._stores_lock:
                    lock = self._reindex_locks.setdefault(thread.id, threading.Lock())
                with lock:
                    if store.needs_reindex():
                        started = time.perf_counter()
                        ingest_pdf(path.read_bytes(), store, self.settings.chunk_size, self.settings.chunk_overlap)
                        print(f"[PERF] legacy_reindex_seconds={time.perf_counter() - started:.4f}")
        return store

    # --------------------------------------------------------------- public API
    def ask(self, thread: Thread, question: str) -> tuple[str, list[Source]]:
        tokens, sources = self.stream(thread, question)
        answer = "".join(tokens).strip() or NOT_FOUND_ANSWER
        return answer, self.finalize_sources(answer, sources)

    def stream(self, thread: Thread, question: str) -> tuple[Iterator[str], list[Source]]:
        started = time.perf_counter()
        store = self._ready_store(thread)
        structure = store.structure() or {"sections": [], "units": [], "page_count": 0, "captions": []}
        structure.setdefault("captions", [])
        previous_question = _previous_question(thread, question)
        route = route_query(question, structure, previous_question, document_name=thread.pdf_name)
        print(
            f"[PERF] route intent={route.intent} target={route.target} mode={route.mode} "
            f"units={len(route.units)} pages={route.pages[:5]} subqueries={len(route.subqueries)}"
        )
        handler = getattr(self, f"_handle_{route.intent}")
        tokens, sources = handler(thread, question, route, store, structure)
        print(f"[PERF] planning_seconds={time.perf_counter() - started:.4f}")
        return tokens, sources

    @staticmethod
    def finalize_sources(answer: str, sources: list[Source]) -> list[Source]:
        if answer.strip() == NOT_FOUND_ANSWER:
            return []
        cited = set()
        for match in _CITATION.finditer(answer):
            for start, end in re.findall(r"(\d+)(?:\s*[–-]\s*(\d+))?", match.group(1)):
                first = int(start)
                last = int(end) if end else first
                cited.update(range(first, min(last, first + 50) + 1))
        if not cited:
            return sources
        kept = [source for source in sources if source.page in cited]
        return kept or sources

    # ------------------------------------------------------ structural answers
    def _handle_count(self, thread, question, route: Route, store, structure):
        name = _pdf_name(thread)
        sections = structure["sections"]
        if route.target == "pages":
            count = structure.get("page_count") or (thread.document.pages if thread.document else 0)
            return _direct(f"The document has {count} pages." if count else NOT_FOUND_ANSWER)
        if route.target == "captions":
            captions = [c for c in structure["captions"] if not route.caption_kind or c["kind"] == route.caption_kind]
            kind = route.caption_kind or "figure or table"
            if not captions and route.caption_kind == "table" and structure.get("tables"):
                return _direct(*_layout_tables_answer(structure, name))
            if not captions:
                return _direct(f"I couldn't find any captioned {plural_noun(kind)} in the uploaded PDF.")
            answer = f"The document has {len(captions)} captioned {_plural(kind, len(captions))}:\n\n" + "\n".join(
                f"- {c['text']} [Page {c['page']}]" for c in captions
            )
            return _direct(answer, [Source(page=c["page"], text=c["text"], pdf_name=name) for c in captions])
        if route.target == "children":
            return _direct(*self._children_answer(route, structure, name, count_only=True))
        if route.target == "headings":
            if not sections:
                return _direct("I couldn't detect any headings in the uploaded PDF.")
            levels = Counter(section["level"] for section in sections)
            answer = f"The document has {len(sections)} headings."
            if len(levels) > 1:
                answer += " By level: " + ", ".join(
                    f"{count} at level {level}" for level, count in sorted(levels.items())
                ) + "."
            return _direct(answer, [_section_source(s, name) for s in sections])
        labelled = _labelled_sections(structure, route.noun)
        if labelled and not route.field_filter:
            noun = route.noun
            lines = [f"{n}. {_section_line(structure, sections[i], noun)}" for n, i in enumerate(labelled, start=1)]
            answer = f"The document has {len(labelled)} {_plural(noun, len(labelled))}:\n\n" + "\n".join(lines)
            return _direct(answer, [_section_source(sections[i], name) for i in labelled])
        if not structure["units"]:
            noun = plural_noun(route.noun or "section")
            return _direct(f"I couldn't detect any {noun} in the uploaded PDF's layout.")
        units = self._filtered_units(route, structure)
        requested = route.noun
        noun = _display_noun(structure, requested)
        lines = [_section_line(structure, sections[i], requested) for i in units]
        if route.field_filter:
            key, value = route.field_filter
            answer = f"{len(units)} {_plural(noun, len(units))} in the uploaded PDF have {key}: {value}."
            if units:
                answer += "\n\n" + "\n".join(f"{n}. {line}" for n, line in enumerate(lines, start=1))
        else:
            # Annexes/appendices carry their own numbering and are not chapters.
            same_kind = lambda i: sections[i].get("noun") in (None, requested, structure.get("unit_noun"))
            numbered = [i for i in units if sections[i].get("number") is not None and same_kind(i)]
            others = [i for i in units if i not in numbered]
            if len(numbered) >= 2 and others:
                # Mixed top level (Abstract, 1..3, Glossary): count the numbered ones, name the rest.
                answer = f"The document has {len(numbered)} numbered {_plural(noun, len(numbered))}."
                answer += _numbering_note(structure, numbered)
                answer += (
                    f" It also has {len(others)} other top-level section{'s' if len(others) != 1 else ''}: "
                    + ", ".join(sections[i]["title"] for i in others) + "."
                )
                answer += "\n\n" + "\n".join(
                    f"{n}. {_section_line(structure, sections[i], requested)}" for n, i in enumerate(numbered, start=1)
                )
                return _direct(answer, [_section_source(sections[i], name) for i in units])
            answer = f"The document contains {len(units)} {_plural(noun, len(units))}."
            answer += _numbering_note(structure, units)
            answer += _noun_note(structure, requested)
            if len(units) <= 40:
                answer += "\n\n" + "\n".join(f"{n}. {line}" for n, line in enumerate(lines, start=1))
        return _direct(answer, [_section_source(sections[i], name) for i in units])

    def _handle_list(self, thread, question, route: Route, store, structure):
        name = _pdf_name(thread)
        sections = structure["sections"]
        if route.target == "captions":
            captions = [c for c in structure["captions"] if not route.caption_kind or c["kind"] == route.caption_kind]
            if not captions and route.caption_kind == "table" and structure.get("tables"):
                return _direct(*_layout_tables_answer(structure, name))
            if not captions:
                return _direct(f"I couldn't find any captioned {plural_noun(route.caption_kind or 'figure')} in the uploaded PDF.")
            answer = f"{plural_noun(route.caption_kind or 'caption').title()} in the uploaded PDF ({len(captions)}):\n\n" + "\n".join(
                f"- {c['text']} [Page {c['page']}]" for c in captions
            )
            return _direct(answer, [Source(page=c["page"], text=c["text"], pdf_name=name) for c in captions])
        if route.target == "children":
            return _direct(*self._children_answer(route, structure, name, count_only=False))
        if route.target in ("headings", "outline") or not structure["units"]:
            if not sections:
                return _direct("I couldn't detect any headings or sections in the uploaded PDF.")
            top = min(section["level"] for section in sections)
            shown = [s for s in sections if s["level"] - top < 3][:400]
            lines = [f"{'  ' * (s['level'] - top)}- {s['title']} [{pages_text(s)}]" for s in shown]
            title = structure.get("title")
            heading = f"Outline of {title}" if title and route.target == "outline" else "Headings in the uploaded PDF"
            answer = f"{heading} ({len(sections)} headings):\n\n" + "\n".join(lines)
            if len(shown) < len(sections):
                answer += f"\n\n(Showing the top three levels; {len(sections) - len(shown)} deeper headings omitted.)"
            return _direct(answer, [_section_source(s, name) for s in shown])
        labelled = _labelled_sections(structure, route.noun)
        if labelled and not route.field_filter and not route.number_range:
            lines = [f"{n}. {_section_line(structure, sections[i], route.noun)}" for n, i in enumerate(labelled, start=1)]
            answer = f"{plural_noun(route.noun).title()} in the uploaded PDF ({len(labelled)}):\n\n" + "\n".join(lines)
            return _direct(answer, [_section_source(sections[i], name) for i in labelled])
        units = self._range_units(route, structure)
        requested = route.noun
        noun = _display_noun(structure, requested)
        if not units:
            return _direct(_missing_answer(structure, route, noun))
        if not route.number_range and not route.field_filter:
            # Mixed top level (Executive summary, chapters 1-5, annexures): list the numbered chapters, name the rest.
            same_kind = lambda i: sections[i].get("noun") in (None, requested, structure.get("unit_noun"))
            numbered = [i for i in units if sections[i].get("number") is not None and same_kind(i)]
            others = [i for i in units if i not in numbered]
            if len(numbered) >= 2 and others:
                lines = [f"{n}. {_section_line(structure, sections[i], requested)}" for n, i in enumerate(numbered, start=1)]
                answer = f"The document has {len(numbered)} numbered {_plural(noun, len(numbered))}:\n\n" + "\n".join(lines)
                answer += "\n\nOther top-level parts:\n" + "\n".join(
                    f"- {unit_heading(structure, sections[i], None)} [{pages_text(sections[i])}]" for i in others
                )
                return _direct(answer, [_section_source(sections[i], name) for i in numbered + others])
        heading = f"{plural_noun(noun).title()} in the uploaded PDF"
        if route.field_filter:
            heading += f" with {route.field_filter[0]}: {route.field_filter[1]}"
        if route.number_range:
            heading += f" ({route.number_range[0]}–{route.number_range[1]})"
        lines = [f"{n}. {_section_line(structure, sections[i], requested)}" for n, i in enumerate(units, start=1)]
        answer = f"{heading} — {len(units)} found:\n\n" + "\n".join(lines)
        if route.number_range and route.missing_numbers:
            answer += f"\n\nNot found in the document: {noun} " + ", ".join(map(str, route.missing_numbers)) + "."
        return _direct(answer, [_section_source(sections[i], name) for i in units])

    def _range_units(self, route: Route, structure: dict) -> list[int]:
        if route.number_range and route.units:
            return route.units
        if route.number_range:
            noun_sections = _labelled_sections(structure, route.noun)
            pool = noun_sections or structure["units"]
            start, end = route.number_range
            return [i for i in pool if start <= (structure["sections"][i].get("number") or -1) <= end]
        return self._filtered_units(route, structure)

    def _children_answer(self, route: Route, structure: dict, name: str, count_only: bool):
        sections = structure["sections"]
        blocks, sources = [], []
        for index in route.units:
            section = sections[index]
            children = _major_children(structure, index)
            heading = f"{unit_heading(structure, section, route.noun)} ({pages_text(section)})"
            if not children:
                blocks.append(f"{heading} has no sub-sections in the document's outline.")
                continue
            lines = "\n".join(f"- {sections[c]['title']} [{pages_text(sections[c])}]" for c in children)
            block = f"{heading} has {len(children)} sub-section{'s' if len(children) != 1 else ''}:\n{lines}"
            minor = [c for c in section.get("children", []) if c not in children]
            if minor:
                block += "\nOther headings within it: " + "; ".join(sections[c]["title"] for c in minor) + "."
            blocks.append(block)
            sources.extend(_section_source(sections[c], name) for c in children)
        return "\n\n".join(blocks), sources

    def _handle_lookup(self, thread, question, route: Route, store, structure):
        sections = structure["sections"]
        requested = route.noun
        lines = []
        for index in route.units:
            section = sections[index]
            label = unit_label(structure, section, requested)
            fields = "".join(f"; {key}: {value}" for key, value in section.get("fields", {}).items())
            if label == section["title"]:
                lines.append(f"\"{section['title']}\"{fields} [{pages_text(section)}].")
            else:
                lines.append(f"{label} is \"{section['name']}\"{fields} [{pages_text(section)}].")
        if route.missing_numbers:
            lines.append(_missing_answer(structure, route, _display_noun(structure, requested)))
        return _direct("\n".join(lines), [_section_source(sections[i], _pdf_name(thread)) for i in route.units])

    def _handle_locate(self, thread, question, route: Route, store, structure):
        """Which sections discuss X: grouped search hits with a quoted sentence each - no generation needed."""
        entries = store.entries()
        hits = store.search(route.search_query, 12)
        if not self._has_evidence(hits):
            return _direct(NOT_FOUND_ANSWER)
        terms = set(tokenize(" ".join(query_terms(route.search_query))))
        threshold = self.settings.retrieval_min_similarity + 0.05
        sections = structure["sections"]
        groups: dict[int, list[dict]] = {}
        for hit in hits:
            section_index = entries[hit["pos"]]["section"]
            matter = 0 <= section_index < len(sections) and sections[section_index].get("matter")
            strong = hit["coverage"] >= 0.5 or (hit["similarity"] or 0) >= threshold
            if strong and not matter:
                groups.setdefault(section_index, []).append(hit)
        if not groups:
            return _direct(NOT_FOUND_ANSWER)

        def weight(item):
            # Total evidence in the section, plus a bonus when its own title names the topic.
            section_index, section_hits = item
            title_terms = set(tokenize(sections[section_index]["title"])) if 0 <= section_index < len(sections) else set()
            title_bonus = 0.05 * len(terms & title_terms)
            return sum(hit["score"] for hit in section_hits) + title_bonus

        ranked = sorted(groups.items(), key=weight, reverse=True)
        lines, sources = [], []
        for number, (section_index, section_hits) in enumerate(ranked[:6], start=1):
            best = entries[section_hits[0]["pos"]]
            pages = sorted({entries[h["pos"]]["page"] for h in section_hits})
            where = section_path(structure, section_index) if 0 <= section_index < len(sections) else "Text before the first heading"
            page_label = f"Page {pages[0]}" if len(pages) == 1 else "Pages " + ", ".join(map(str, pages))
            quote = best_sentence(best["text"], terms)
            lines.append(f"{number}. {where} [{page_label}]\n   \"{quote}\" [Page {best['page']}]")
            sources.append(Source(page=best["page"], text=best["text"][:1200], pdf_name=_pdf_name(thread)))
        answer = "The topic is discussed in these parts of the document (most relevant first):\n\n" + "\n".join(lines)
        return _direct(answer, sources)

    def _filtered_units(self, route: Route, structure: dict) -> list[int]:
        sections = structure["sections"]
        units = list(structure["units"])
        if route.number_range:
            start, end = route.number_range
            units = [i for i in units if start <= (sections[i].get("number") or -1) <= end]
        if route.field_filter:
            key, value = route.field_filter
            units = [i for i in units if (sections[i]["fields"].get(key) or "").casefold() == value.casefold()]
        return units

    # ------------------------------------------------------ generated answers
    def _handle_explain(self, thread, question, route: Route, store, structure):
        sections = structure["sections"]
        noun = _display_noun(structure, route.noun)
        if not route.units:
            return _direct(_missing_answer(structure, route, noun))
        if len(route.units) > 1:
            return self._explain_many(thread, question, route, store, structure)
        section = sections[route.units[0]]
        positions = _section_positions(store, structure, section["index"])
        if not positions:
            return _direct(NOT_FOUND_ANSWER)
        entries = store.entries()
        heading = f"{unit_heading(structure, section, route.noun)} ({pages_text(section)})"
        budget = self.settings.explain_context_chars
        total = sum(len(entries[p]["text"]) for p in positions)
        noun_word = unit_noun_for(structure, section, route.noun)
        if route.detail == "detailed" and total > budget * 1.2 and len(section.get("children", [])) >= 2:
            return self._explain_hierarchical(thread, question, route, store, structure, section, heading)
        # The sub-section list comes from the outline (exact and complete); the model only writes the explanation.
        major = _major_children(structure, section["index"])
        outline = ""
        if len(major) >= 2:
            outline = "Sub-sections:\n" + "\n".join(
                f"- {sections[c]['title']} [{pages_text(sections[c])}]" for c in major
            ) + "\n\n"
        if route.detail == "detailed":
            task = (
                f"Explain this {noun_word} completely: its main idea, then every part or subtopic in the order it appears, "
                "with the key details, requirements, examples, facts and conclusions."
            )
            items = excerpts(entries, budgeted(entries, positions, budget))
        else:
            task = (
                f"Explain what this {noun_word} is about: its purpose and its main points"
                + (", touching on its sub-sections" if outline else "")
                + ". Do not list the sub-section titles again; they are shown separately."
            )
            items = excerpts(entries, budgeted(entries, positions, int(budget * 0.6)))
        content = (
            f"PDF section: {heading}\n\n{render(items)}\n\n"
            f"Pages available for citation: {_page_list(items)}\n\n"
            f"Request: {question}\n\n"
            f"Using only the section text above: {task} Cite pages. Do not repeat the section heading.\n"
            f"{_LENGTH_INSTRUCTIONS[route.detail]}"
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        tokens = self._generate(messages, self._answer_tokens(route.detail), prefix=f"{heading}\n\n{outline}")
        return tokens, sources_for(items, _pdf_name(thread))

    def _explain_hierarchical(self, thread, question, route, store, structure, section, heading):
        """A section too long for one prompt: summarize each sub-section in parallel (cached) and stream the summaries
        in reading order as soon as each is ready - no extra combining call, so text appears after the first summary."""
        sections = structure["sections"]
        entries = store.entries()
        children = section["children"][: self.settings.max_units_per_request]
        plans = []
        for child in children:
            positions = _section_positions(store, structure, child)
            plans.append((sections[child], excerpts(entries, budgeted(entries, positions, self.settings.range_unit_context_chars))))
        sources = []
        for _, items in plans:
            sources.extend(sources_for(items, _pdf_name(thread))[:2])

        def generate() -> Iterator[str]:
            started = time.perf_counter()
            yield f"{heading}\n\nIt has {len(children)} sub-section{'s' if len(children) != 1 else ''}:\n\n"
            executor = ThreadPoolExecutor(max_workers=max(1, self.settings.llm_workers))
            try:
                futures = [executor.submit(self._digest_safe, store, structure, child, items, None) for child, items in plans]
                for (child, _), future in zip(plans, futures):
                    yield f"{child['title']} ({pages_text(child)})\n"
                    yield future.result().strip() + "\n\n"
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            if len(section["children"]) > len(children):
                yield f"Showing the first {len(children)} of {len(section['children'])} sub-sections.\n"
            print(f"[PERF] hierarchical_explain_seconds={time.perf_counter() - started:.4f} subsections={len(children)}")

        return generate(), sources

    def _explain_many(self, thread, question, route: Route, store, structure):
        sections = structure["sections"]
        requested = list(route.units)
        units = requested[: self.settings.max_units_per_request]
        entries = store.entries()
        name = _pdf_name(thread)
        plans = []
        for index in units:
            positions = _section_positions(store, structure, index)
            items = excerpts(entries, budgeted(entries, positions, self.settings.range_unit_context_chars))
            plans.append((sections[index], items))
        sources = []
        for _, items in plans:
            sources.extend(sources_for(items, name)[:2])

        def generate() -> Iterator[str]:
            started = time.perf_counter()
            noun = _display_noun(structure, route.noun)
            yield f"{len(units)} {_plural(noun, len(units))}:\n\n"
            executor = ThreadPoolExecutor(max_workers=max(1, self.settings.llm_workers))
            try:
                futures = [executor.submit(self._digest, store, structure, section, items, route.noun) for section, items in plans]
                failures = 0
                for (section, _), future in zip(plans, futures):
                    yield f"{unit_heading(structure, section, route.noun)} ({pages_text(section)})\n"
                    try:
                        yield future.result().strip() + "\n\n"
                    except BackendError as error:
                        failures += 1
                        if failures == len(plans) or error.code == "llm_quota":
                            raise
                        yield f"(This summary could not be generated: {error.message})\n\n"
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            if len(requested) > len(units):
                yield (
                    f"Showing the first {len(units)} of {len(requested)} requested. "
                    "Ask for the remaining ones in a follow-up question.\n"
                )
            if route.missing_numbers:
                yield f"Not found in the document: {noun} " + ", ".join(map(str, route.missing_numbers)) + ".\n"
            print(f"[PERF] multi_unit_generation_seconds={time.perf_counter() - started:.4f} units={len(units)}")

        return generate(), sources

    def _digest_safe(self, store, structure, section, items, requested) -> str:
        try:
            return self._digest(store, structure, section, items, requested)
        except BackendError as error:
            if error.code == "llm_quota":
                raise
            return f"(Summary unavailable: {error.message})"

    def _digest(self, store, structure, section, items, requested) -> str:
        key = f"digest:{_DIGEST_VERSION}:{section['index']}"
        cached = store.digest(key) if hasattr(store, "digest") else None
        if cached:
            return cached
        if not items:
            return NOT_FOUND_ANSWER
        noun = unit_noun_for(structure, section, requested)
        content = (
            f"PDF section: {unit_heading(structure, section, requested)} ({pages_text(section)})\n\n{render(items)}\n\n"
            f"Pages available for citation: {_page_list(items)}\n\n"
            f"Summarize this {noun} using only the text above: one or two sentences on its main idea, then 3-7 "
            "bullet points covering all of its main points in order. Cite pages."
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        text = self._complete(messages, 450).strip()
        if text and text != NOT_FOUND_ANSWER and hasattr(store, "save_digest"):
            store.save_digest(key, text)
        return text or NOT_FOUND_ANSWER

    def _handle_compare(self, thread, question, route: Route, store, structure):
        sections = structure["sections"]
        units = route.units[:6]
        entries = store.entries()
        share = max(2000, self.settings.explain_context_chars // len(units))
        blocks, all_items = [], []
        for index in units:
            items = excerpts(entries, budgeted(entries, _section_positions(store, structure, index), share))
            all_items.extend(items)
            blocks.append(f"=== {unit_heading(structure, sections[index], route.noun)} ({pages_text(sections[index])})\n{render(items)}")
        content = (
            "PDF sections:\n\n" + "\n\n".join(blocks)
            + f"\n\nPages available for citation: {_page_list(all_items)}\n\nQuestion: {question}\n\n"
            + _MODE_INSTRUCTIONS["compare"].format(entities="these sections", aspect="") + " Cite pages.\n"
            + _LENGTH_INSTRUCTIONS[route.detail]
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_history(thread, question), {"role": "user", "content": content}]
        tokens = self._generate(messages, self._answer_tokens(route.detail, wide=True), model=self._model_for("compare"))
        return tokens, sources_for(all_items, _pdf_name(thread))

    def _handle_scoped_qa(self, thread, question, route: Route, store, structure):
        entries = store.entries()
        budget = self.settings.analysis_context_chars if route.mode != "direct" else self.settings.qa_context_chars
        scope = sorted({p for index in route.units for p in _section_positions(store, structure, index)})
        if not scope:
            return _direct(NOT_FOUND_ANSWER)
        if sum(len(entries[p]["text"]) for p in scope) <= budget:
            positions = scope
        else:
            queries = [route.search_query, *route.subqueries]
            hit_lists = _search_many(store, queries, self.settings.top_k, positions=scope)
            hits = interleave(hit_lists, self.settings.top_k)
            positions = expand_hits(entries, hits, budget, len(hits), int(budget * 0.6)) or budgeted(entries, scope, budget)
        # Safety net against fuzzy title matches and imperfect outlines: strong evidence from elsewhere is added too.
        remaining = budget - sum(len(entries[p]["text"]) for p in positions)
        reserve = remaining if not route.explicit else min(remaining, int(budget * 0.3))
        if reserve > 1500 and query_terms(route.search_query):
            taken = set(positions)
            scope_set = set(scope)
            hits = [
                hit for hit in store.search(route.search_query, self.settings.top_k)
                if hit["pos"] not in scope_set and hit["coverage"] >= 0.75
                and (hit["similarity"] is None or hit["similarity"] >= self.settings.retrieval_min_similarity + 0.1)
            ]
            positions = sorted(taken | set(expand_hits(entries, hits, reserve, 3, reserve)))
        return self._answer_from(
            thread, question, excerpts(entries, positions), route, note=_outline_note(structure, route.units)
        )

    def _handle_caption(self, thread, question, route: Route, store, structure):
        entries = store.entries()
        positions, described = set(), []
        for caption in route.captions[:4]:
            page_positions = store.positions_for_pages([caption["page"]])
            # Captions often sit under the table/figure; include the facing page when the caption ends a page.
            if page_positions and caption["text"][:60] in entries[page_positions[-1]]["text"]:
                page_positions += store.positions_for_pages([caption["page"] + 1])[:2]
            positions.update(budgeted(entries, page_positions, self.settings.qa_context_chars // max(1, len(route.captions))))
            described.append(f"{caption['text']} (page {caption['page']})")
        if not positions:
            return _direct(NOT_FOUND_ANSWER)
        note = (
            "The question refers to: " + "; ".join(described) + ". Tables are extracted as rows with cells separated by "
            "\" | \". Images themselves are not available - only captions and the text around them. If the answer "
            "depends on visual content that is not described in the text, say so."
        )
        return self._answer_from(thread, question, excerpts(entries, sorted(positions)), route, note=note)

    def _handle_page(self, thread, question, route: Route, store, structure):
        entries = store.entries()
        positions = store.positions_for_pages(route.pages)
        if not positions:
            return _direct(NOT_FOUND_ANSWER)
        positions = budgeted(entries, positions, self.settings.explain_context_chars)
        return self._answer_from(thread, question, excerpts(entries, positions), route)

    def _handle_qa(self, thread, question, route: Route, store, structure):
        started = time.perf_counter()
        entries = store.entries()
        # Multi-part, comparison and analysis questions need evidence from several places; lists and vague
        # questions need a bit more than a direct lookup, but too much loosely related text hurts precision.
        wide = route.mode in ("multi", "compare", "analysis")
        seeds = self.settings.top_k + (4 if wide else 2 if route.mode != "direct" else 0)
        if wide:
            budget = self.settings.analysis_context_chars
        elif route.mode != "direct":
            budget = int(self.settings.qa_context_chars * 1.25)
        else:
            budget = self.settings.qa_context_chars
        queries = [route.search_query, *route.subqueries]
        hit_lists = _search_many(store, queries, seeds)
        print(f"[PERF] retrieval_seconds={time.perf_counter() - started:.4f} queries={len(queries)}")
        role_positions = []
        if route.role_sections:
            role_scope = sorted({p for index in route.role_sections for p in _section_positions(store, structure, index)})
            role_positions = budgeted(entries, role_scope, int(budget * 0.4))
        evidence = [hits for hits in hit_lists if self._has_evidence(hits)]
        if not evidence and not role_positions:
            best = [(max((h["similarity"] or 0) for h in hits), max(h["coverage"] for h in hits)) for hits in hit_lists if hits]
            print(f"[PERF] abstained best={best}")
            return _direct(NOT_FOUND_ANSWER)

        if route.mode == "compare" and len(route.entities) >= 2:
            entity_hits = [store.search(query, seeds) for query in route.subqueries]
            return self._compare_entities(thread, question, route, store, entity_hits, budget)

        remaining = budget - sum(len(entries[p]["text"]) for p in role_positions)
        if len(hit_lists) > 1:
            hits = interleave(evidence or hit_lists, max(3, seeds // len(hit_lists) + 1))
        else:
            hits = hit_lists[0] if evidence else []
        positions = set(role_positions)
        positions |= set(expand_hits(entries, [h for h in hits if h["pos"] not in positions], max(0, remaining), len(hits), int(budget * 0.5)))
        items = by_relevance(excerpts(entries, sorted(positions)), hits) if hits else excerpts(entries, sorted(positions))
        return self._answer_from(thread, question, items, route)

    def _compare_entities(self, thread, question, route: Route, store, entity_hits, budget):
        """Comparison of two concepts: retrieve evidence for each side separately and show it grouped."""
        entries = store.entries()
        # Passages that mention the items together show how they relate - the best evidence for a comparison.
        entity_terms = [set(tokenize(" ".join(query_terms(entity)))) for entity in route.entities]

        def mentions(entry, terms):
            return terms and len(terms & entry["term_set"]) * 2 >= len(terms)

        candidates = {hit["pos"] for hits in entity_hits for hit in hits[:8]}
        joint = sorted(
            (pos for pos in candidates if all(mentions(entries[pos], terms) for terms in entity_terms)),
            key=lambda pos: min((rank for hits in entity_hits for rank, hit in enumerate(hits) if hit["pos"] == pos), default=99),
        )[:3]
        joint_items = excerpts(entries, joint)
        joint_chars = sum(len(entries[p]["text"]) for p in joint)
        share = max(1500, (budget - joint_chars) // max(1, len(route.entities)))
        # Each passage goes to the side whose search ranks it higher, so one side can't claim the other's evidence.
        owner: dict[int, tuple[int, int]] = {}
        for side, hits in enumerate(entity_hits):
            for rank, hit in enumerate(hits):
                if hit["pos"] not in owner or rank < owner[hit["pos"]][1]:
                    owner[hit["pos"]] = (side, rank)
        blocks, all_items, used = [], list(joint_items), set(joint)
        if joint_items:
            blocks.append(f"=== Passages mentioning {' and '.join(route.entities)} together\n{render(joint_items)}")
        for side, (entity, hits) in enumerate(zip(route.entities, entity_hits)):
            hits = [h for h in hits if h["pos"] not in used and owner[h["pos"]][0] == side]
            supported = self._has_evidence(hits)
            positions = expand_hits(entries, hits, share, self.settings.top_k, int(share * 0.6)) if supported else []
            used.update(positions)
            items = by_relevance(excerpts(entries, positions), hits) if positions else []
            all_items.extend(items)
            body = render(items) if items else "(No passages about this were found in the PDF.)"
            blocks.append(f"=== Evidence about: {entity}\n{body}")
        if not all_items:
            return _direct(NOT_FOUND_ANSWER)
        content = (
            "\n\n".join(blocks) + f"\n\nPages available for citation: {_page_list(all_items)}\n\nQuestion: {question}\n\n"
            + _mode_instruction(route) + "\n" + _LENGTH_INSTRUCTIONS[route.detail]
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_history(thread, question), {"role": "user", "content": content}]
        tokens = self._generate(messages, self._answer_tokens(route.detail, wide=True), model=self._model_for("compare"))
        return tokens, sources_for(all_items, _pdf_name(thread))

    def _handle_overview(self, thread, question, route: Route, store, structure):
        entries = store.entries()
        if not entries:
            return _direct(NOT_FOUND_ANSWER)
        sections = structure["sections"]
        units = structure["units"]
        budget = self.settings.overview_context_chars
        name = _pdf_name(thread)
        lines = [f"Document: {structure.get('title') or name}", f"Pages: {structure.get('page_count') or '?'}"]
        items = []
        summary_sections = [s["index"] for s in sections if "summary" in s.get("roles", [])][:2]
        summary_text = ""
        if summary_sections:
            scope = sorted({p for i in summary_sections for p in _section_positions(store, structure, i)})
            summary_items = excerpts(entries, budgeted(entries, scope, int(budget * 0.4)))
            items.extend(summary_items)
            summary_text = "The document's own summary/abstract:\n" + render(summary_items) + "\n\n"
        if units:
            noun = _display_noun(structure, None)
            lines.append(f"Structure: {len(units)} top-level {_plural(noun, len(units))}")
            for key in structure.get("fields") or []:
                values = Counter(sections[i]["fields"].get(key) for i in units if sections[i]["fields"].get(key))
                lines.append(f"{key} values: " + ", ".join(f"{value} ({count})" for value, count in values.most_common()))
            outline_entries = [s for s in sections if s["level"] <= structure["unit_level"] + 1 and not s["matter"]][:300]
            top = structure["unit_level"]
            outline = "\n".join(f"{'  ' * max(0, s['level'] - top)}- {s['title']} [{pages_text(s)}]" for s in outline_entries)
            remaining = budget - len(outline) - len(summary_text)
            share = max(150, remaining // max(1, len(units)))
            openings = []
            for index in units[:300]:
                positions = [p for p in store.positions_for_units([index]) if entries[p]["section"] == index][:1] or store.positions_for_units([index])[:1]
                if positions:
                    entry = entries[positions[0]]
                    openings.append(f"[Page {entry['page']}] {unit_heading(structure, sections[index], None)}: {entry['text'][:share]}")
                    items.extend(excerpts(entries, positions))
            context = "\n".join(lines) + f"\n\n{summary_text}Outline:\n{outline}\n\nOpening text of each part:\n" + "\n\n".join(openings)
        else:
            step = max(1, len(entries) // max(1, budget // 1000))
            positions = list(range(0, len(entries), step))
            sample = excerpts(entries, budgeted(entries, positions, budget - len(summary_text)))
            items.extend(sample)
            context = "\n".join(lines) + f"\n\n{summary_text}Sample passages from across the document:\n\n" + render(sample)
        content = (
            f"{context}\n\nRequest: {question}\n\n"
            "Using only the material above, answer the request about the whole document: its overall subject and "
            "purpose, how it is organized, and the main topics it covers, mentioning the parts by name. If the request "
            "names topics to cover, cover each of them. Follow any format or number of points the request asks for. "
            "Do not add anything that is not in the material.\n"
            f"{_LENGTH_INSTRUCTIONS[route.detail]}"
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        sources = sources_for(items, name)[:30]
        # Only the plain "what does it contain" overview is cached; requests with their own format are generated.
        generic = route.detail == "normal" and not re.search(r"\d", question)
        cache_key = f"overview:{_DIGEST_VERSION}"
        cached = store.digest(cache_key) if generic and hasattr(store, "digest") else None
        if cached:
            return iter((cached,)), sources
        tokens = max(self._answer_tokens(route.detail, wide=True), 500)
        return self._generate(messages, tokens, cache=(store, cache_key) if generic else None), sources

    def _answer_from(self, thread, question, items, route: Route, tokens=None, note: str = ""):
        if not items:
            return _direct(NOT_FOUND_ANSWER)
        instruction = _mode_instruction(route)
        content = (
            f"PDF excerpts:\n\n{render(items)}\n\nPages available for citation: {_page_list(items)}\n\n"
            + (f"{note}\n\n" if note else "")
            + f"Question: {question}"
            + (f"\n\n{instruction}" if instruction else "")
            + f"\n{_LENGTH_INSTRUCTIONS[route.detail]}"
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_history(thread, question), {"role": "user", "content": content}]
        wide = route.mode in ("multi", "compare", "analysis", "enumerate")
        max_tokens = tokens or self._answer_tokens(route.detail, wide=wide)
        return self._generate(messages, max_tokens, model=self._model_for(route.mode)), sources_for(items, _pdf_name(thread))

    def _answer_tokens(self, detail: str, wide: bool = False) -> int:
        """Output budget from the requested length; multi-part, comparison and list answers get a bit more room."""
        tokens = self.settings.max_answer_tokens * _LENGTH_TOKENS.get(detail, 0.7) * (1.4 if wide and detail != "brief" else 1)
        return max(200, int(tokens))

    def _has_evidence(self, hits: list[dict]) -> bool:
        if not hits:
            return False
        similarities = [hit["similarity"] for hit in hits if hit["similarity"] is not None]
        best_similarity = max(similarities) if similarities else None
        best_coverage = max(hit["coverage"] for hit in hits)
        reranked = [hit["rerank"] for hit in hits if "rerank" in hit]
        if reranked and max(reranked) > 0:
            return True
        return best_coverage >= 0.5 or (best_similarity is not None and best_similarity >= self.settings.retrieval_min_similarity)

    def _model_for(self, mode: str) -> str:
        analysis_model = getattr(self.settings, "analysis_model", "")
        if analysis_model and mode in ("analysis", "compare", "multi", "enumerate", "ambiguous"):
            return analysis_model
        return self.settings.chat_model

    # ---------------------------------------------------------------- LLM calls
    def _generate(self, messages, max_tokens: int, prefix: str = "", cache=None, model: str | None = None) -> Iterator[str]:
        def run() -> Iterator[str]:
            started = time.perf_counter()
            parts = []
            try:
                response = self._hf_client(model or self.settings.chat_model).chat_completion(
                    model=model or self.settings.chat_model,
                    messages=messages,
                    temperature=self.settings.temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                plain = PlainTextFilter()
                guard = RepetitionGuard()
                for chunk in response:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    cleaned = guard.feed(plain.feed(delta)) if delta else ""
                    if cleaned:
                        if not parts and prefix:
                            yield prefix
                        parts.append(cleaned)
                        yield cleaned
                    if guard.stopped:
                        print("[PERF] repetition_stopped=true")
                        break
                tail = "" if guard.stopped else guard.feed(plain.flush()) + guard.flush()
                if tail:
                    if not parts and prefix:
                        yield prefix
                    parts.append(tail)
                    yield tail
            except BackendError:
                raise
            except Exception as error:
                raise _llm_error(error) from error
            finally:
                print(f"[PERF] llm_generation_seconds={time.perf_counter() - started:.4f}")
            text = "".join(parts).strip()
            if not text:
                yield NOT_FOUND_ANSWER
            elif cache and text != NOT_FOUND_ANSWER:
                store, key = cache
                store.save_digest(key, text)

        return run()

    def _complete(self, messages, max_tokens: int, model: str | None = None) -> str:
        started = time.perf_counter()
        try:
            response = self._hf_client(model or self.settings.chat_model).chat_completion(
                model=model or self.settings.chat_model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=max_tokens,
            )
            plain = PlainTextFilter()
            text = plain.feed(response.choices[0].message.content or "")
            return text + plain.flush()
        except BackendError:
            raise
        except Exception as error:
            raise _llm_error(error) from error
        finally:
            print(f"[PERF] llm_completion_seconds={time.perf_counter() - started:.4f}")


# ------------------------------------------------------------------- helpers
class RepetitionGuard:
    """Stops a model that starts looping: a repeated line, or "... (continued)" headings.

    Text is released per line; a long unfinished line is released early once it can no longer be the start of
    a line that was already written, so streaming stays smooth."""

    def __init__(self):
        self.pending = ""
        self.seen: set[str] = set()
        self.stopped = False

    @staticmethod
    def _key(line: str) -> str:
        return re.sub(r"\W+", " ", line.casefold()).strip()

    def _is_repeat(self, line: str) -> bool:
        key = self._key(line)
        if re.search(r"\(\s*continued\s*\)", line, re.I):
            return True
        if len(key) >= 20 and key in self.seen:
            return True
        # A long paragraph that starts like an earlier one (which may have been released in parts).
        return len(key) >= 60 and any(len(seen) >= 60 and seen[:60] == key[:60] for seen in self.seen)

    def feed(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        self.pending += text
        out = []
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            if self._is_repeat(line):
                self.stopped, self.pending = True, ""
                break
            if len(self._key(line)) >= 20:
                self.seen.add(self._key(line))
            out.append(line + "\n")
        if not self.stopped and len(self.pending) > 160:
            start = self._key(self.pending[:80])
            if not any(seen.startswith(start) for seen in self.seen):
                out.append(self.pending)
                self.seen.add(self._key(self.pending))  # partial line: remember its start
                self.pending = ""
        return "".join(out)

    def flush(self) -> str:
        line, self.pending = self.pending, ""
        return "" if self.stopped or (line and self._is_repeat(line)) else line


class PlainTextFilter:
    """Streams model output as plain text: drops **bold** markers and #-headings, turns "* " bullets into "- "."""

    def __init__(self):
        self.carry = ""
        self.line_start = True

    def feed(self, text: str) -> str:
        text = self.carry + text
        cut = len(text)
        while cut and text[cut - 1] in "*#_":
            cut -= 1  # hold back a possibly incomplete marker until the next token arrives
        text, self.carry = text[:cut], text[cut:]
        return self._clean(text)

    def flush(self) -> str:
        text, self.carry = self.carry, ""
        return self._clean(text.replace("*", "").replace("#", "")) if text.strip("*#_") == "" else self._clean(text)

    def _clean(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("**", "").replace("__", "")
        work = ("\n" if self.line_start else "\x00") + text
        work = re.sub(r"(\n[ \t]*)[*•][ \t]+", r"\1- ", work)
        work = re.sub(r"\n[ \t]*#{1,6}[ \t]*", "\n", work)
        self.line_start = text.endswith("\n")
        return work[1:]


def _llm_error(error: Exception) -> BackendError:
    text = str(error)
    if "402" in text or "Payment Required" in text or "depleted" in text:
        return BackendError(
            "Hugging Face inference credits are exhausted for this token. Add credits or use another HF_TOKEN.",
            402,
            "llm_quota",
        )
    if "429" in text or "rate limit" in text.casefold():
        return BackendError("The language model is rate limited right now. Try again in a moment.", 429, "llm_rate_limited")
    return BackendError(f"Chat generation failed: {error}", 502, "llm_error")


def _search_many(store, queries: list[str], limit: int, positions=None) -> list[list[dict]]:
    queries = list(dict.fromkeys(q for q in queries if q and q.strip()))
    if hasattr(store, "search_many"):
        return store.search_many(queries, limit, positions)
    return [store.search(query, limit, positions=positions) if positions is not None else store.search(query, limit) for query in queries]


def _mode_instruction(route: Route) -> str:
    template = _MODE_INSTRUCTIONS.get(route.mode, "")
    if not template:
        return ""
    if route.mode == "multi":
        return template.format(parts="; ".join(f"({n}) {part}" for n, part in enumerate(route.subqueries, start=1)))
    if route.mode == "compare":
        aspect = f" in terms of their {route.aspect}" if route.aspect else ""
        return template.format(entities=", ".join(f'"{e}"' for e in route.entities) or "the items in the question", aspect=aspect)
    instruction = template
    if route.analysis and route.mode != "analysis":
        instruction += " " + _MODE_INSTRUCTIONS["analysis"]
    return instruction


def _layout_tables_answer(structure: dict, pdf_name: str):
    """Tables found from the page layout when the PDF has no "Table N" captions."""
    tables = structure["tables"]
    sections = structure["sections"]
    lines = []
    for table in tables:
        where = sections[table["section"]]["title"] if 0 <= table.get("section", -1) < len(sections) else ""
        columns = table["header"].replace(" | ", ", ")
        lines.append(f"- Page {table['page']}{f' ({where})' if where else ''}: {table['rows'] - 1} rows; columns: {columns}")
    answer = (
        f"The document has {len(tables)} tables (detected from the page layout; they have no numbered captions):\n\n"
        + "\n".join(lines)
    )
    return answer, [Source(page=t["page"], text=t["header"], pdf_name=pdf_name) for t in tables]


def _major_children(structure: dict, index: int) -> list[int]:
    """Sub-sections that matter: the numbered ones (2.1, 2.2 ...) when the section numbers them, otherwise all."""
    sections = structure["sections"]
    children = sections[index].get("children", [])
    numbered = [c for c in children if sections[c].get("number_path") or sections[c].get("noun")]
    return numbered if len(numbered) >= 2 else children


def _outline_note(structure: dict, indices: list[int]) -> str:
    """The sub-section titles of the sections in scope, so "which/what are all ..." answers can be complete."""
    sections = structure["sections"]
    blocks = []
    for index in indices[:4]:
        children = sections[index].get("children", [])
        if 2 <= len(children) <= 60:
            lines = "\n".join(f"- {sections[c]['title']} [{pages_text(sections[c])}]" for c in children)
            blocks.append(f"Outline of {sections[index]['title']} (its sub-sections, from the PDF's headings):\n{lines}")
    return "\n\n".join(blocks)


def _page_list(items) -> str:
    pages = cited_pages(items)
    return ", ".join(map(str, pages)) if pages else "none"


def _direct(answer: str, sources: list[Source] | None = None):
    return iter((answer,)), list(sources or [])


def _pdf_name(thread: Thread) -> str:
    return thread.pdf_name or "Uploaded PDF"


def _section_source(section: dict, pdf_name: str) -> Source:
    fields = "".join(f"\n{key}: {value}" for key, value in section.get("fields", {}).items())
    return Source(page=section["page_start"], text=f"{section['title']}{fields}", pdf_name=pdf_name)


def _section_positions(store, structure: dict, index: int) -> list[int]:
    positions = set(store.positions_for_sections(descendants(structure, index)))
    if index in structure.get("units", []):
        positions.update(store.positions_for_units([index]))
    return sorted(positions)


def _labelled_sections(structure: dict, noun: str | None) -> list[int]:
    """Sections the PDF itself labels with this noun ("Annex A", "Part Two", "Article 5"), one per number."""
    if not noun:
        return []
    best: dict[int, dict] = {}
    for section in structure["sections"]:
        if section.get("noun") != noun or section.get("number") is None:
            continue
        span = section.get("page_end", section["page_start"]) - section["page_start"]
        current = best.get(section["number"])
        if current is None or span > current.get("page_end", current["page_start"]) - current["page_start"]:
            best[section["number"]] = section
    result = [section["index"] for section in sorted(best.values(), key=lambda s: s["index"])]
    return result if result and (noun != structure.get("unit_noun") or len(result) < len(structure["units"])) else []


def _section_line(structure: dict, section: dict, requested: str | None) -> str:
    fields = "".join(f" — {key}: {value}" for key, value in section.get("fields", {}).items())
    return f"{unit_heading(structure, section, requested)}{fields} [{pages_text(section)}]"


def _display_noun(structure: dict, requested: str | None) -> str:
    return structure.get("unit_noun") or requested or "section"


def _plural(noun: str, count: int) -> str:
    return noun if count == 1 else plural_noun(noun)


def _numbering_note(structure: dict, units: list[int]) -> str:
    sections = structure["sections"]
    numbered = [i for i in units if sections[i].get("number") is not None]
    if not numbered or len(numbered) * 2 < len(units):
        return ""
    first, last = sections[numbered[0]], sections[numbered[-1]]
    first_label = first.get("display_number") or first.get("number")
    last_label = last.get("display_number") or last.get("number")
    if first_label == last_label:
        return ""
    return f" They are numbered {first_label} to {last_label}."


def _noun_note(structure: dict, requested: str | None) -> str:
    detected = structure.get("unit_noun")
    if requested and detected and requested != detected:
        return (
            f" (The PDF is divided into {plural_noun(detected)}, not {plural_noun(requested)}; "
            f"these {plural_noun(detected)} are its top-level parts.)"
        )
    return ""


def _missing_answer(structure: dict, route: Route, noun: str) -> str:
    units = structure.get("units", [])
    if route.missing_numbers:
        numbers = ", ".join(map(str, route.missing_numbers[:20]))
        answer = f"I couldn't find {noun} {numbers} in the uploaded PDF."
    else:
        answer = NOT_FOUND_ANSWER
    if units:
        answer += f" It contains {len(units)} top-level {_plural(noun, len(units))}." + _numbering_note(structure, units)
    return answer


def _previous_question(thread: Thread, question: str) -> str | None:
    messages = list(thread.messages)
    if messages and messages[-1].role == "user" and messages[-1].content == question:
        messages = messages[:-1]
    return next((m.content for m in reversed(messages) if m.role == "user"), None)


def _history(thread: Thread, question: str) -> list[dict]:
    messages = list(thread.messages)
    if messages and messages[-1].role == "user" and messages[-1].content == question:
        messages = messages[:-1]
    pairs = []
    for user, assistant in zip(messages, messages[1:]):
        if user.role == "user" and assistant.role == "assistant" and assistant.content.strip() != NOT_FOUND_ANSWER:
            pairs.append((user.content[:500], assistant.content[:700]))
    history = []
    for user, assistant in pairs[-2:]:
        history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    return history
