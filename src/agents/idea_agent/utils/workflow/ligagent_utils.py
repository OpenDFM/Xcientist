from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def enrich_papers_with_content(
    papers: List[Dict[str, Any]],
    paper_repository,
    artifact: Dict[str, Any],
    logger,
) -> None:
    if not papers:
        return
    paper_ids = [paper.get("paper_id") for paper in papers if paper.get("paper_id")]
    if not paper_ids:
        return
    try:
        prepared = paper_repository.prepare_papers(paper_ids)
    except Exception as exc:  # pragma: no cover - network/file heavy
        logger.warning("Failed to prepare parsed papers: %s", exc)
        return

    storage = artifact.setdefault("paper_contents", {})
    for paper in papers:
        pid = paper.get("paper_id")
        if not pid:
            continue
        content = prepared.get(pid)
        keynote_data = None
        if content:
            keynote_data = content.get("keynote") or content

        if not keynote_data:
            fallback_text = paper.get("abstract") or paper.get("title") or "No content available."
            keynote_data = {
                "tldr": fallback_text,
                "source": "title_abstract_fallback",
            }
            paper["has_parsed_markdown"] = False
        else:
            paper["has_parsed_markdown"] = True

        paper["keynote"] = keynote_data
        logger.info(
            f"🗒️ Enriched paper {paper.get('title')}: {keynote_data}",
        )
        storage[pid] = {
            "keynote": keynote_data,
            "source_keywords": paper.get("source_keywords"),
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
        }


def collect_paper_context_entries(
    artifact: Dict[str, Any],
    reference_batches: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Collect all paper context entries from memory.

    Since papers are already processed upstream (top-5 fully read, the rest
    compressed), no truncation limit is needed — every stored paper is
    included in the returned list.
    """
    storage = artifact.get("paper_contents", {})
    if not storage:
        return []

    ordered_ids: List[str] = []
    for batch in reference_batches or []:
        for paper in batch or []:
            if not isinstance(paper, dict):
                continue
            pid = paper.get("paper_id")
            if pid and pid in storage and pid not in ordered_ids:
                ordered_ids.append(pid)
    for pid in storage:
        if pid not in ordered_ids:
            ordered_ids.append(pid)

    entries: List[Dict[str, Any]] = []
    for pid in ordered_ids:
        data = storage.get(pid)
        if not data:
            continue
        keynote = data.get("keynote")
        summary = data.get("summary") or summarize_keynote(keynote, data.get("abstract"))
        source = "parsed"
        if data.get("compression") == "compressed":
            source = "compressed"
        elif isinstance(keynote, dict) and keynote.get("source"):
            source = keynote["source"]
        entries.append(
            {
                "paper_id": pid,
                "title": data.get("title") or pid,
                "summary": summary,
                "source": source,
                "authors": data.get("authors"),
            }
        )
    return entries


def summarize_keynote(keynote: Any, fallback: Optional[str]) -> str:
    if isinstance(keynote, dict):
        for key in ("tldr", "summary", "abstract"):
            value = keynote.get(key)
            if value:
                return str(value)
        return json.dumps(keynote, ensure_ascii=False)[:500]
    if keynote:
        return str(keynote)
    return (fallback or "No summary available.").strip()


def paper_context_text(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "No curated papers available yet."
    lines = []
    for idx, entry in enumerate(entries, 1):
        title = entry.get("title") or entry.get("paper_id")
        summary = entry.get("summary") or "No summary"
        source = entry.get("source") or "parsed"
        lines.append(f"{idx}. {title} ({source}): {summary}")
    return "\n".join(lines)


def generate_idea_introduction(
    chat_fn,
    prompt_template: str,
    model: str,
    topic: str,
    best_entry: Dict[str, Any],
    paper_entries: List[Dict[str, Any]],
    logger,
) -> str:
    entries = paper_entries or []
    if not entries:
        return fallback_introduction_text(best_entry, entries)
    prompt = prompt_template.format(
        topic=topic,
        idea=json.dumps(best_entry, ensure_ascii=False, indent=2),
        papers=json.dumps(entries, ensure_ascii=False, indent=2),
    )
    try:
        response = chat_fn(prompt, temperature=0.3, max_output_tokens=2048, model=model)
        payload = parse_json_response(response)
        intro = payload.get("introduction") or payload.get("intro")
        if intro:
            return intro.strip()
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Introduction generation failed: %s", exc)
    return fallback_introduction_text(best_entry, entries)


def fallback_introduction_text(
    best_entry: Dict[str, Any], paper_entries: List[Dict[str, Any]]
) -> str:
    title = best_entry.get("title", "This work")
    abstract = best_entry.get("abstract") or ""
    intro_lines = [
        f"{title} builds on recent literature to tackle the current topic. {abstract}".strip()
    ]
    if paper_entries:
        cite_lines = []
        for entry in paper_entries:
            cite_lines.append(
                f"- {entry.get('title') or entry.get('paper_id')}: {entry.get('summary', 'No summary available.')}"
            )
        intro_lines.append("Key references informing this idea:\n" + "\n".join(cite_lines))
    return "\n\n".join(intro_lines)


def parse_json_response(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty response")
    if text.startswith("```"):
        fence_end = text.find("\n")
        if fence_end != -1:
            text = text[fence_end + 1 :]
        if text.endswith("```"):
            text = text[: -3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch in "{[":
                try:
                    parsed, _ = decoder.raw_decode(text[idx:])
                    return parsed
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"Unable to parse JSON from response: {text[:200]}")
