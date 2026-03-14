"""Workflow helpers for retrieval, analysis shaping, seed conversion, and persistence prep."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

from src.agents.idea_agent.agent.artifacts import artifact_get
from src.agents.idea_agent.agent.prompts.experiment_findings_extraction import (
    EXPERIMENT_FINDINGS_EXTRACTION_PROMPT,
)
from src.agents.idea_agent.utils.workflow.idea_helpers import fallback_algorithm_spec
from src.agents.idea_agent.utils.workflow.idea_contract import normalize_idea_contract
from src.agents.idea_agent.utils.workflow.ligagent_utils import (
    enrich_papers_with_content,
    parse_json_response,
    summarize_keynote,
)
from src.agents.idea_agent.utils.prompting.prompt_views import format_paper_context_prompt_view


_ABLATION_RESULT_SIGN = {
    "positive": 1.0,
    "negative": -1.0,
    "inconclusive": 0.0,
    "mixed": 0.0,
    "neutral": 0.0,
}


def _normalize_ablation_result_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "pos": "positive",
        "beneficial": "positive",
        "good": "positive",
        "works": "positive",
        "neg": "negative",
        "harmful": "negative",
        "bad": "negative",
        "fails": "negative",
        "failure": "negative",
        "unclear": "inconclusive",
        "unknown": "inconclusive",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in _ABLATION_RESULT_SIGN else "inconclusive"


def _normalize_ablation_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_ablation_component_entry(
    component_name: str,
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    component = str(component_name or payload.get("component") or "").strip()
    if not component:
        return None

    result = _normalize_ablation_result_label(payload.get("result"))
    confidence = _normalize_ablation_confidence(payload.get("confidence"), default=0.5)
    normalized = {
        "component": component,
        "op": str(payload.get("op") or "remove").strip().lower() or "remove",
        "result": result,
        "metric": str(payload.get("metric") or "").strip(),
        "value": str(payload.get("value") or "").strip(),
        "analysis": str(payload.get("analysis") or payload.get("rationale") or "").strip(),
        "method_context": str(payload.get("method_context") or "").strip(),
        "confidence": confidence,
    }
    return normalized


def normalize_ablation_results_payload(results: Any) -> List[Dict[str, Any]]:
    if not results:
        return []

    normalized: List[Dict[str, Any]] = []
    if isinstance(results, list):
        for entry in results:
            if not isinstance(entry, dict):
                continue
            item = _normalize_ablation_component_entry(str(entry.get("component") or ""), entry)
            if item is not None:
                normalized.append(item)
        return normalized

    if not isinstance(results, dict):
        return []

    components = results.get("components") if isinstance(results.get("components"), dict) else {}
    for component_name, payload in components.items():
        if not isinstance(payload, dict):
            continue
        item = _normalize_ablation_component_entry(str(component_name), payload)
        if item is not None:
            normalized.append(item)
    return normalized


def generate_background_brief(
    topic: str,
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> Optional[str]:
    template = prompts.get("topic_background")
    if not template:
        return None
    prompt = template.format(topic=topic)
    try:
        payload = parse_json_response(chat_fn(prompt, model=model))
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Failed to bootstrap background knowledge for %s: %s", topic, exc)
        return None

    def _join_list(values: Any, prefix: str) -> Optional[str]:
        if isinstance(values, list) and values:
            joined = "; ".join(str(item) for item in values if item)
            if joined:
                return f"{prefix}: {joined}"
        return None

    if isinstance(payload, dict):
        sections = []
        summary = payload.get("background") or payload.get("summary")
        if summary:
            sections.append(str(summary).strip())
        extra_sections = [
            _join_list(payload.get("key_questions"), "Key questions"),
            _join_list(payload.get("canonical_methods"), "Representative methods"),
        ]
        sections.extend(line for line in extra_sections if line)
        compiled = " ".join(line for line in sections if line).strip()
        if compiled:
            return compiled
    if isinstance(payload, list):
        compiled = " ".join(str(item) for item in payload if item).strip()
        if compiled:
            return compiled
    if isinstance(payload, str):
        return payload.strip()
    return None


def normalize_search_papers(
    papers: List[Any],
    source_keywords: str,
    logger,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not papers:
        return normalized
    for i, paper in enumerate(papers, 1):
        if isinstance(paper, dict):
            title = paper.get("title") or f"Paper {i}"
            abstract = paper.get("abstract", "No abstract available.")
            authors_field = paper.get("authors", [])
            if isinstance(authors_field, list):
                authors = [a.get("name", str(a)) for a in authors_field]
            elif authors_field:
                authors = [str(authors_field)]
            else:
                authors = []
            paper_entry = {
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": paper.get("year"),
                "url": paper.get("url"),
                "tldr": paper.get("tldr"),
                "paper_id": paper.get("paperId") or paper.get("paper_id"),
                "source_keywords": source_keywords,
            }
        else:
            title = str(paper)
            paper_entry = {
                "title": title,
                "abstract": "No abstract available.",
                "authors": [],
                "year": None,
                "url": None,
                "tldr": None,
                "paper_id": None,
                "source_keywords": source_keywords,
            }
        logger.info(f"📄 {i}. {paper_entry['title']}")
        normalized.append(paper_entry)
    return normalized


def prepare_query_papers(
    papers: Optional[List[Dict[str, Any]]],
    paper_repository,
    logger,
    limit: int = 6,
) -> List[Dict[str, Any]]:
    if not papers:
        return []
    shortlist = papers[:limit]
    paper_ids = [paper.get("paper_id") for paper in shortlist if paper.get("paper_id")]
    prepared = {}
    if paper_ids:
        try:
            prepared = paper_repository.prepare_papers(paper_ids)
        except Exception as exc:  # pragma: no cover - network
            logger.warning("⚠️ Failed to prepare papers for query generation: %s", exc)
    query_papers: List[Dict[str, Any]] = []
    for paper in shortlist:
        pid = paper.get("paper_id")
        keynote = None
        if pid and prepared.get(pid):
            keynote = prepared[pid].get("keynote") or prepared[pid]
        summary = summarize_keynote(keynote, paper.get("abstract"))
        query_papers.append(
            {
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "keynote": summary,
                "paper_id": pid,
            }
        )
    return query_papers


def _single_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return "No summary available."
    for sep in (". ", "? ", "! "):
        if sep in cleaned:
            return cleaned.split(sep, 1)[0].strip() + sep.strip()
    return cleaned


def filter_and_compress_papers(
    topic: str,
    mature_idea: Optional[str],
    papers: List[Dict[str, Any]],
    artifact: Dict[str, Any],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if not papers:
        return papers
    prompt_template = prompts.get("paper_filtering")
    if not prompt_template:
        logger.warning("⚠️ paper_filtering prompt missing; skipping LLM triage.")
        return papers

    storage = artifact_get(artifact, "paper_contents", {})
    candidates: List[Dict[str, Any]] = []
    paper_map: Dict[str, Dict[str, Any]] = {}
    for idx, paper in enumerate(papers, 1):
        if not isinstance(paper, dict):
            continue
        pid = paper.get("paper_id")
        if not pid:
            pid = f"fallback-{uuid.uuid4().hex}"
            paper["paper_id"] = pid
        paper_map[pid] = paper
        keynote = paper.get("keynote") or storage.get(pid, {}).get("keynote")
        summary = summarize_keynote(keynote, paper.get("abstract"))
        candidates.append(
            {
                "index": idx,
                "paper_id": pid,
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "summary": summary,
                "source_keywords": paper.get("source_keywords"),
            }
        )

    if not candidates:
        return papers

    prompt = prompt_template.format(
        topic=topic or "",
        mature_idea=(mature_idea or "").strip() or "N/A",
        top_k=min(top_k, len(candidates)),
        papers=json.dumps(candidates, ensure_ascii=False, indent=2),
    )
    prompt += "\nDirectly output JSON."
    top_ids: List[str] = []
    compressed_map: Dict[str, str] = {}
    try:
        response = chat_fn(prompt, temperature=0.1, max_output_tokens=65536, model=model)
        payload = parse_json_response(response)
        raw_top = (
            payload.get("top_paper_ids")
            or payload.get("top_papers")
            or payload.get("top_ids")
        )
        if isinstance(raw_top, list):
            for item in raw_top:
                if isinstance(item, dict):
                    pid = (
                        item.get("paper_id")
                        or item.get("paperId")
                        or item.get("id")
                    )
                else:
                    pid = str(item)
                if pid and pid in paper_map and pid not in top_ids:
                    top_ids.append(pid)
        raw_compressed = payload.get("compressed") or payload.get("summaries") or []
        if isinstance(raw_compressed, dict):
            for pid, summary in raw_compressed.items():
                if pid in paper_map and summary:
                    compressed_map[pid] = _single_sentence(str(summary))
        elif isinstance(raw_compressed, list):
            for item in raw_compressed:
                if not isinstance(item, dict):
                    continue
                pid = item.get("paper_id") or item.get("paperId") or item.get("id")
                summary = item.get("summary")
                if pid in paper_map and summary:
                    compressed_map[pid] = _single_sentence(str(summary))
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Paper triage failed: %s", exc)

    if not top_ids:
        top_ids = [entry["paper_id"] for entry in candidates[:top_k]]
    if len(top_ids) < min(top_k, len(candidates)):
        for entry in candidates:
            pid = entry["paper_id"]
            if pid not in top_ids:
                top_ids.append(pid)
            if len(top_ids) >= min(top_k, len(candidates)):
                break

    top_set = set(top_ids)
    fallback_map = {entry["paper_id"]: entry["summary"] for entry in candidates}
    for pid in paper_map:
        if pid not in top_set and pid not in compressed_map:
            compressed_map[pid] = _single_sentence(fallback_map.get(pid, ""))

    ordered: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for pid in top_ids:
        paper = paper_map.get(pid)
        if paper and pid not in seen:
            ordered.append(paper)
            seen.add(pid)
    for paper in papers:
        pid = paper.get("paper_id")
        if pid and pid not in seen:
            ordered.append(paper)
            seen.add(pid)

    for pid, paper in paper_map.items():
        if pid in top_set:
            entry = storage.get(pid, {})
            entry.setdefault("keynote", paper.get("keynote"))
            entry.setdefault("title", paper.get("title") or pid)
            entry.setdefault("abstract", paper.get("abstract"))
            entry.setdefault("authors", paper.get("authors"))
            entry.setdefault("source_keywords", paper.get("source_keywords"))
            entry["compression"] = "full"
            storage[pid] = entry
            continue
        summary = compressed_map.get(pid) or "No summary available."
        entry = storage.get(pid, {})
        entry.update(
            {
                "title": paper.get("title") or pid,
                "abstract": None,
                "authors": paper.get("authors"),
                "source_keywords": paper.get("source_keywords"),
                "summary": summary,
                "keynote": {"summary": summary, "source": "llm_compressed"},
                "compression": "compressed",
            }
        )
        storage[pid] = entry
        paper["summary"] = summary
        paper["compressed"] = True
        paper.pop("abstract", None)
        paper.pop("tldr", None)
        paper.pop("keynote", None)

    logger.info(
        "🗂️ Paper triage complete: %d full, %d compressed.",
        len(top_set),
        max(0, len(paper_map) - len(top_set)),
    )
    return ordered


def generate_rag_query(
    topic: str,
    papers: List[Dict[str, Any]],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
    mature_idea: Optional[str] = None,
) -> str:
    prompt = prompts["rag_query"].format(
        topic=topic,
        mature_idea=(mature_idea or "").strip(),
        papers=json.dumps(papers, ensure_ascii=False, indent=2) if papers is not None else "[]",
    )
    try:
        response = chat_fn(prompt, model=model, temperature=0.3, max_output_tokens=65536)
        try:
            payload = parse_json_response(response)
            if isinstance(payload, dict) and payload.get("query"):
                query = str(payload["query"]).strip()
            else:
                query = str(payload).strip()
        except Exception:
            query = (response or "").strip()
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Failed to generate RAG query: %s", exc)
        query = topic
    if not query:
        query = topic
    return query


def retrieve_outcome_rag(query: str, top_k: int, paper_repository, logger) -> List[Dict[str, Any]]:
    try:
        hits = paper_repository.retrieve_outcome_rag(query=query, top_k=top_k)
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ OutcomeRAG retrieval failed: %s", exc)
        hits = []
    return hits


def collect_rag_citations(hits: List[Dict[str, Any]]) -> List[str]:
    titles: List[str] = []
    seen = set()
    for hit in hits or []:
        citations = hit.get("citations", [])
        for title in citations:
            cleaned = (title or "").strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            titles.append(cleaned)
    return titles

def collect_rag_contents(hits: List[Dict[str, Any]]) -> List[str]:
    contents: List[str] = []
    for hit in hits or []:
        subsection = hit.get("subsection", "").strip()
        contents.append(subsection)
    return contents


def search_papers_from_citations(
    titles: List[str],
    rag_query: str,
    paper_repository,
) -> List[Dict[str, Any]]:
    if not titles:
        return []
    papers = paper_repository.search_papers_by_title(titles)
    papers = [paper for paper in papers if paper.get("paper_id")]
    for paper in papers:
        paper["source_keywords"] = rag_query
    return papers


def fallback_paper_summaries(
    papers: List[Dict[str, Any]],
    artifact: Dict[str, Any],
    reason: str,
) -> None:
    storage = artifact_get(artifact, "paper_contents", {})
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        pid = paper.get("paper_id")
        if not pid:
            pid = f"fallback-{uuid.uuid4().hex}"
            paper["paper_id"] = pid
        if pid in storage and storage[pid].get("keynote"):
            continue
        fallback_text = (
            paper.get("abstract")
            or paper.get("title")
            or paper.get("tldr")
            or "No parsed content available yet."
        )
        keynote_data = {
            "tldr": fallback_text,
            "source": reason,
        }
        paper["keynote"] = keynote_data
        paper["has_parsed_markdown"] = False
        storage[pid] = {
            "keynote": keynote_data,
            "source_keywords": paper.get("source_keywords"),
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "authors": paper.get("authors"),
        }


def safely_enrich_papers_with_content(
    papers: List[Dict[str, Any]],
    timeout: int,
    paper_repository,
    artifact: Dict[str, Any],
    logger,
) -> None:
    if not papers:
        return
    timeout = max(30, int(timeout or 0))
    executor = ThreadPoolExecutor(max_workers=10)
    future = executor.submit(
        enrich_papers_with_content, papers, paper_repository, artifact, logger
    )
    try:
        future.result(timeout=timeout)
    except FuturesTimeoutError:
        future.cancel()
        logger.warning(
            "⚠️ Paper enrichment exceeded %ss. Falling back to lightweight summaries.",
            timeout,
        )
        fallback_paper_summaries(papers, artifact, reason="timeout_fallback")
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Paper enrichment failed: %s", exc)
        fallback_paper_summaries(papers, artifact, reason="error_fallback")
    finally:
        executor.shutdown(wait=False)


def format_rag_context(artifact: Dict[str, Any], max_hits: int = 5, max_chars: int = 320) -> str:
    rag_entries = artifact_get(artifact, "rag_hits", [])
    latest = rag_entries[-1] if rag_entries else None
    hits = []
    if isinstance(latest, dict):
        hits = latest.get("hits") or []
    elif isinstance(latest, list):
        hits = latest
    if not hits:
        return ""
    lines = []
    for idx, hit in enumerate(hits[:max_hits], 1):
        title = hit.get("title") or f"RAG hit {idx}"
        snippet = (hit.get("subsection") or "").strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3] + "..."
        citations = hit.get("citations") or []
        cite_preview = "; ".join(citations[:5]) if citations else "no citations"
        lines.append(f"{idx}. {title}: {snippet} (citations: {cite_preview})")
    return "\n".join(lines)

def format_survey_context(
    artifact: Dict[str, Any],
    max_items: int = 5,
    max_chars: int = 360,
) -> str:
    contents = artifact_get(artifact, "rag_contents", [])
    latest = contents[-1] if contents else None
    if isinstance(latest, list):
        items = latest
    elif isinstance(latest, str):
        items = [latest]
    else:
        items = []
    if not items:
        return ""
    lines = []
    for idx, section in enumerate(items[:max_items], 1):
        snippet = (section or "").strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3] + "..."
        if snippet:
            lines.append(f"{idx}. {snippet}")
    return "\n".join(lines)


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = float(default)
    return max(0.0, min(1.0, score))


def _normalize_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _extractor_config_value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        value = config.get(key, default)
        return default if value is None else value
    value = getattr(config, key, default)
    return default if value is None else value


def _normalize_component_finding(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    component = str(item.get("component") or "").strip()
    if not component:
        return None
    result = str(item.get("result") or "inconclusive").strip().lower() or "inconclusive"
    return {
        "component": component,
        "result": result,
        "metric": str(item.get("metric") or "").strip(),
        "value": str(item.get("value") or "").strip(),
        "confidence": _coerce_confidence(item.get("confidence"), default=0.0),
        "analysis": str(item.get("analysis") or "").strip(),
        "action_hint": str(item.get("action_hint") or "").strip(),
        "is_critical": bool(item.get("is_critical")),
        "theme_tags": _normalize_string_list(item.get("theme_tags")),
    }


def _normalize_theme_entry(item: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    theme = str(item.get("theme") or "").strip()
    if not theme:
        return None
    try:
        count = int(item.get("count", 0))
    except (TypeError, ValueError):
        count = 0
    return {
        "theme": theme,
        "count": max(0, count),
        "components": _normalize_string_list(item.get("components")),
    }


def _normalize_experiment_findings_payload(payload: Any) -> Dict[str, Any]:
    empty_payload = {
        "hypothesis_status": "inconclusive",
        "feasible": None,
        "overall_confidence": 0.0,
        "tldr": "No structured ablation findings available.",
        "key_findings": [],
        "component_findings": [],
        "negative_components": [],
        "positive_components": [],
        "inconclusive_components": [],
        "critical_failures": [],
        "promising_components": [],
        "dominant_themes": [],
    }
    if not isinstance(payload, dict):
        return empty_payload

    component_findings = [
        item
        for item in (
            _normalize_component_finding(raw)
            for raw in (payload.get("component_findings") or [])
        )
        if item is not None
    ]
    if not component_findings:
        negative_components = _normalize_string_list(payload.get("negative_components"))
        positive_components = _normalize_string_list(payload.get("positive_components"))
        inconclusive_components = _normalize_string_list(payload.get("inconclusive_components"))
    else:
        negative_components = [item["component"] for item in component_findings if item["result"] == "negative"]
        positive_components = [item["component"] for item in component_findings if item["result"] == "positive"]
        inconclusive_components = [
            item["component"] for item in component_findings if item["result"] not in {"negative", "positive"}
        ]

    critical_failures = [
        item
        for item in (
            _normalize_component_finding(raw)
            for raw in (payload.get("critical_failures") or [])
        )
        if item is not None
    ] or [item for item in component_findings if item["is_critical"]]

    promising_components = [
        item
        for item in (
            _normalize_component_finding(raw)
            for raw in (payload.get("promising_components") or [])
        )
        if item is not None
    ] or [item for item in component_findings if item["result"] == "positive"]

    dominant_themes = [
        item
        for item in (
            _normalize_theme_entry(raw)
            for raw in (payload.get("dominant_themes") or [])
        )
        if item is not None
    ]

    feasible_value = payload.get("feasible")
    feasible = feasible_value if isinstance(feasible_value, bool) else None

    return {
        "hypothesis_status": str(payload.get("hypothesis_status") or "inconclusive").strip().lower() or "inconclusive",
        "feasible": feasible,
        "overall_confidence": _coerce_confidence(payload.get("overall_confidence"), default=0.0),
        "tldr": str(payload.get("tldr") or "No structured ablation findings available.").strip(),
        "key_findings": _normalize_string_list(payload.get("key_findings")),
        "component_findings": component_findings,
        "negative_components": negative_components,
        "positive_components": positive_components,
        "inconclusive_components": inconclusive_components,
        "critical_failures": critical_failures,
        "promising_components": promising_components,
        "dominant_themes": dominant_themes,
    }


def extract_experiment_findings_from_raw_ablation(
    raw_ablation: Any,
    *,
    chat_fn,
    model: Optional[str] = None,
    logger=None,
    prompt_template: Optional[str] = None,
    stage: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    extractor_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Use an LLM to extract structured findings from raw ablation JSON."""
    if not isinstance(raw_ablation, dict):
        return _normalize_experiment_findings_payload({})

    resolved_model = str(
        model
        or _extractor_config_value(extractor_config, "model", "gpt-5-mini")
        or "gpt-5-mini"
    ).strip()
    resolved_stage = str(
        stage
        or _extractor_config_value(
            extractor_config,
            "stage",
            "experiment_findings_extraction",
        )
        or "experiment_findings_extraction"
    ).strip()
    resolved_temperature = float(
        temperature
        if temperature is not None
        else _extractor_config_value(extractor_config, "temperature", 0.1)
    )
    resolved_max_output_tokens = int(
        max_output_tokens
        if max_output_tokens is not None
        else _extractor_config_value(extractor_config, "max_output_tokens", 65536)
    )

    prompt = (prompt_template or EXPERIMENT_FINDINGS_EXTRACTION_PROMPT).format(
        raw_ablation=json.dumps(raw_ablation, ensure_ascii=False, indent=2),
    )
    try:
        response = chat_fn(
            prompt,
            model=resolved_model,
            stage=resolved_stage,
            temperature=resolved_temperature,
            max_output_tokens=resolved_max_output_tokens,
        )
        payload = parse_json_response(response)
        return _normalize_experiment_findings_payload(payload)
    except Exception as exc:
        if logger is not None:
            logger.warning("⚠️ Failed to extract experiment findings from raw ablation: %s", exc)
        return _normalize_experiment_findings_payload({})


def paper_context_with_rag(entries: List[Dict[str, Any]], artifact: Dict[str, Any]) -> str:
    return format_paper_context_prompt_view(entries, artifact)


def get_paper_content(
    paper_id: str,
    include_markdown: bool,
    artifact: Dict[str, Any],
    paper_repository,
    logger,
) -> Dict[str, Any]:
    if not paper_id:
        return {}
    stored = artifact_get(artifact, "paper_contents", {}).get(paper_id, {}).copy()
    stored["paper_id"] = paper_id
    if include_markdown:
        try:
            stored["markdown"] = paper_repository.get_markdown(paper_id)
        except Exception as exc:  # pragma: no cover - network
            logger.warning("Unable to load markdown for %s: %s", paper_id, exc)
    return stored


def normalize_analysis_entry(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict):
                return item
        return {
            "key_methods": [],
            "existing_problems": [],
            "future_directions": [],
            "tldr": "; ".join(str(it) for it in response[:3]),
        }
    if isinstance(response, str):
        return {
            "key_methods": [],
            "existing_problems": [],
            "future_directions": [],
            "tldr": response,
        }
    return {
        "key_methods": [],
        "existing_problems": [],
        "future_directions": [],
        "tldr": "Analysis output was not structured; falling back to placeholder.",
    }


def ingest_analysis_background(analysis_entry: Dict[str, Any], artifact: Dict[str, Any]) -> None:
    for line in collect_analysis_background_lines(analysis_entry):
        if not line:
            continue
        background_store = artifact_get(artifact, "background_knowledge", [])
        if line not in background_store:
            background_store.append(line)


def collect_analysis_background_lines(analysis_entry: Dict[str, Any]) -> List[str]:
    if not isinstance(analysis_entry, dict):
        return []
    root_idea = analysis_entry.get("root_idea")
    if isinstance(root_idea, dict):
        try:
            normalized_root = normalize_idea_contract(root_idea, allow_legacy=True, keep_extra=True)
            title = normalized_root.get("title") or "Root Idea"
            abstract = normalized_root.get("abstract") or ""
            mechanism = normalized_root.get("method") or ""
            root_line = f"[Root Idea] {title}: {abstract}".strip()
            if mechanism:
                root_line += f" | Mechanism: {mechanism}"
            if root_line:
                return [root_line]
        except Exception:
            pass
    seeds = (
        analysis_entry.get("divergent_idea_seeds")
        or analysis_entry.get("moonshot_hypotheses")
        or []
    )
    if not isinstance(seeds, list) or not seeds:
        return []
    background_lines = []
    for seed in seeds[:3]:
        if not isinstance(seed, dict):
            continue
        title = seed.get("title") or seed.get("hypothesis") or "Moonshot Seed"
        hypothesis = seed.get("hypothesis") or ""
        method = seed.get("method_sketch") or seed.get("method") or ""
        gap = seed.get("why_it_is_not_incremental") or seed.get("why_now") or ""
        snippet = f"[Moonshot Seed] {title}: {hypothesis}".strip()
        if method:
            snippet += f" | Mechanism: {method}"
        if gap:
            snippet += f" | Differentiator: {gap}"
        background_lines.append(snippet)
    return [line for line in background_lines if line]


def extract_root_idea_from_analysis(
    analysis_entry: Dict[str, Any],
    *,
    topic: str,
) -> Dict[str, Any]:
    if not isinstance(analysis_entry, dict):
        return normalize_idea_contract(
            {
                "title": f"{topic} root idea",
                "abstract": f"Root idea derived from topic '{topic}'.",
                "core_contribution": "Establish a concrete mechanism-level root idea from analysis.",
                "method": "Synthesize the dominant method cluster with one explicit gap-closing mechanism.",
                "experiments": "Validate against the main baselines and gap-focused stress tests.",
                "risks": "Risk of under-specifying the mechanism before search refinement.",
                "target_defects": ["unclear_mechanism"],
                "rationale": "Fallback root idea because structured analysis output was unavailable.",
            },
            keep_extra=True,
        )

    root_idea = analysis_entry.get("root_idea")
    if isinstance(root_idea, dict):
        try:
            return normalize_idea_contract(root_idea, allow_legacy=True, keep_extra=True)
        except Exception:
            pass

    seeds = analysis_candidate_ideas({"analysis": [analysis_entry]})
    if seeds:
        seed = dict(seeds[0])
        seed["operator"] = "analysis_root"
        return normalize_idea_contract(seed, allow_legacy=True, keep_extra=True)

    problems = analysis_entry.get("existing_problems") or []
    gaps = analysis_entry.get("evaluation_gaps") or []
    future = analysis_entry.get("future_directions") or []
    key_methods = analysis_entry.get("key_methods") or []
    tldr = str(analysis_entry.get("tldr") or "").strip()

    gap_text = ""
    if isinstance(gaps, list) and gaps:
        first_gap = gaps[0]
        if isinstance(first_gap, dict):
            gap_text = str(first_gap.get("gap") or "").strip()
        else:
            gap_text = str(first_gap).strip()
    problem_text = str(problems[0]).strip() if isinstance(problems, list) and problems else ""
    method_text = str(key_methods[0]).strip() if isinstance(key_methods, list) and key_methods else ""
    future_text = str(future[0]).strip() if isinstance(future, list) and future else ""

    abstract_parts = [part for part in [tldr, problem_text, gap_text] if part]
    core = future_text or gap_text or problem_text or tldr or f"Root idea for {topic}."
    method = method_text or "Use the dominant method family as the starting mechanism and refine it during search."
    experiments = (
        f"Run fair baselines and targeted stress tests for: {gap_text or problem_text or 'the main open gap'}."
    )
    risks = problem_text or "The root idea may still be under-specified before search refinement."
    return normalize_idea_contract(
        {
            "title": f"{topic} root idea",
            "abstract": " ".join(abstract_parts) or f"Root idea derived from topic '{topic}'.",
            "core_contribution": core,
            "method": method,
            "experiments": experiments,
            "risks": risks,
            "target_defects": ["unclear_mechanism"],
            "rationale": "Root idea synthesized from the latest advanced analysis.",
        },
        keep_extra=True,
    )


def analysis_candidate_ideas(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    analysis_entries = artifact_get(artifact, "analysis", [])
    if not analysis_entries:
        return []
    latest = analysis_entries[-1]
    if not isinstance(latest, dict):
        return []
    seeds = latest.get("divergent_idea_seeds") or latest.get("moonshot_hypotheses") or []
    return convert_analysis_candidates_to_ideas(seeds)


def convert_analysis_candidates_to_ideas(seeds: Any) -> List[Dict[str, Any]]:
    if not isinstance(seeds, list):
        return []
    payloads: List[Dict[str, Any]] = []
    for idx, seed in enumerate(seeds):
        if not isinstance(seed, dict):
            continue
        title = seed.get("title") or seed.get("hypothesis") or f"Moonshot Seed #{idx + 1}"
        hypothesis = seed.get("hypothesis") or ""
        method = seed.get("method_sketch") or seed.get("method") or ""
        differentiator = seed.get("why_it_is_not_incremental") or seed.get("why_now") or ""
        evaluation_plan = seed.get("evaluation_plan") or seed.get("evaluation") or ""
        risk = seed.get("risk") or seed.get("risk_surface") or ""
        supporting = seed.get("supporting_papers", [])
        if isinstance(supporting, str):
            supporting = [supporting]
        tags = ["analysis-seed", "moonshot"]
        if seed.get("source_field"):
            tags.append(str(seed["source_field"]).lower().replace(" ", "-"))
        custom_tags = seed.get("tags")
        if isinstance(custom_tags, list):
            tags.extend(str(tag) for tag in custom_tags if tag)
        payloads.append(
            normalize_idea_contract(
                {
                    "title": title,
                    "abstract": " | ".join(
                        part
                        for part in [
                            hypothesis,
                            f"Mechanism: {method}" if method else "",
                        ]
                        if part
                    ),
                    "core_contribution": differentiator or hypothesis or method or "Analysis-seeded moonshot hypothesis.",
                    "method": method or "Derived from divergent analysis seed; requires fleshing out.",
                    "experiments": evaluation_plan or "Design ICML-grade evaluation including failure-surface probes.",
                    "risks": risk or "High novelty risk; feasibility unknown.",
                    "tags": tags,
                    "operator": "analysis_root_candidate",
                    "target_defects": seed.get("target_defects", ["stagnant_novelty"]),
                    "memory_refs": supporting if isinstance(supporting, list) else [str(supporting)],
                    "rationale": differentiator or "Seed extracted from advanced analysis.",
                }
            )
        )
    return payloads


def build_algorithm_spec(
    idea: Dict[str, Any],
    topic: str,
    raw_references: List[Dict[str, Any]],
    artifact: Dict[str, Any],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> List[Dict[str, Any]]:
    idea = normalize_idea_contract(idea, keep_extra=True)
    idea_for_prompt = {
        key: value
        for key, value in idea.items()
        if key not in {"search_path", "search_trace", "pareto_candidates", "evaluation"}
    }
    analysis_entries = artifact_get(artifact, "analysis", [])
    latest_analysis = analysis_entries[-1] if analysis_entries else {}
    base_inputs: List[str] = []
    if topic and topic != "unspecified topic":
        base_inputs.append(f"Topic focus: {topic}")
    retrieval_history = artifact_get(artifact, "retrieval_keywords", [])
    if retrieval_history:
        base_inputs.append(f"Latest retrieval keywords: {retrieval_history[-1]}")
    if isinstance(latest_analysis, dict):
        tldr = latest_analysis.get("tldr")
        if tldr:
            base_inputs.append(f"Analysis TL;DR: {tldr}")
        key_methods = latest_analysis.get("key_methods")
        if key_methods:
            base_inputs.append("Key methods referenced: " + "; ".join(key_methods[:3]))
    if raw_references:
        ref_titles = [r.get("title") for r in raw_references[:3] if r.get("title")]
        if ref_titles:
            base_inputs.append("Reference anchors: " + "; ".join(ref_titles))
    target_defects = idea.get("target_defects")
    if target_defects:
        base_inputs.append(f"Target defects: {', '.join(target_defects)}")

    base_outputs: List[str] = []
    abstract = idea.get("abstract")
    if abstract:
        base_outputs.append(f"Abstract focus: {abstract}")
    core = idea.get("core_contribution")
    if core:
        base_outputs.append(f"Core contribution: {core}")
    method = idea.get("method")
    if method:
        base_outputs.append(f"Methodology: {method}")
    experiments = idea.get("experiments")
    if experiments:
        base_outputs.append(f"Experiment design: {experiments}")
    score = idea.get("search_score")
    if isinstance(score, (int, float)):
        base_outputs.append(f"MCTS search score: {score:.2f}")

    prompt = prompts["algorithm_structuring"].format(
        topic=topic,
        idea_title=idea.get("title", ""),
        idea_abstract=idea.get("abstract", ""),
        idea=json.dumps(idea_for_prompt, ensure_ascii=False, indent=2),
        base_inputs=json.dumps(base_inputs, ensure_ascii=False, indent=2),
        base_outputs=json.dumps(base_outputs, ensure_ascii=False, indent=2),
        analysis=json.dumps(latest_analysis, ensure_ascii=False, indent=2),
        references=json.dumps(raw_references[:5], ensure_ascii=False, indent=2),
    )
    prompt += "\n Directly output JSON."
    try:
        response = chat_fn(prompt, temperature=0.01, max_output_tokens=65536, model=model)
        payload = parse_json_response(response)
        candidate = payload.get("algorithms", payload)
        if isinstance(candidate, list) and candidate:
            return align_algorithms_with_idea(
                idea, candidate, prompts, chat_fn, model, logger
            )
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Algorithm structuring failed: %s", exc)

    return fallback_algorithm_spec(idea, base_inputs, base_outputs)


def align_algorithms_with_idea(
    idea: Dict[str, Any],
    algorithms: List[Dict[str, Any]],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> List[Dict[str, Any]]:
    title = (idea.get("title") or "").strip()
    abstract = (idea.get("abstract") or "").strip()
    if not algorithms or (not title and not abstract):
        return algorithms

    prompt = prompts["algorithm_alignment"].format(
        idea_title=title,
        idea_abstract=abstract or "No abstract provided.",
        algorithms=json.dumps(algorithms, ensure_ascii=False, indent=2),
    )
    prompt += "\nDirectly output JSON."
    try:
        response = chat_fn(prompt, temperature=0.01, max_output_tokens=65536, model=model)
        payload = parse_json_response(response)
        candidate = payload.get("algorithms", payload)
        if isinstance(candidate, list) and candidate:
            return candidate
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Algorithm alignment failed: %s", exc)
    return algorithms


def synthesize_reference_summaries(
    topic: str,
    best_entry: Dict[str, Any],
    algorithm: List[Dict[str, Any]],
    raw_references: List[Dict[str, Any]],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> List[Dict[str, Any]]:
    if not raw_references:
        return []
    prompt = prompts["reference_grounding"].format(
        topic=topic,
        idea_title=best_entry.get("title", ""),
        idea_abstract=best_entry.get("abstract", ""),
        algorithm=json.dumps(algorithm, ensure_ascii=False, indent=2),
        references=json.dumps(raw_references, ensure_ascii=False, indent=2),
    )
    prompt += "\n Directly output JSON."
    try:
        response = chat_fn(prompt, temperature=0.01, max_output_tokens=65536, model=model)
        payload = parse_json_response(response)
        candidate = payload.get("reference_papers", payload)
        if isinstance(candidate, list):
            return candidate
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Reference synthesis failed: %s", exc)
    return raw_references
