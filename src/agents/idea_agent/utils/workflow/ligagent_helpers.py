"""Workflow helpers for retrieval, analysis shaping, seed conversion, and persistence prep."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.agents.idea_agent.agent.artifacts import artifact_get
from src.agents.idea_agent.agent.prompts.experiment_findings_extraction import (
    EXPERIMENT_FINDINGS_EXTRACTION_PROMPT,
)
from src.agents.idea_agent.utils.workflow.idea_helpers import fallback_algorithm_spec
from src.agents.idea_agent.utils.workflow.idea_contract import normalize_idea_contract
from src.agents.idea_agent.utils.workflow.ligagent_utils import (
    parse_json_response,
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


def generate_rag_query(
    topic: str,
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
    mature_idea: Optional[str] = None,
) -> str:
    prompt = prompts["rag_query"].format(
        topic=topic,
        mature_idea=(mature_idea or "").strip(),
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


def search_core_nodes_from_citations(
    titles: List[str],
    rag_query: str,
    paper_repository,
) -> List[Dict[str, Any]]:
    references = paper_repository.search_core_nodes_by_titles(titles)
    for reference in references:
        reference["source_keywords"] = rag_query
    return references


def search_core_nodes_from_query(
    query: str,
    paper_repository,
) -> List[Dict[str, Any]]:
    return paper_repository.search_core_nodes_by_query(query)


def select_core_references(
    references: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    return list(references[: max(0, int(top_k))])


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
    return _normalize_component_finding_with_fallback(item)


def _normalize_component_finding_with_fallback(
    item: Any,
    fallback: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    source = item if isinstance(item, dict) else {}
    base = fallback if isinstance(fallback, dict) else {}
    component = str(source.get("component") or base.get("component") or "").strip()
    if not component:
        return None
    result = _normalize_ablation_result_label(source.get("result") or base.get("result"))
    metric = source.get("metric")
    if metric is None:
        metric = base.get("metric")
    value = source.get("value")
    if value is None:
        value = base.get("value")
    confidence_value = source.get("confidence")
    if confidence_value is None:
        confidence_value = base.get("confidence")
    analysis = source.get("analysis")
    if analysis is None:
        analysis = base.get("analysis")
    return {
        "component": component,
        "result": result,
        "metric": str(metric or "").strip(),
        "value": str(value or "").strip(),
        "confidence": _coerce_confidence(confidence_value, default=0.0),
        "analysis": str(analysis or "").strip(),
    }


def _normalize_experiment_summary(
    payload: Dict[str, Any],
    raw_ablation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload_summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    raw_summary = raw_ablation.get("summary") if isinstance(raw_ablation, dict) and isinstance(raw_ablation.get("summary"), dict) else {}

    feasible_value = payload_summary.get("feasible")
    if not isinstance(feasible_value, bool):
        feasible_value = payload.get("feasible")
    if not isinstance(feasible_value, bool):
        feasible_value = raw_summary.get("feasible")
    feasible = feasible_value if isinstance(feasible_value, bool) else None

    confidence_value = payload_summary.get("overall_confidence")
    if confidence_value is None:
        confidence_value = payload.get("overall_confidence")
    if confidence_value is None:
        confidence_value = payload_summary.get("confidence")
    if confidence_value is None:
        confidence_value = payload.get("confidence")
    if confidence_value is None:
        confidence_value = raw_summary.get("confidence")

    key_findings = _normalize_string_list(payload_summary.get("key_findings"))
    if not key_findings:
        key_findings = _normalize_string_list(payload.get("key_findings"))
    if not key_findings:
        key_findings = _normalize_string_list(raw_summary.get("key_findings"))

    tldr = str(payload_summary.get("tldr") or "").strip()
    if not tldr:
        tldr = str(payload.get("tldr") or "").strip()
    if not tldr and key_findings:
        tldr = "; ".join(key_findings[:3]).strip()
    if not tldr:
        tldr = "No structured ablation findings available."

    return {
        "hypothesis_status": str(
            payload_summary.get("hypothesis_status")
            or payload.get("hypothesis_status")
            or "inconclusive"
        ).strip().lower() or "inconclusive",
        "feasible": feasible,
        "overall_confidence": _coerce_confidence(confidence_value, default=0.0),
        "tldr": tldr,
        "key_findings": key_findings,
    }


def _normalize_experiment_findings_payload(
    payload: Any,
    raw_ablation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    empty_payload = {
        "summary": {
            "hypothesis_status": "inconclusive",
            "feasible": None,
            "overall_confidence": 0.0,
            "tldr": "No structured ablation findings available.",
            "key_findings": [],
        },
        "component_findings": [],
    }
    if not isinstance(payload, dict):
        payload = {}

    raw_component_findings = normalize_ablation_results_payload(raw_ablation or {})
    raw_component_map = {
        item["component"]: {
            "component": item["component"],
            "result": item["result"],
            "metric": item["metric"],
            "value": item["value"],
            "confidence": item["confidence"],
            "analysis": item["analysis"],
        }
        for item in raw_component_findings
        if item.get("component")
    }

    llm_component_map: Dict[str, Dict[str, Any]] = {}
    for raw_item in payload.get("component_findings") or []:
        if not isinstance(raw_item, dict):
            continue
        component = str(raw_item.get("component") or "").strip()
        normalized = _normalize_component_finding_with_fallback(
            raw_item,
            fallback=raw_component_map.get(component),
        )
        if normalized is not None:
            llm_component_map[normalized["component"]] = normalized

    if raw_component_findings:
        component_findings = []
        for raw_item in raw_component_findings:
            component = raw_item.get("component")
            if not component:
                continue
            merged = llm_component_map.get(component)
            if merged is None:
                merged = _normalize_component_finding_with_fallback(
                    raw_item,
                    fallback=raw_component_map.get(component),
                )
            if merged is not None:
                component_findings.append(merged)
    else:
        component_findings = [
            item
            for item in (
                _normalize_component_finding_with_fallback(raw)
                for raw in (payload.get("component_findings") or [])
            )
            if item is not None
        ]

    return {
        "summary": _normalize_experiment_summary(payload, raw_ablation=raw_ablation),
        "component_findings": component_findings,
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
        return _normalize_experiment_findings_payload({}, raw_ablation=None)

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
        return _normalize_experiment_findings_payload(payload, raw_ablation=raw_ablation)
    except Exception as exc:
        if logger is not None:
            logger.warning("⚠️ Failed to extract experiment findings from raw ablation: %s", exc)
        return _normalize_experiment_findings_payload({}, raw_ablation=raw_ablation)


def paper_context_with_rag(entries: List[Dict[str, Any]], artifact: Dict[str, Any]) -> str:
    return format_paper_context_prompt_view(entries, artifact)


def get_paper_content(
    paper_id: str,
    include_markdown: bool,
    artifact: Dict[str, Any],
    paper_repository,
    logger,
) -> Dict[str, Any]:
    del include_markdown, paper_repository, logger
    if not paper_id:
        return {}
    reference_batches = artifact_get(artifact, "references", [])
    for batch in reversed(reference_batches):
        for reference in batch or []:
            if not isinstance(reference, dict):
                continue
            node_id = str(reference.get("node_id") or reference.get("paper_id") or "").strip()
            if node_id == paper_id:
                stored = dict(reference)
                stored["paper_id"] = node_id
                return stored
    stored: Dict[str, Any] = {"paper_id": paper_id}
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


def root_idea_to_mature_idea_text(root_idea: Dict[str, Any]) -> str:
    """Flatten a structured root idea into the text anchor used by mature_idea."""
    try:
        normalized = normalize_idea_contract(root_idea, allow_legacy=True, keep_extra=True)
    except Exception:
        return ""

    parts: List[str] = []
    seen = set()
    for raw in (
        normalized.get("abstract"),
        normalized.get("core_contribution"),
        normalized.get("method"),
    ):
        text = str(raw or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        if text[-1] not in ".!?":
            text += "."
        parts.append(text)

    if not parts:
        title = str(normalized.get("title") or "").strip()
        if not title:
            return ""
        return title if title[-1] in ".!?" else f"{title}."
    return " ".join(parts[:3]).strip()


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

    prompt = prompts["algorithm_structuring"].format(
        topic=topic,
        idea_title=idea.get("title", ""),
        idea_abstract=idea.get("abstract", ""),
        idea=json.dumps(idea_for_prompt, ensure_ascii=False, indent=2),
    )
    try:
        response = chat_fn(
            prompt,
            temperature=0.01,
            max_output_tokens=65536,
            model=model,
            stage="algorithm_structuring",
        )
        payload = parse_json_response(response)
        candidate = payload.get("algorithms", payload)
        if isinstance(candidate, list) and candidate:
            return align_algorithms_with_idea(
                idea, candidate, prompts, chat_fn, model, logger
            )
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Algorithm structuring failed: %s", exc)

    return fallback_algorithm_spec(idea)


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
        response = chat_fn(
            prompt,
            temperature=0.01,
            max_output_tokens=65536,
            model=model,
            stage="algorithm_alignment",
        )
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
