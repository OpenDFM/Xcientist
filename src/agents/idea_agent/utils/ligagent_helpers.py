from __future__ import annotations

import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Set

from src.agents.idea_agent.utils.idea_helpers import fallback_algorithm_spec, search_web
from src.agents.idea_agent.utils.graph_baseline_search import rank_method_paper_nodes_weighted
from src.agents.idea_agent.utils.ligagent_react_utils import react_websearch
from src.agents.idea_agent.utils.ligagent_suggestion_utils import (
    build_baseline_idea_card,
    build_baseline_seed_queries,
    build_dataset_idea_card,
    build_dataset_fallback_queries,
    build_dataset_seed_queries,
    build_datasetsearch_direct_candidates,
    collect_graph_baseline_candidates,
    collect_top_baseline_names_from_memory,
    collect_top_dataset_names_from_memory,
    dedupe_named,
    derive_label,
    extract_candidate_names,
    merge_dataset_candidates,
    merge_baseline_candidates,
    postprocess_suggestions,
    preprocess_candidate_names,
    score_dataset_candidate,
    score_graph_baseline_match,
    score_baseline_candidate,
    select_dataset_candidates,
    select_baseline_candidates,
)
from src.agents.idea_agent.utils.ligagent_utils import (
    enrich_papers_with_content,
    parse_json_response,
    paper_context_text,
    summarize_keynote,
)
from src.agents.idea_agent.utils.config_loader import get_config_value

_ARXIV_URL_RE = re.compile(r"https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/[^\s\]>\"')]+", re.IGNORECASE)
_GITHUB_URL_RE = re.compile(r"https?://github\.com/[^\s\]>\"')]+", re.IGNORECASE)


def sanitize_action_token(action: str) -> str:
    return "".join(ch for ch in action.lower().strip() if ch.isalnum())


def build_action_lookup(action_aliases: Dict[str, Set[str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for canonical, aliases in action_aliases.items():
        for alias in aliases:
            key = sanitize_action_token(alias)
            if key:
                lookup[key] = canonical
    return lookup


def _extract_first_url(pattern: re.Pattern[str], texts: List[str]) -> str:
    for text in texts:
        if not text:
            continue
        match = pattern.search(text)
        if match:
            return match.group(0)
    return ""


def _is_allowed_dataset_link(link: str) -> bool:
    lowered = (link or "").lower()
    return (
        "huggingface.co/datasets" in lowered
        or "github.com" in lowered
        or "paperswithcode.com/dataset" in lowered
        or "paperwithcodes.com/dataset" in lowered
    )


def _huggingface_search_link(query: str) -> str:
    safe = (query or "").strip().replace(" ", "%20")
    return f"https://huggingface.co/datasets?search={safe}"


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
            _join_list(payload.get("datasets"), "Datasets"),
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
    papers: List[Dict[str, Any]],
    paper_repository,
    logger,
    limit: int = 6,
) -> List[Dict[str, Any]]:
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


def generate_rag_query(
    topic: str,
    papers: List[Dict[str, Any]],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> str:
    prompt = prompts["rag_query"].format(
        topic=topic,
        papers=json.dumps(papers, ensure_ascii=False, indent=2),
    )
    try:
        response = chat_fn(prompt, model=model, temperature=0.3, max_tokens=512)
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


def retrieve_outcome_rag(query: str, paper_repository, logger) -> List[Dict[str, Any]]:
    try:
        hits = paper_repository.retrieve_outcome_rag(query=query, top_k=3)
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ OutcomeRAG retrieval failed: %s", exc)
        hits = []
    return hits


def collect_rag_citations(hits: List[Dict[str, Any]]) -> List[str]:
    titles: List[str] = []
    seen = set()
    for hit in hits or []:
        citations = hit.get("citations") or []
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
    memory: Dict[str, Any],
    reason: str,
) -> None:
    storage = memory.setdefault("paper_contents", {})
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
    memory: Dict[str, Any],
    logger,
) -> None:
    if not papers:
        return
    timeout = max(30, int(timeout or 0))
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        enrich_papers_with_content, papers, paper_repository, memory, logger
    )
    try:
        future.result(timeout=timeout)
    except FuturesTimeoutError:
        future.cancel()
        logger.warning(
            "⚠️ Paper enrichment exceeded %ss. Falling back to lightweight summaries.",
            timeout,
        )
        fallback_paper_summaries(papers, memory, reason="timeout_fallback")
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Paper enrichment failed: %s", exc)
        fallback_paper_summaries(papers, memory, reason="error_fallback")
    finally:
        executor.shutdown(wait=False)


def format_rag_context(memory: Dict[str, Any], max_hits: int = 5, max_chars: int = 320) -> str:
    rag_entries = memory.get("rag_hits", [])
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


def paper_context_with_rag(entries: List[Dict[str, Any]], memory: Dict[str, Any]) -> str:
    base = paper_context_text(entries)
    rag_context = format_rag_context(memory)
    if rag_context:
        return f"{base}\n\nRAG excerpts:\n{rag_context}"
    return base


def get_paper_content(
    paper_id: str,
    include_markdown: bool,
    memory: Dict[str, Any],
    paper_repository,
    logger,
) -> Dict[str, Any]:
    if not paper_id:
        return {}
    stored = memory.setdefault("paper_contents", {}).get(paper_id, {}).copy()
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


def ingest_analysis_background(analysis_entry: Dict[str, Any], memory: Dict[str, Any]) -> None:
    if not isinstance(analysis_entry, dict):
        return
    seeds = (
        analysis_entry.get("divergent_idea_seeds")
        or analysis_entry.get("moonshot_hypotheses")
        or []
    )
    if not isinstance(seeds, list) or not seeds:
        return
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
    if not background_lines:
        return
    background_store = memory.setdefault("background_knowledge", [])
    existing = set(background_store)
    for line in background_lines:
        if line and line not in existing:
            background_store.append(line)
            existing.add(line)


def latest_analysis_seed_ideas(memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not memory.get("analysis"):
        return []
    latest = memory["analysis"][-1]
    if not isinstance(latest, dict):
        return []
    seeds = latest.get("divergent_idea_seeds") or latest.get("moonshot_hypotheses") or []
    return convert_seeds_to_ideas(seeds)


def convert_seeds_to_ideas(seeds: Any) -> List[Dict[str, Any]]:
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
                "core_contribute": differentiator or hypothesis or method or "Analysis-seeded moonshot hypothesis.",
                "methodology": method or "Derived from divergent analysis seed; requires fleshing out.",
                "experiment_design": evaluation_plan or "Design ICML-grade evaluation including failure-surface probes.",
                "risks": risk or "High novelty risk; feasibility unknown.",
                "tags": tags,
                "operator": "analysis_seed",
                "target_defects": seed.get("target_defects", ["stagnant_novelty"]),
                "memory_refs": supporting if isinstance(supporting, list) else [str(supporting)],
                "rationale": differentiator or "Seed extracted from advanced analysis.",
            }
        )
    return payloads


def build_algorithm_spec(
    idea: Dict[str, Any],
    topic: str,
    raw_references: List[Dict[str, Any]],
    memory: Dict[str, Any],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> List[Dict[str, Any]]:
    analysis_entries = memory.get("analysis", [])
    latest_analysis = analysis_entries[-1] if analysis_entries else {}
    base_inputs: List[str] = []
    if topic and topic != "unspecified topic":
        base_inputs.append(f"Topic focus: {topic}")
    retrieval_history = memory.get("retrieval_keywords", [])
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
    core = idea.get("core_contribute") or idea.get("core_contribution")
    if core:
        base_outputs.append(f"Core contribution: {core}")
    methodology = idea.get("methodology") or idea.get("method")
    if methodology:
        base_outputs.append(f"Methodology: {methodology}")
    experiments = idea.get("experiment_design") or idea.get("experiments")
    if experiments:
        base_outputs.append(f"Experiment design: {experiments}")
    score = idea.get("search_score")
    if isinstance(score, (int, float)):
        base_outputs.append(f"MCTS search score: {score:.2f}")

    prompt = prompts["algorithm_structuring"].format(
        topic=topic,
        idea_title=idea.get("title", ""),
        idea_abstract=idea.get("abstract", ""),
        idea=json.dumps(idea, ensure_ascii=False, indent=2),
        base_inputs=json.dumps(base_inputs, ensure_ascii=False, indent=2),
        base_outputs=json.dumps(base_outputs, ensure_ascii=False, indent=2),
        analysis=json.dumps(latest_analysis, ensure_ascii=False, indent=2),
        references=json.dumps(raw_references[:5], ensure_ascii=False, indent=2),
    )
    prompt += "\n Directly output JSON."
    try:
        response = chat_fn(prompt, temperature=0.01, max_tokens=4096, model=model)
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
        response = chat_fn(prompt, temperature=0.01, max_tokens=2048, model=model)
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
        response = chat_fn(prompt, temperature=0.01, max_tokens=4096, model=model)
        payload = parse_json_response(response)
        candidate = payload.get("reference_papers", payload)
        if isinstance(candidate, list):
            return candidate
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Reference synthesis failed: %s", exc)
    return raw_references


def suggest_datasets(
    topic: str,
    best_entry: Dict[str, Any],
    algorithm: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
    memory: Optional[Dict[str, Any]] = None,
    config: Optional[object] = None,
) -> List[Dict[str, Any]]:
    llm_temperature = get_config_value(config, "dataset.llm.temperature", 0.1)
    idea_card_max_tokens = get_config_value(config, "dataset.llm.idea_card_max_tokens", 800)
    query_generation_max_tokens = get_config_value(
        config, "dataset.llm.query_generation_max_tokens", 512
    )
    name_extraction_max_tokens = get_config_value(
        config, "dataset.llm.name_extraction_max_tokens", 300
    )
    preprocess_max_tokens = get_config_value(config, "dataset.llm.preprocess_max_tokens", 400)
    postprocess_max_tokens = get_config_value(config, "dataset.llm.postprocess_max_tokens", 200)
    candidate_scoring_max_tokens = get_config_value(
        config, "dataset.llm.candidate_scoring_max_tokens", 700
    )
    memory_top_k = get_config_value(config, "dataset.memory_top_k", 5)
    preprocess_max_items = get_config_value(config, "dataset.preprocess_max_items", 5)
    seed_extract_max_names = get_config_value(config, "dataset.seed.extract_max_names", 8)
    seed_preprocess_max_items = get_config_value(config, "dataset.seed.preprocess_max_items", 5)
    seed_keep_names = get_config_value(config, "dataset.seed.keep_names", 3)
    react_max_steps = get_config_value(config, "dataset.react.max_steps", 5)
    react_max_urls = get_config_value(config, "dataset.react.max_urls", 2)
    react_browse_max_chars = get_config_value(config, "dataset.react.browse_max_chars", 18000)
    react_observation_limit = get_config_value(config, "dataset.react.observation_limit", 6)
    react_search_max_retry = get_config_value(
        config,
        "dataset.react.search_max_retry",
        get_config_value(config, "search.max_retry", 3),
    )
    react_llm_temperature = get_config_value(config, "dataset.react.llm.temperature", 0.1)
    react_step_max_tokens = get_config_value(config, "dataset.react.llm.step_max_tokens", 200)
    react_browse_max_tokens = get_config_value(
        config, "dataset.react.llm.browse_max_tokens", 700
    )
    search_max_retry = get_config_value(config, "search.max_retry", 3)
    primary_target = get_config_value(config, "dataset.selection.primary_target", 5)
    extra_target = get_config_value(config, "dataset.selection.extra_target", 3)
    output_limit = get_config_value(config, "dataset.selection.output_limit", 8)
    postprocess_max_keep = get_config_value(config, "dataset.selection.postprocess_max_keep", 8)
    dedupe_limit = get_config_value(config, "dataset.selection.dedupe_limit", 8)
    min_results = get_config_value(config, "dataset.selection.min_results", 2)
    score_fetch_max_chars = get_config_value(config, "dataset.scoring.fetch_max_chars", 2500)
    score_page_text_max_chars = get_config_value(
        config, "dataset.scoring.page_text_max_chars", 2000
    )
    score_evidence_max_chars = get_config_value(
        config, "dataset.scoring.evidence_max_chars", 3500
    )
    score_bonus_hf = get_config_value(config, "dataset.scoring.bonus.huggingface", 0.5)
    score_bonus_kaggle = get_config_value(config, "dataset.scoring.bonus.kaggle", 0.4)
    score_bonus_pwc = get_config_value(config, "dataset.scoring.bonus.paperswithcode", 0.4)
    score_bonus_datasetsearch = get_config_value(
        config, "dataset.scoring.bonus.datasetsearch", 0.2
    )
    idea_card = build_dataset_idea_card(
        topic=topic,
        best_entry=best_entry,
        references=references,
        prompts=prompts,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        temperature=llm_temperature,
        max_tokens=idea_card_max_tokens,
    )
    top_from_keynotes = collect_top_dataset_names_from_memory(memory, top_k=memory_top_k)
    top_from_keynotes = preprocess_candidate_names(
        "dataset",
        top_from_keynotes,
        topic,
        idea_card,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        max_items=preprocess_max_items,
        temperature=llm_temperature,
        max_tokens=preprocess_max_tokens,
    )
    primary_candidates: List[Dict[str, Any]] = []
    primary_browse: List[Dict[str, Any]] = []
    primary_found: Set[str] = set()
    if top_from_keynotes:
        primary_result = react_websearch(
            "dataset",
            top_from_keynotes,
            topic,
            idea_card,
            chat_fn=chat_fn,
            model=model,
            logger=logger,
            search_fn=search_web,
            max_steps=react_max_steps,
            max_urls=react_max_urls,
            browse_max_chars=react_browse_max_chars,
            observation_limit=react_observation_limit,
            search_max_retry=react_search_max_retry,
            llm_temperature=react_llm_temperature,
            llm_step_max_tokens=react_step_max_tokens,
            llm_browse_max_tokens=react_browse_max_tokens,
        )
        logger.info("🔎 Dataset websearch (primary) found names: %s", primary_result.get("found_names"))
        logger.info("🔎 Dataset websearch (primary) browse candidates: %s", primary_result.get("browse_candidates"))
        primary_text = primary_result.get("search_text", "")
        primary_candidates = merge_dataset_candidates(primary_text)
        primary_browse = primary_result.get("browse_candidates", [])
        primary_found = {name.lower() for name in primary_result.get("found_names", [])}
        if not primary_candidates:
            fallback_queries = build_dataset_fallback_queries(top_from_keynotes, topic=topic, idea_card=idea_card)
            fallback_payload = search_web(fallback_queries, max_retry=search_max_retry)
            fallback_text = fallback_payload.get("results", "")
            primary_candidates = merge_dataset_candidates(fallback_text)
        primary_candidates.extend(
            build_datasetsearch_direct_candidates(top_from_keynotes, topic=topic, idea_card=idea_card)
        )

    primary_scored: List[Dict[str, Any]] = []
    for cand in primary_candidates:
        primary_scored.append(
            score_dataset_candidate(
                idea_card,
                cand,
                chat_fn=chat_fn,
                model=model,
                logger=logger,
                temperature=llm_temperature,
                max_tokens=candidate_scoring_max_tokens,
                fetch_max_chars=score_fetch_max_chars,
                page_text_max_chars=score_page_text_max_chars,
                evidence_max_chars=score_evidence_max_chars,
                bonus_huggingface=score_bonus_hf,
                bonus_kaggle=score_bonus_kaggle,
                bonus_paperswithcode=score_bonus_pwc,
                bonus_datasetsearch=score_bonus_datasetsearch,
            )
        )
    primary_scored.sort(key=lambda c: c.get("total_score", 0), reverse=True)
    primary_selected = select_dataset_candidates(primary_scored, target=primary_target)
    if not primary_selected and top_from_keynotes:
        for name in top_from_keynotes:
            query_name = name.strip()
            if not query_name:
                continue
            link = f"https://datasetsearch.research.google.com/search?query={query_name.replace(' ', '%20')}"
            primary_selected.append(
                {
                    "title": query_name,
                    "url": link,
                    "snippet": "Mentioned in paper keynote data; requires verification.",
                }
            )

    seed_queries = build_dataset_seed_queries(
        topic=topic,
        best_entry=best_entry,
        idea_card=idea_card,
        references=references,
    )
    seed_payload = search_web(seed_queries, max_retry=search_max_retry)
    seed_text = seed_payload.get("results", "")

    dataset_names = extract_candidate_names(
        "dataset",
        idea_card,
        seed_text,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        max_names=seed_extract_max_names,
        temperature=llm_temperature,
        max_tokens=name_extraction_max_tokens,
    )
    dataset_names = preprocess_candidate_names(
        "dataset",
        dataset_names,
        topic,
        idea_card,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        max_items=seed_preprocess_max_items,
        temperature=llm_temperature,
        max_tokens=preprocess_max_tokens,
    )
    keynote_names_lower = {name.lower() for name in top_from_keynotes}
    avoid_names = set(keynote_names_lower)
    if primary_found:
        avoid_names.update(primary_found)
    dataset_names = [
        name for name in dataset_names if name.lower() not in avoid_names
    ][:seed_keep_names]

    extra_candidates: List[Dict[str, Any]] = []
    extra_browse: List[Dict[str, Any]] = []
    if dataset_names:
        extra_result = react_websearch(
            "dataset",
            dataset_names,
            topic,
            idea_card,
            chat_fn=chat_fn,
            model=model,
            logger=logger,
            search_fn=search_web,
            max_steps=react_max_steps,
            max_urls=react_max_urls,
            browse_max_chars=react_browse_max_chars,
            observation_limit=react_observation_limit,
            search_max_retry=react_search_max_retry,
            llm_temperature=react_llm_temperature,
            llm_step_max_tokens=react_step_max_tokens,
            llm_browse_max_tokens=react_browse_max_tokens,
        )
        logger.info("🔎 Dataset websearch (fallback) found names: %s", extra_result.get("found_names"))
        logger.info("🔎 Dataset websearch (fallback) browse candidates: %s", extra_result.get("browse_candidates"))
        search_text = extra_result.get("search_text", "")
        extra_candidates = merge_dataset_candidates(search_text)
        extra_browse = extra_result.get("browse_candidates", [])
        if not extra_candidates:
            fallback_queries = build_dataset_fallback_queries(dataset_names, topic=topic, idea_card=idea_card)
            fallback_payload = search_web(fallback_queries, max_retry=search_max_retry)
            fallback_text = fallback_payload.get("results", "")
            extra_candidates = merge_dataset_candidates(fallback_text)
        extra_candidates.extend(
            build_datasetsearch_direct_candidates(dataset_names, topic=topic, idea_card=idea_card)
        )

    extra_scored: List[Dict[str, Any]] = []
    for cand in extra_candidates:
        extra_scored.append(
            score_dataset_candidate(
                idea_card,
                cand,
                chat_fn=chat_fn,
                model=model,
                logger=logger,
                temperature=llm_temperature,
                max_tokens=candidate_scoring_max_tokens,
                fetch_max_chars=score_fetch_max_chars,
                page_text_max_chars=score_page_text_max_chars,
                evidence_max_chars=score_evidence_max_chars,
                bonus_huggingface=score_bonus_hf,
                bonus_kaggle=score_bonus_kaggle,
                bonus_paperswithcode=score_bonus_pwc,
                bonus_datasetsearch=score_bonus_datasetsearch,
            )
        )
    extra_scored.sort(key=lambda c: c.get("total_score", 0), reverse=True)
    extra_selected = select_dataset_candidates(extra_scored, target=extra_target)

    datasets: List[Dict[str, Any]] = []
    for cand in primary_browse + extra_browse:
        name = cand.get("dataset_name") or ""
        access = cand.get("access_link") or ""
        if not name or not access or not _is_allowed_dataset_link(access):
            continue
        evidence = cand.get("evidence_snippets") or []
        usage = ""
        if evidence:
            usage = str(evidence[0])[:200]
        datasets.append(
            {
                "name": name,
                "source_paper": "browse",
                "usage": usage or "Extracted from browse results.",
                "access": access,
                "evidence": evidence,
                "link": access,
                "scores": {},
            }
        )
    for cand in primary_selected[:primary_target]:
        link = cand.get("access") or cand.get("url") or ""
        if not _is_allowed_dataset_link(link):
            continue
        datasets.append(
            {
                "name": cand.get("dataset_name") or cand.get("title") or derive_label(cand.get("title", "")),
                "source_paper": "websearch",
                "usage": cand.get("usage") or cand.get("snippet", "")[:200],
                "access": link or "websearch",
                "evidence": cand.get("evidence_snippets") or [],
                "link": link or "",
                "scores": {
                    "match": cand.get("match_score"),
                    "scale": cand.get("scale_score"),
                    "availability": cand.get("availability_score"),
                },
            }
        )
    for cand in extra_selected[:extra_target]:
        link = cand.get("access") or cand.get("url") or ""
        if not _is_allowed_dataset_link(link):
            continue
        datasets.append(
            {
                "name": cand.get("dataset_name") or cand.get("title") or derive_label(cand.get("title", "")),
                "source_paper": "websearch",
                "usage": cand.get("usage") or cand.get("snippet", "")[:200],
                "access": link or "websearch",
                "evidence": cand.get("evidence_snippets") or [],
                "link": link or "",
                "scores": {
                    "match": cand.get("match_score"),
                    "scale": cand.get("scale_score"),
                    "availability": cand.get("availability_score"),
                },
            }
        )

    datasets = dedupe_named(datasets, "name", limit=dedupe_limit)
    datasets = postprocess_suggestions(
        "dataset",
        datasets,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        max_keep=postprocess_max_keep,
        temperature=llm_temperature,
        max_tokens=postprocess_max_tokens,
    )
    datasets = [d for d in datasets if _is_allowed_dataset_link(d.get("link") or d.get("access") or "")]

    if len(datasets) < min_results:
        while len(datasets) < min_results:
            idx = len(datasets) + 1
            query_name = f"{topic or 'Target'} dataset candidate {idx}"
            datasets.append(
                {
                    "name": query_name,
                    "source_paper": "websearch",
                    "usage": "Requires validation; suggested due to sparse evidence.",
                    "access": _huggingface_search_link(query_name),
                    "link": _huggingface_search_link(query_name),
                }
            )
    return datasets[:output_limit]


def suggest_baselines(
    topic: str,
    best_entry: Dict[str, Any],
    algorithm: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
    memory: Optional[Dict[str, Any]] = None,
    config: Optional[object] = None,
) -> List[Dict[str, Any]]:
    llm_temperature = get_config_value(config, "baseline.llm.temperature", 0.1)
    idea_card_max_tokens = get_config_value(config, "baseline.llm.idea_card_max_tokens", 1200)
    query_generation_max_tokens = get_config_value(
        config, "baseline.llm.query_generation_max_tokens", 512
    )
    name_extraction_max_tokens = get_config_value(
        config, "baseline.llm.name_extraction_max_tokens", 300
    )
    preprocess_max_tokens = get_config_value(config, "baseline.llm.preprocess_max_tokens", 400)
    postprocess_max_tokens = get_config_value(config, "baseline.llm.postprocess_max_tokens", 200)
    graph_match_max_tokens = get_config_value(config, "baseline.llm.graph_match_max_tokens", 200)
    candidate_scoring_max_tokens = get_config_value(
        config, "baseline.llm.candidate_scoring_max_tokens", 800
    )
    memory_top_k = get_config_value(config, "baseline.memory_top_k", 5)
    preprocess_max_items = get_config_value(config, "baseline.preprocess_max_items", 5)
    seed_extract_max_names = get_config_value(config, "baseline.seed.extract_max_names", 8)
    seed_preprocess_max_items = get_config_value(config, "baseline.seed.preprocess_max_items", 6)
    graph_top_k = get_config_value(config, "baseline.graph.top_k", 20)
    graph_degree_weight = get_config_value(config, "baseline.graph.degree_weight", 0.05)
    graph_similarity_weight = get_config_value(config, "baseline.graph.similarity_weight", 0.95)
    graph_repo_search_max_retry = get_config_value(config, "baseline.graph.repo_search_max_retry", 3)
    react_max_steps = get_config_value(config, "baseline.react.max_steps", 5)
    react_max_urls = get_config_value(config, "baseline.react.max_urls", 2)
    react_browse_max_chars = get_config_value(config, "baseline.react.browse_max_chars", 18000)
    react_observation_limit = get_config_value(config, "baseline.react.observation_limit", 6)
    react_search_max_retry = get_config_value(
        config,
        "baseline.react.search_max_retry",
        get_config_value(config, "search.max_retry", 3),
    )
    react_llm_temperature = get_config_value(config, "baseline.react.llm.temperature", 0.1)
    react_step_max_tokens = get_config_value(config, "baseline.react.llm.step_max_tokens", 200)
    react_browse_max_tokens = get_config_value(config, "baseline.react.llm.browse_max_tokens", 700)
    search_max_retry = get_config_value(config, "search.max_retry", 3)
    selection_target = get_config_value(config, "baseline.selection.target", 8)
    selection_combined_limit = get_config_value(config, "baseline.selection.combined_limit", 5)
    selection_output_limit = get_config_value(config, "baseline.selection.output_limit", 5)
    postprocess_max_keep = get_config_value(config, "baseline.selection.postprocess_max_keep", 5)
    dedupe_limit = get_config_value(config, "baseline.selection.dedupe_limit", 5)
    score_fetch_max_chars = get_config_value(config, "baseline.scoring.fetch_max_chars", 3500)
    score_page_text_max_chars = get_config_value(
        config, "baseline.scoring.page_text_max_chars", 2000
    )
    score_evidence_max_chars = get_config_value(
        config, "baseline.scoring.evidence_max_chars", 3500
    )
    score_bonus_github = get_config_value(config, "baseline.scoring.github_bonus", 0.5)
    score_bonus_arxiv = get_config_value(config, "baseline.scoring.arxiv_bonus", 0.3)
    score_bonus_evidence_github = get_config_value(
        config, "baseline.scoring.evidence_github_bonus", 0.2
    )
    score_missing_penalty = get_config_value(
        config, "baseline.scoring.missing_link_penalty", -0.8
    )
    graph_task = None
    if memory:
        graph_task = memory.get("run_topic") or None
    if not graph_task:
        graph_task = os.environ.get("IDEA_AGENT_TASK_TOPIC") or None
    if not graph_task:
        graph_task = topic
    idea_card = build_baseline_idea_card(
        topic=topic,
        best_entry=best_entry,
        algorithm=algorithm,
        references=references,
        prompts=prompts,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        temperature=llm_temperature,
        max_tokens=idea_card_max_tokens,
    )
    graph_ranked = rank_method_paper_nodes_weighted(
        topic=graph_task,
        top_k=graph_top_k,
        degree_weight=graph_degree_weight,
        similarity_weight=graph_similarity_weight,
    )
    if graph_ranked:
        logger.info("🧭 Graph task: %s", graph_task)
        logger.info(
            "🧭 Graph top paper titles: %s",
            ", ".join(item.get("paper_title", "") for item in graph_ranked if item.get("paper_title")),
        )
        logger.info(
            "🧭 Graph scores: %s",
            "; ".join(
                f"{item.get('paper_title')} (degree={item.get('degree_score'):.3f}, sim={item.get('similarity_score'):.3f}, w={item.get('score'):.3f})"
                for item in graph_ranked
                if item.get("paper_title")
            ),
        )
    graph_queries = [item.get("paper_title") for item in graph_ranked if item.get("paper_title")]
    if graph_queries:
        repo_candidates = collect_graph_baseline_candidates(
            graph_queries,
            search_fn=search_web,
            logger=logger,
            max_retry=graph_repo_search_max_retry,
        )
        repo_map = {cand.get("name", "").lower(): cand for cand in repo_candidates if cand.get("repo_url")}
        logger.info("🧭 Graph GitHub candidates: %s", [c.get("repo_url") for c in repo_candidates])
        filtered_nodes: List[Dict[str, Any]] = []
        for item in graph_ranked:
            query = (item.get("paper_title") or "").lower()
            repo = repo_map.get(query)
            if not repo:
                continue
            merged = dict(item)
            merged["repo_url"] = repo.get("repo_url")
            merged["evidence"] = repo.get("evidence", [])
            merged["usage"] = repo.get("usage", "")
            merged["link"] = repo.get("repo_url")
            filtered_nodes.append(merged)
        logger.info("🧭 Graph nodes with GitHub: %s", len(filtered_nodes))
        if filtered_nodes:
            for item in filtered_nodes:
                node_fields = {
                    "keywords": item.get("keywords", ""),
                    "problem": item.get("problem", ""),
                    "innovation": item.get("innovation", ""),
                    "scenarios": item.get("scenarios", ""),
                }
                item["match_score"] = score_graph_baseline_match(
                    idea_card=idea_card,
                    node_fields=node_fields,
                    chat_fn=chat_fn,
                    model=model,
                    logger=logger,
                    temperature=llm_temperature,
                    max_tokens=graph_match_max_tokens,
                )
            filtered_nodes.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            logger.info(
                "🧭 Graph rerank scores: %s",
                "; ".join(
                    f"{item.get('paper_title')} ({item.get('match_score'):.1f})"
                    for item in filtered_nodes[:selection_output_limit]
                ),
            )
            baselines: List[Dict[str, Any]] = []
            for item in filtered_nodes[:selection_output_limit]:
                name = item.get("paper_title") or item.get("title") or ""
                if not name or not item.get("repo_url"):
                    continue
                baselines.append(
                    {
                        "name": name,
                        "source": "paper_graph",
                        "repo_url": item.get("repo_url", ""),
                        "usage": item.get("innovation")
                        or item.get("problem")
                        or item.get("usage", "")
                        or "Matched via graph rerank.",
                        "evidence": item.get("evidence", []),
                        "link": item.get("link") or item.get("repo_url", ""),
                        "scores": {
                            "graph_degree": item.get("degree_score"),
                            "graph_similarity": item.get("similarity_score"),
                            "graph_weighted": item.get("score"),
                            "match": item.get("match_score"),
                        },
                    }
                )
            if baselines:
                return baselines
    top_from_keynotes = collect_top_baseline_names_from_memory(memory, top_k=memory_top_k)
    top_from_keynotes = preprocess_candidate_names(
        "baseline",
        top_from_keynotes,
        topic,
        idea_card,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        max_items=preprocess_max_items,
        temperature=llm_temperature,
        max_tokens=preprocess_max_tokens,
    )
    primary_candidates: List[Dict[str, Any]] = []
    primary_browse: List[Dict[str, Any]] = []
    primary_found: Set[str] = set()
    if top_from_keynotes:
        primary_result = react_websearch(
            "baseline",
            top_from_keynotes,
            topic,
            idea_card,
            chat_fn=chat_fn,
            model=model,
            logger=logger,
            search_fn=search_web,
            max_steps=react_max_steps,
            max_urls=react_max_urls,
            browse_max_chars=react_browse_max_chars,
            observation_limit=react_observation_limit,
            search_max_retry=react_search_max_retry,
            llm_temperature=react_llm_temperature,
            llm_step_max_tokens=react_step_max_tokens,
            llm_browse_max_tokens=react_browse_max_tokens,
        )
        logger.info("🔎 React websearch (primary) found names: %s", primary_result.get("found_names"))
        logger.info("🔎 React websearch (primary) browse candidates: %s", primary_result.get("browse_candidates"))
        primary_text = primary_result.get("search_text", "")
        primary_candidates = merge_baseline_candidates(primary_text)
        primary_browse = primary_result.get("browse_candidates", [])
        primary_found = {name.lower() for name in primary_result.get("found_names", [])}

    primary_scored: List[Dict[str, Any]] = []
    for cand in primary_candidates:
        primary_scored.append(
            score_baseline_candidate(
                idea_card,
                cand,
                chat_fn=chat_fn,
                model=model,
                logger=logger,
                temperature=llm_temperature,
                max_tokens=candidate_scoring_max_tokens,
                fetch_max_chars=score_fetch_max_chars,
                page_text_max_chars=score_page_text_max_chars,
                evidence_max_chars=score_evidence_max_chars,
                bonus_github=score_bonus_github,
                bonus_arxiv=score_bonus_arxiv,
                bonus_evidence_github=score_bonus_evidence_github,
                missing_link_penalty=score_missing_penalty,
            )
        )
    primary_scored.sort(key=lambda c: c.get("total_score", 0), reverse=True)

    def _has_github(candidate: Dict[str, Any]) -> bool:
        url = (candidate.get("url") or "").lower()
        if "github.com" in url:
            return True
        evidence = candidate.get("evidence_snippets") or []
        if any("github.com" in str(item).lower() for item in evidence):
            return True
        snippet = (candidate.get("snippet") or "").lower()
        return "github.com" in snippet

    primary_selected = [cand for cand in primary_scored if _has_github(cand)]

    seed_queries = build_baseline_seed_queries(
        topic=topic,
        best_entry=best_entry,
        idea_card=idea_card,
        references=references,
    )
    seed_payload = search_web(seed_queries, max_retry=search_max_retry)
    seed_text = seed_payload.get("results", "")

    baseline_names = extract_candidate_names(
        "baseline",
        idea_card,
        seed_text,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        max_names=seed_extract_max_names,
        temperature=llm_temperature,
        max_tokens=name_extraction_max_tokens,
    )
    baseline_names = preprocess_candidate_names(
        "baseline",
        baseline_names,
        topic,
        idea_card,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        max_items=seed_preprocess_max_items,
        temperature=llm_temperature,
        max_tokens=preprocess_max_tokens,
    )
    if primary_found:
        baseline_names = [name for name in baseline_names if name.lower() not in primary_found]
    search_result = react_websearch(
        "baseline",
        baseline_names,
        topic,
        idea_card,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        search_fn=search_web,
        max_steps=react_max_steps,
        max_urls=react_max_urls,
        browse_max_chars=react_browse_max_chars,
        observation_limit=react_observation_limit,
        search_max_retry=react_search_max_retry,
        llm_temperature=react_llm_temperature,
        llm_step_max_tokens=react_step_max_tokens,
        llm_browse_max_tokens=react_browse_max_tokens,
    )
    logger.info("🔎 React websearch (fallback) found names: %s", search_result.get("found_names"))
    logger.info("🔎 React websearch (fallback) browse candidates: %s", search_result.get("browse_candidates"))
    search_text = search_result.get("search_text", "")
    extra_browse = search_result.get("browse_candidates", [])
    candidates = merge_baseline_candidates(search_text)
    scored: List[Dict[str, Any]] = []
    for cand in candidates:
        scored.append(
            score_baseline_candidate(
                idea_card,
                cand,
                chat_fn=chat_fn,
                model=model,
                logger=logger,
                temperature=llm_temperature,
                max_tokens=candidate_scoring_max_tokens,
                fetch_max_chars=score_fetch_max_chars,
                page_text_max_chars=score_page_text_max_chars,
                evidence_max_chars=score_evidence_max_chars,
                bonus_github=score_bonus_github,
                bonus_arxiv=score_bonus_arxiv,
                bonus_evidence_github=score_bonus_evidence_github,
                missing_link_penalty=score_missing_penalty,
            )
        )
    scored.sort(key=lambda c: c.get("total_score", 0), reverse=True)

    selected = select_baseline_candidates(scored, idea_card, target=selection_target)
    fallback_selected = [cand for cand in selected if _has_github(cand)]

    combined: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    for cand in primary_selected + fallback_selected:
        key = (cand.get("url") or cand.get("title") or "").lower().strip()
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        combined.append(cand)
        if len(combined) >= selection_combined_limit:
            break

    if not combined:
        combined = fallback_selected[:selection_combined_limit]
    if not combined:
        combined = primary_selected[:selection_combined_limit]
    if not combined:
        combined = selected[:selection_combined_limit]

    baselines: List[Dict[str, Any]] = []
    for cand in primary_browse + extra_browse:
        name = cand.get("paper_title") or ""
        arxiv = cand.get("arxiv_link") or ""
        github = cand.get("github_link") or ""
        if not name or not arxiv or not github:
            continue
        evidence = cand.get("evidence_snippets") or []
        usage = ""
        if evidence:
            usage = str(evidence[0])[:200]
        baselines.append(
            {
                "name": name,
                "source": "browse",
                "repo_url": github,
                "usage": usage or "Extracted from browse results.",
                "evidence": evidence,
                "link": arxiv or github,
                "scores": {},
            }
        )
    for cand in combined[:selection_output_limit]:
        evidence = cand.get("evidence_snippets") or []
        texts = [
            cand.get("url") or "",
            cand.get("snippet") or "",
            cand.get("usage") or "",
            cand.get("setting") or "",
        ]
        for item in evidence:
            texts.append(str(item))

        arxiv_link = ""
        repo_url = ""
        url = cand.get("url") or ""
        if "arxiv.org" in url or "export.arxiv.org" in url:
            arxiv_link = url
        if "github.com" in url:
            repo_url = url
        if not arxiv_link:
            arxiv_link = _extract_first_url(_ARXIV_URL_RE, texts)
        if not repo_url:
            repo_url = _extract_first_url(_GITHUB_URL_RE, texts)

        title = cand.get("title") or derive_label(cand.get("title", ""))
        if not title or not arxiv_link or not repo_url:
            continue

        baselines.append(
            {
                "name": title,
                "source": "websearch",
                "repo_url": repo_url,
                "usage": cand.get("usage")
                or cand.get("setting")
                or cand.get("snippet", "")[:200],
                "evidence": evidence,
                "link": arxiv_link,
                "scores": {
                    "match": cand.get("match_score"),
                    "representativeness": cand.get("representativeness_score"),
                    "reproducibility": cand.get("reproducibility_score"),
                },
            }
        )

    pre_post_baselines = baselines[:]
    baselines = postprocess_suggestions(
        "baseline",
        baselines,
        chat_fn=chat_fn,
        model=model,
        logger=logger,
        max_keep=postprocess_max_keep,
        temperature=llm_temperature,
        max_tokens=postprocess_max_tokens,
    )
    baselines = [b for b in baselines if (b.get("name") and b.get("repo_url"))]
    if not baselines:
        baselines = [
            b
            for b in pre_post_baselines
            if (b.get("name") and b.get("repo_url"))
        ][:selection_output_limit]

    baselines = dedupe_named(baselines, "name", limit=dedupe_limit)
    return baselines[:selection_output_limit]
