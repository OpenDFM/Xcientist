from __future__ import annotations

import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional, Set

from src.agents.idea_agent.utils.idea_helpers import (
    fallback_algorithm_spec,
    parse_search_results,
    search_web,
)
from src.agents.idea_agent.utils.ligagent_utils import (
    enrich_papers_with_content,
    parse_json_response,
    paper_context_text,
    summarize_keynote,
)


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
) -> List[Dict[str, Any]]:
    def _dedupe_named(entries: List[Dict[str, Any]], name_key: str) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        output: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = (entry.get(name_key) or entry.get("title") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            entry[name_key] = name
            output.append(entry)
            if len(output) >= 5:
                break
        return output

    def _derive_label(title: str) -> str:
        cleaned = (title or "").strip()
        for sep in (" - ", " | ", " — ", " – ", ":"):
            if sep in cleaned:
                cleaned = cleaned.split(sep, 1)[0].strip()
        return cleaned

    def _fallback_from_search(search_text: str, max_items: int) -> List[Dict[str, Any]]:
        results = parse_search_results(search_text)
        fallback: List[Dict[str, Any]] = []
        for item in results:
            name = _derive_label(item.get("title", ""))
            if not name:
                continue
            snippet = (item.get("snippet") or "").strip()
            fallback.append(
                {
                    "name": name,
                    "source_paper": "websearch",
                    "usage": snippet[:200] if snippet else "Referenced in web search results.",
                    "access": item.get("url") or "websearch",
                }
            )
            if len(fallback) >= max_items:
                break
        return fallback

    def _trim_query_context(text: str, max_words: int = 10) -> str:
        if not text:
            return ""
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\\-]+", text)
        if not words:
            return ""
        return " ".join(words[:max_words])

    idea_title = best_entry.get("title", "")
    idea_context = _trim_query_context(
        best_entry.get("abstract") or best_entry.get("core_contribute") or ""
    )
    queries: List[str] = []
    if idea_title:
        queries.append(f"\"{idea_title}\" dataset")
    if topic and topic != "unspecified topic":
        queries.append(f"{topic} dataset benchmark")
        queries.append(f"{topic} dataset")
    if idea_context:
        queries.append(f"{topic} {idea_context} dataset" if topic else f"{idea_context} dataset")
    for ref in references[:3]:
        ref_title = (ref.get("title") or "").strip()
        if ref_title:
            queries.append(f"\"{ref_title}\" dataset")
    # Deduplicate queries while preserving order.
    seen_queries: Set[str] = set()
    deduped_queries: List[str] = []
    for query in queries:
        if query and query not in seen_queries:
            deduped_queries.append(query)
            seen_queries.add(query)
    search_payload = search_web(deduped_queries[:5], max_retry=3)
    search_text = search_payload.get("results", "")

    datasets = _fallback_from_search(search_text, max_items=5)
    datasets = _dedupe_named(datasets, "name")

    if len(datasets) < 2:
        while len(datasets) < 2:
            idx = len(datasets) + 1
            datasets.append(
                {
                    "name": f"{topic or 'Target'} dataset candidate {idx}",
                    "source_paper": "websearch",
                    "usage": "Requires validation; suggested due to sparse evidence.",
                    "access": "websearch",
                }
            )
    return datasets[:5]


def suggest_baselines(
    topic: str,
    best_entry: Dict[str, Any],
    algorithm: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> List[Dict[str, Any]]:
    def _dedupe_named(entries: List[Dict[str, Any]], name_key: str) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        output: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = (entry.get(name_key) or entry.get("title") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            entry[name_key] = name
            output.append(entry)
            if len(output) >= 5:
                break
        return output

    def _derive_label(title: str) -> str:
        cleaned = (title or "").strip()
        for sep in (" - ", " | ", " — ", " – ", ":"):
            if sep in cleaned:
                cleaned = cleaned.split(sep, 1)[0].strip()
        return cleaned

    def _fallback_from_search(search_text: str, max_items: int) -> List[Dict[str, Any]]:
        results = parse_search_results(search_text)
        github_first = [r for r in results if "github.com" in (r.get("url") or "")]
        others = [r for r in results if r not in github_first]
        ordered = github_first + others
        fallback: List[Dict[str, Any]] = []
        for item in ordered:
            name = _derive_label(item.get("title", ""))
            if not name:
                continue
            url = item.get("url") or ""
            snippet = (item.get("snippet") or "").strip()
            fallback.append(
                {
                    "name": name,
                    "source": "websearch",
                    "repo_url": url if "github.com" in url else "",
                    "usage": snippet[:200] if snippet else "Referenced in web search results.",
                }
            )
            if len(fallback) >= max_items:
                break
        return fallback

    def _trim_query_context(text: str, max_words: int = 10) -> str:
        if not text:
            return ""
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\\-]+", text)
        if not words:
            return ""
        return " ".join(words[:max_words])

    idea_title = best_entry.get("title", "")
    idea_context = _trim_query_context(
        best_entry.get("abstract") or best_entry.get("core_contribute") or ""
    )
    queries: List[str] = []
    if idea_title:
        queries.append(f"\"{idea_title}\" baseline github")
        queries.append(f"\"{idea_title}\" implementation github")
    if topic and topic != "unspecified topic":
        queries.append(f"{topic} baseline github")
        queries.append(f"{topic} state of the art baseline github")
    if idea_context:
        queries.append(
            f"{topic} {idea_context} baseline github" if topic else f"{idea_context} baseline github"
        )
    for ref in references[:3]:
        ref_title = (ref.get("title") or "").strip()
        if ref_title:
            queries.append(f"\"{ref_title}\" baseline github")
    seen_queries: Set[str] = set()
    deduped_queries: List[str] = []
    for query in queries:
        if query and query not in seen_queries:
            deduped_queries.append(query)
            seen_queries.add(query)
    search_payload = search_web(deduped_queries[:6], max_retry=3)
    search_text = search_payload.get("results", "")

    baselines = _fallback_from_search(search_text, max_items=5)
    baselines = _dedupe_named(baselines, "name")

    if len(baselines) < 2:
        while len(baselines) < 2:
            idx = len(baselines) + 1
            baselines.append(
                {
                    "name": f"{topic or 'Target'} baseline candidate {idx}",
                    "source": "websearch",
                    "repo_url": "",
                    "usage": "Requires validation; suggested due to sparse evidence.",
                }
            )
    return baselines[:5]
