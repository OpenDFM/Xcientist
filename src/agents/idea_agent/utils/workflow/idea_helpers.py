"""Workflow helpers for idea traces, fallback specs, and lightweight web search."""

import logging
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
import requests

from src.agents.idea_agent.utils.workflow.idea_contract import normalize_idea_contract

logger = logging.getLogger(__name__)

def build_mcts_evolution(best_entry: Dict[str, Any]) -> Dict[str, Any]:
    trace = best_entry.get("search_trace") or []
    iterations: List[Dict[str, Any]] = []
    for hop in trace:
        if not isinstance(hop, dict):
            continue
        entry = {
            "iteration": hop.get("iteration"),
            "node_id": hop.get("node_id"),
            "depth": hop.get("depth"),
            "title": hop.get("title"),
            "operator": hop.get("operator"),
            "defects": hop.get("defects"),
            "score": hop.get("score"),
            "visits": hop.get("visits"),
            "path": hop.get("path"),
            "action_summary": hop.get("action_summary"),
        }
        evaluation = hop.get("evaluation")
        if evaluation is not None:
            entry["evaluation"] = evaluation
        memory_refs = hop.get("memory_refs")
        if memory_refs:
            entry["memory_refs"] = memory_refs
        rationale = hop.get("rationale")
        if rationale:
            entry["rationale"] = rationale
        signature = hop.get("signature")
        if signature:
            entry["signature"] = signature
        iterations.append(entry)
    evolution = {
        "best_path": best_entry.get("search_path"),
        "best_operator": best_entry.get("operator"),
        "target_defects": best_entry.get("target_defects"),
        "iterations": iterations,
    }
    pareto = best_entry.get("pareto_candidates")
    if pareto:
        evolution["pareto_front"] = pareto
    return evolution


def build_fusion_evolution(best_entry: Dict[str, Any]) -> Dict[str, Any]:
    fusion_metadata = best_entry.get("fusion_metadata")
    if not isinstance(fusion_metadata, dict):
        fusion_metadata = {}

    return {
        "source_modes": best_entry.get("source_modes") or [],
        "host_idea_mode": fusion_metadata.get("host_idea_mode"),
        "selected_components": fusion_metadata.get("selected_components") or [],
        "rejected_components": fusion_metadata.get("rejected_components") or [],
        "conflicts_and_resolutions": fusion_metadata.get("conflicts_and_resolutions") or [],
        "fused_core_thesis": fusion_metadata.get("fused_core_thesis") or "",
        "why_stronger_than_each_input": fusion_metadata.get("why_stronger_than_each_input") or "",
        "minimal_validation_plan": fusion_metadata.get("minimal_validation_plan") or "",
        "post_fusion_evaluation": build_mcts_evolution(best_entry),
    }


def collect_reference_material(reference_batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    seen_titles = set()
    for batch in reference_batches or []:
        for paper in batch:
            if not isinstance(paper, dict):
                continue
            title = (paper.get("title") or "").strip()
            if not title:
                continue
            key = title.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            entry = {
                "title": title,
                "authors": paper.get("authors") or [],
                "abstract": paper.get("abstract"),
                "tldr": paper.get("tldr"),
                "url": paper.get("url"),
                "year": paper.get("year"),
                "paper_id": paper.get("paper_id"),
                "source_keywords": paper.get("source_keywords"),
            }
            references.append(entry)
    return references


def derive_pipeline_steps(idea: Dict[str, Any]) -> List[str]:
    idea = normalize_idea_contract(idea, keep_extra=True)
    sections = [
        idea.get("method"),
        idea.get("experiments"),
        idea.get("abstract"),
        idea.get("core_contribution"),
    ]
    sentences: List[str] = []
    for section in sections:
        if not section:
            continue
        chunks = re.split(r"(?<=[.;])\s+", section)
        for chunk in chunks:
            cleaned = chunk.strip(" .;\n")
            if cleaned:
                sentences.append(cleaned)
            if len(sentences) >= 6:
                break
        if len(sentences) >= 3:
            break
    if not sentences:
        sentences = ["Outline the proposed method using available context."]
    return [f"Step {idx + 1}: {sentence}" for idx, sentence in enumerate(sentences)]


def fallback_algorithm_spec(idea: Dict[str, Any], inputs: List[str], outputs: List[str]) -> List[Dict[str, Any]]:
    pipeline = derive_pipeline_steps(idea)
    algorithm_entry = {
        "name": idea.get("title") or "Research Algorithm",
        "input": inputs,
        "output": outputs,
        "pipeline": pipeline,
    }
    return [algorithm_entry]


def search_google(query: str, num: int = 10) -> List[Dict[str, Any]]:
    SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
    SERPER_API_ENDPOINT = os.environ.get("SERPER_API_ENDPOINT")
    if not query.strip():
        return []
    key = SERPER_API_KEY
    if not key:
        raise RuntimeError("SERPER_API_KEY is not configured")
    headers = {
        "X-API-KEY": key,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": num}
    response = requests.post(
        SERPER_API_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json() or {}
    return data.get("organic", []) or []


def get_brief_text(contents: List[Dict[str, Any]]) -> str:
    source_text = ""
    for content_json in contents:
        if not isinstance(content_json, dict):
            continue
        if "extra_snippets" in content_json and content_json.get("extra_snippets"):
            snippet = "\n".join(content_json["extra_snippets"])
        elif "snippet" in content_json:
            snippet = content_json["snippet"]
        else:
            snippet = content_json.get("description", "")
        title = content_json.get("title", "").strip()
        url = content_json.get("url") or content_json.get("link", "")
        source_text += (
            f"<title>{title}</title>\n"
            f"<url>{url}</url>\n"
            f"<snippet>\n{snippet}\n</snippet>\n\n"
        )
    return source_text.strip()


def get_search_results(query: str, max_retry: int = 3) -> str:
    source_text = "Search result is empty. Please try again."
    if not query.strip():
        return source_text
    time.sleep(random.uniform(0, 8))
    for retry_cnt in range(max_retry):
        try:
            result = search_google(query)
            source_text = get_brief_text(result)
            break
        except Exception as exc:
            logger.warning("Search retry %s failed: %s", retry_cnt, exc)
            time.sleep(random.uniform(1, 8))

    if source_text == "":
        logger.warning("Search result for query [%s] is empty", query)
        source_text = "Search result is empty. Please try again."
    return source_text


def get_searches_results(queries: List[str], max_retry: int = 3) -> str:
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return ""
    futures = []
    with ThreadPoolExecutor(max_workers=len(queries)) as executor:
        for i, query in enumerate(queries):
            futures.append(
                executor.submit(
                    lambda j, q: (j, get_search_results(q, max_retry=max_retry)),
                    i,
                    query,
                )
            )
    results = ["" for _ in range(len(queries))]
    for future in as_completed(futures):
        i, output_i = future.result()
        results[i] = output_i
    output = ""
    for i, result in enumerate(results):
        output += (
            f"--- search result for [{queries[i]}] ---\n"
            f"{result}\n"
            f"--- end of search result ---\n\n"
        )
    return output.strip()


def search_web(queries: List[str], max_retry: int = 3) -> Dict[str, Any]:
    if not queries:
        return {"success": False, "error": "No queries provided", "results": ""}
    try:
        result = get_searches_results(queries, max_retry=max_retry)
        return {"success": True, "results": result}
    except Exception as exc:
        return {"success": False, "error": str(exc), "results": ""}


def parse_search_results(search_text: str) -> List[Dict[str, str]]:
    if not search_text:
        return []
    pattern = re.compile(
        r"<title>(.*?)</title>\s*<url>(.*?)</url>\s*<snippet>\s*(.*?)\s*</snippet>",
        re.DOTALL,
    )
    results: List[Dict[str, str]] = []
    for match in pattern.findall(search_text):
        title, url, snippet = match
        entry = {
            "title": (title or "").strip(),
            "url": (url or "").strip(),
            "snippet": (snippet or "").strip(),
        }
        if entry["title"] or entry["url"] or entry["snippet"]:
            results.append(entry)
    return results
