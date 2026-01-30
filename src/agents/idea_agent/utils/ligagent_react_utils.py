from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Set

from src.agents.idea_agent.agent.prompts import (
    BROWSE_PROMPT_TEMPLATE,
    BROWSE_SCHEMA_BASELINE,
    BROWSE_SCHEMA_DATASET,
    REACT_WEBSEARCH_PROMPT,
)
from src.agents.idea_agent.utils.ligagent_utils import parse_json_response
from src.agents.idea_agent.utils.ligagent_suggestion_utils import (
    _compact_search_results,
    _log_llm_output,
    fetch_url_text,
    parse_search_results_limited,
)

_SITE_RESTRICTION_RE = re.compile(r"\bsite:[^\s]+", re.IGNORECASE)


def _strip_site_restrictions(query: str) -> str:
    if not query:
        return query
    cleaned = _SITE_RESTRICTION_RE.sub("", query)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _log_react_event(label: str, message: str, logger, max_chars: int = 1200) -> None:
    text = str(message or "")
    if not text:
        return
    clipped = text if len(text) <= max_chars else text[:max_chars].rstrip() + "...[truncated]"
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        fn="",
        lno=0,
        msg="\U0001F50E ReAct %s: %s",
        args=(label, clipped),
        exc_info=None,
    )
    handled = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.handle(record)
            handled = True
    if not handled:
        print(f"\U0001F50E ReAct {label}: {clipped}")


def _browse_prompt(source_text: str, browse_query: str, kind: str) -> str:
    schema = BROWSE_SCHEMA_DATASET if kind == "dataset" else BROWSE_SCHEMA_BASELINE
    return BROWSE_PROMPT_TEMPLATE.format(
        schema=schema,
        source_text=source_text,
        browse_query=browse_query,
    )


def _normalize_browse_candidates(kind: str, payload: Any) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    items: List[Any] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = [payload]
    else:
        return candidates

    for item in items:
        if not isinstance(item, dict):
            continue
        if kind == "dataset":
            name = (
                item.get("dataset_name")
                or item.get("name")
                or item.get("dataset")
                or item.get("title")
            )
            access = item.get("access_link") or item.get("access") or item.get("url") or item.get("link")
            if not name or not access:
                continue
            is_dataset = item.get("is_dataset")
            if isinstance(is_dataset, str):
                is_dataset = is_dataset.strip().lower()
                if is_dataset in {"true", "yes", "y", "1"}:
                    is_dataset = True
                elif is_dataset in {"false", "no", "n", "0"}:
                    is_dataset = False
            if not isinstance(is_dataset, bool):
                is_dataset = None
            candidates.append(
                {
                    "dataset_name": str(name).strip(),
                    "access_link": str(access).strip(),
                    "is_dataset": is_dataset,
                    "dataset_description": item.get("dataset_description") or item.get("description") or "",
                    "license": item.get("license") or "",
                    "evidence_snippets": item.get("evidence_snippets") or [],
                }
            )
        else:
            paper_title = item.get("paper_title") or item.get("title") or ""
            method = (
                item.get("baseline_method")
                or item.get("method")
                or item.get("baseline")
                or item.get("name")
            )
            arxiv = item.get("arxiv_link") or item.get("arxiv") or item.get("paper_link") or ""
            github = item.get("github_link") or item.get("repo_url") or item.get("github") or ""
            if not paper_title or not arxiv or not github:
                continue
            candidates.append(
                {
                    "paper_title": str(paper_title).strip(),
                    "baseline_method": str(method).strip() if method else "",
                    "arxiv_link": str(arxiv).strip(),
                    "github_link": str(github).strip(),
                    "evidence_snippets": item.get("evidence_snippets") or [],
                }
            )
    return candidates


def _browse_url(
    url: str,
    browse_query: str,
    kind: str,
    chat_fn,
    model: str,
    logger,
    max_chars: int = 18000,
    temperature: float = 0.1,
    max_output_tokens: int = 700,
) -> tuple[str, List[Dict[str, Any]]]:
    if not url:
        return "", []
    source_text = fetch_url_text(url, logger=logger, max_chars=max_chars)
    if not source_text.strip():
        return "", []
    prompt = _browse_prompt(source_text, browse_query, kind)
    try:
        response = chat_fn(prompt, temperature=temperature, max_output_tokens=max_output_tokens, model=model)
        _log_llm_output("browse_answer", response, logger)
        text = (response or "").strip()
        try:
            parsed = parse_json_response(text)
        except Exception:
            parsed = None
        candidates = _normalize_browse_candidates(kind, parsed) if parsed is not None else []
        return text, candidates
    except Exception as exc:  # pragma: no cover - network
        logger.warning("\u26A0\ufe0f Browse failed for %s: %s", url, exc)
        return "", []


def _browse_search_results(
    search_text: str,
    browse_query: str,
    kind: str,
    chat_fn,
    model: str,
    logger,
    max_urls: int = 5,
    browse_max_chars: int = 18000,
    temperature: float = 0.1,
    max_output_tokens: int = 700,
) -> tuple[str, List[Dict[str, Any]], List[str]]:
    results = parse_search_results_limited(search_text, per_query=5)
    urls: List[str] = []
    for item in results:
        url = (item.get("url") or "").strip()
        if not url or url in urls:
            continue
        urls.append(url)
        if len(urls) >= max_urls:
            break
    if not urls:
        return "", [], []
    outputs: List[str] = []
    candidates: List[Dict[str, Any]] = []
    for url in urls:
        answer, extracted = _browse_url(
            url,
            browse_query,
            kind,
            chat_fn=chat_fn,
            model=model,
            logger=logger,
            max_chars=browse_max_chars,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if answer:
            outputs.append(f"--- answer based on [{url}] ---\n{answer}\n--- end of answer ---")
        if extracted:
            candidates.extend(extracted)
    names: List[str] = []
    for item in candidates:
        if kind == "dataset":
            name = item.get("dataset_name") or ""
        else:
            name = item.get("paper_title") or item.get("baseline_method") or ""
        name = str(name).strip()
        if name:
            names.append(name)
    return "\n\n".join(outputs).strip(), candidates, names


def react_websearch(
    kind: str,
    seed_names: List[str],
    topic: str,
    idea_card: Dict[str, Any],
    chat_fn,
    model: str,
    logger,
    search_fn: Callable[[List[str], int], Dict[str, Any]],
    max_steps: int = 5,
    max_urls: int = 2,
    browse_max_chars: int = 18000,
    observation_limit: int = 6,
    search_max_retry: int = 3,
    llm_temperature: float = 0.1,
    llm_step_max_tokens: int = 200,
    llm_browse_max_tokens: int = 700,
    force_seed_queries: bool = False,
) -> Dict[str, Any]:
    if max_steps <= 0:
        return {"search_text": "", "browse_candidates": [], "found_names": []}
    observations: List[str] = []
    executed: List[str] = []
    combined_text = ""
    seed_pool = [name for name in seed_names if name]
    seed_names_all = list(seed_pool)
    browse_candidates: List[Dict[str, Any]] = []
    found_names: Set[str] = set()

    for step in range(1, max_steps + 1):
        obs_text = "\n".join(observations[-2:]) if observations else "(none)"
        browse_query = (
            "Extract dataset name, access link (dataset page or GitHub repo/release), "
            "whether it is truly a dataset, dataset description, license, and evidence snippets."
            if kind == "dataset"
            else "Extract baseline method name, arXiv link, GitHub link, and evidence snippets."
        )
        payload = REACT_WEBSEARCH_PROMPT.format(
            kind=kind,
            seed_names=json.dumps(seed_pool[:8], ensure_ascii=False),
            topic=topic or "",
            task=idea_card.get("task") or "",
            executed=json.dumps(executed, ensure_ascii=False),
            found_names=json.dumps(sorted(found_names), ensure_ascii=False),
            obs=obs_text,
        )
        query = ""
        stop = False
        think = ""
        try:
            response = chat_fn(
                payload,
                temperature=llm_temperature,
                max_output_tokens=llm_step_max_tokens,
                model=model,
            )
            _log_llm_output(f"{kind}_react_step_{step}", response, logger)
            parsed = parse_json_response(response)
            if isinstance(parsed, dict):
                stop = bool(parsed.get("stop"))
                query = str(parsed.get("query") or "").strip()
                think = str(parsed.get("think") or "").strip()
        except Exception as exc:  # pragma: no cover - network
            logger.warning("\u26A0\ufe0f %s ReAct step %s failed: %s", kind.title(), step, exc)

        think = think.strip()
        if think:
            think = " ".join(think.split())
        else:
            think = "choose next query"
        _log_react_event(f"{kind} step {step} THINK", think, logger)

        if stop:
            _log_react_event(f"{kind} step {step} ACT", "stop", logger)
            break

        if query:
            query = _strip_site_restrictions(query)

        if not query:
            if seed_pool:
                name = seed_pool.pop(0)
                if kind == "dataset":
                    query = f"{name} dataset"
                else:
                    query = f"{name} baseline"
            else:
                _log_react_event(f"{kind} step {step} ACT", "no query available", logger)
                break

        key = query.lower().strip()
        if force_seed_queries and seed_names_all:
            matches = [name for name in seed_names_all if name.lower() in key]
            if len(matches) >= 2:
                selected = matches[0]
                query = f"{selected} dataset" if kind == "dataset" else f"{selected} baseline"
                key = query.lower().strip()
                seed_pool = [name for name in seed_pool if name.lower() != selected.lower()]
                _log_react_event(
                    f"{kind} step {step} ACT",
                    f"rewrite query to single seed name: {selected}",
                    logger,
                )
            elif not matches:
                candidate_pool = seed_pool or seed_names_all
                if candidate_pool:
                    selected = candidate_pool[0]
                    if seed_pool:
                        seed_pool.pop(0)
                    query = f"{selected} dataset" if kind == "dataset" else f"{selected} baseline"
                    key = query.lower().strip()
                    _log_react_event(
                        f"{kind} step {step} ACT",
                        f"override query with seed name: {selected}",
                        logger,
                    )
            else:
                selected = matches[0]
                seed_pool = [name for name in seed_pool if name.lower() != selected.lower()]
        if not key or key in executed:
            _log_react_event(f"{kind} step {step} ACT", f"skip duplicate query: {query}", logger)
            continue
        if any(name in key for name in found_names):
            _log_react_event(f"{kind} step {step} ACT", f"skip found-name query: {query}", logger)
            continue
        executed.append(key)
        _log_react_event(f"{kind} step {step} ACT", f"search: {query}", logger)

        try:
            payload = search_fn([query], max_retry=search_max_retry)
            results_text = payload.get("results", "")
        except Exception as exc:  # pragma: no cover - network
            logger.warning("\u26A0\ufe0f %s ReAct search failed: %s", kind.title(), exc)
            results_text = ""
        if results_text:
            combined_text = f"{combined_text}\n{results_text}" if combined_text else results_text
            observation = _compact_search_results(results_text, limit=observation_limit)
            browse_text, extracted_candidates, extracted_names = _browse_search_results(
                results_text,
                browse_query,
                kind,
                chat_fn=chat_fn,
                model=model,
                logger=logger,
                max_urls=max_urls,
                browse_max_chars=browse_max_chars,
                temperature=llm_temperature,
                max_output_tokens=llm_browse_max_tokens,
            )
            if browse_text:
                combined_text = f"{combined_text}\n{browse_text}"
            browse_obs = browse_text.replace("\n", " ").strip()
            if browse_obs:
                browse_obs = browse_obs[:600]
                observations.append(f"{observation}\nBrowse: {browse_obs}")
            else:
                observations.append(observation)
            if extracted_candidates:
                browse_candidates.extend(extracted_candidates)
            for name in extracted_names:
                found_names.add(name.lower())
            _log_react_event(
                f"{kind} step {step} OBSERVE",
                (f"{observation}\nBrowse: {browse_obs}" if browse_obs else (observation or "(empty)")),
                logger,
            )
        else:
            _log_react_event(f"{kind} step {step} OBSERVE", "(empty)", logger)

    return {
        "search_text": combined_text,
        "browse_candidates": browse_candidates,
        "found_names": sorted(found_names),
    }
