from __future__ import annotations

import html
import json
import logging
import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional, Set

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.agents.idea_agent.utils.idea_helpers import parse_search_results
from src.agents.idea_agent.utils.ligagent_utils import (
    extract_baseline_names,
    extract_dataset_names,
    parse_json_response,
)


def dedupe_named(entries: List[Dict[str, Any]], name_key: str, limit: int = 5) -> List[Dict[str, Any]]:
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
        if len(output) >= limit:
            break
    return output


def derive_label(title: str) -> str:
    cleaned = (title or "").strip()
    for sep in (" - ", " | ", " — ", " – ", ":"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip()
    return cleaned


def trim_query_context(text: str, max_words: int = 10) -> str:
    if not text:
        return ""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\\-]+", text)
    if not words:
        return ""
    return " ".join(words[:max_words])


def dataset_fallback_from_search(search_text: str, max_items: int) -> List[Dict[str, Any]]:
    results = parse_search_results(search_text)
    fallback: List[Dict[str, Any]] = []
    for item in results:
        name = derive_label(item.get("title", ""))
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


def normalize_candidate_key(title: str, url: str) -> str:
    if url:
        cleaned = url.split("#", 1)[0].split("?", 1)[0].lower().strip()
        return cleaned
    return re.sub(r"\\W+", " ", (title or "").lower()).strip()

_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_HTTP_SESSION: Optional[requests.Session] = None
_JINA_SESSION: Optional[requests.Session] = None


def _log_llm_output(label: str, response: str, logger, max_chars: int = 2000) -> None:
    if response is None:
        return
    text = str(response)
    if not text:
        return
    clipped = text if len(text) <= max_chars else text[:max_chars].rstrip() + "...[truncated]"
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        fn="",
        lno=0,
        msg="🧠 LLM %s: %s",
        args=(label, clipped),
        exc_info=None,
    )
    handled = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.handle(record)
            handled = True
    if not handled:
        print(f"🧠 LLM {label}: {clipped}")


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
        msg="🔎 ReAct %s: %s",
        args=(label, clipped),
        exc_info=None,
    )
    handled = False
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.handle(record)
            handled = True
    if not handled:
        print(f"🔎 ReAct {label}: {clipped}")


def compact_search_results(search_text: str, limit: int = 10) -> str:
    return _compact_search_results(search_text, limit=limit)


def preprocess_candidate_names(
    kind: str,
    names: List[str],
    topic: str,
    idea_card: Dict[str, Any],
    chat_fn,
    model: str,
    logger,
    max_items: int = 8,
) -> List[str]:
    if not names:
        return []
    seed = [_normalize_candidate_name(name) for name in names if name]
    seed = [name for name in seed if name and not _is_fragmented_name(name)]
    if not seed:
        return []
    prompt = (
        "You are filtering a list of candidate {kind} names. "
        "Remove irrelevant items, generic terms, or model-only names. "
        "Keep canonical {kind} names. Deduplicate. Return JSON array of names only."
        "\nTopic: {topic}\nTask: {task}\nCandidates: {candidates}\n"
    )
    payload = prompt.format(
        kind=kind,
        topic=topic or "",
        task=idea_card.get("task") or "",
        candidates=json.dumps(seed[:max_items * 2], ensure_ascii=False),
    )
    try:
        response = chat_fn(payload, temperature=0.1, max_tokens=400, model=model)
        _log_llm_output(f"{kind}_preprocess", response, logger)
        parsed = parse_json_response(response)
        if isinstance(parsed, list):
            cleaned = [_normalize_candidate_name(str(item)) for item in parsed if str(item).strip()]
        else:
            cleaned = []
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ %s preprocessing failed: %s", kind.title(), exc)
        cleaned = []

    if not cleaned:
        cleaned = seed

    seen: Set[str] = set()
    output: List[str] = []
    for name in cleaned:
        name = _normalize_candidate_name(name)
        if not name or _is_fragmented_name(name):
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(name)
        if len(output) >= max_items:
            break
    return output


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
) -> str:
    if max_steps <= 0:
        return ""
    observations: List[str] = []
    executed: List[str] = []
    combined_text = ""
    seed_pool = [name for name in seed_names if name]

    for step in range(1, max_steps + 1):
        obs_text = "\n".join(observations[-2:]) if observations else "(none)"
        prompt = (
            "You are running a multi-step web search for {kind} names. "
            "At each step, propose one search query or stop.\n"
            "Rules: Output JSON {{\"think\": \"...\", \"query\": \"...\", \"stop\": false}}. "
            "The think field should be a chain-of-thought. "
            "Prefer authoritative sources. Avoid repeating queries.\n"
            "For datasets, prefer HuggingFace/Kaggle/PapersWithCode. "
            "For baselines, prefer arXiv/GitHub.\n"
            "SeedNames: {seed_names}\nTopic: {topic}\nTask: {task}\n"
            "ExecutedQueries: {executed}\nObservations:\n{obs}\n"
        )
        payload = prompt.format(
            kind=kind,
            seed_names=json.dumps(seed_pool[:8], ensure_ascii=False),
            topic=topic or "",
            task=idea_card.get("task") or "",
            executed=json.dumps(executed, ensure_ascii=False),
            obs=obs_text,
        )
        query = ""
        stop = False
        think = ""
        try:
            response = chat_fn(payload, temperature=0.1, max_tokens=200, model=model)
            _log_llm_output(f"{kind}_react_step_{step}", response, logger)
            parsed = parse_json_response(response)
            if isinstance(parsed, dict):
                stop = bool(parsed.get("stop"))
                query = str(parsed.get("query") or "").strip()
                think = str(parsed.get("think") or "").strip()
        except Exception as exc:  # pragma: no cover - network
            logger.warning("⚠️ %s ReAct step %s failed: %s", kind.title(), step, exc)

        think = think.strip()
        if think:
            think = " ".join(think.split())
        else:
            think = "choose next query"
        _log_react_event(f"{kind} step {step} THINK", think, logger)

        if stop:
            _log_react_event(f"{kind} step {step} ACT", "stop", logger)
            break

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
        if not key or key in executed:
            _log_react_event(f"{kind} step {step} ACT", f"skip duplicate query: {query}", logger)
            continue
        executed.append(key)
        _log_react_event(f"{kind} step {step} ACT", f"search: {query}", logger)

        try:
            payload = search_fn([query], max_retry=3)
            results_text = payload.get("results", "")
        except Exception as exc:  # pragma: no cover - network
            logger.warning("⚠️ %s ReAct search failed: %s", kind.title(), exc)
            results_text = ""
        if results_text:
            combined_text = f"{combined_text}\n{results_text}" if combined_text else results_text
            observation = _compact_search_results(results_text, limit=6)
            observations.append(observation)
            _log_react_event(f"{kind} step {step} OBSERVE", observation or "(empty)", logger)
        else:
            _log_react_event(f"{kind} step {step} OBSERVE", "(empty)", logger)

    return combined_text


def postprocess_suggestions(
    kind: str,
    items: List[Dict[str, Any]],
    chat_fn,
    model: str,
    logger,
    max_keep: int = 5,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    preview: List[str] = []
    for idx, item in enumerate(items[: max_keep * 2]):
        name = item.get("name") or item.get("title") or ""
        link = item.get("link") or item.get("url") or ""
        usage = (item.get("usage") or item.get("snippet") or "")[:160]
        preview.append(f"{idx}. name={name} link={link} usage={usage}")
    prompt = (
        "You are cleaning a list of suggested {kind}. "
        "Remove items that are clearly irrelevant, malformed, or not real {kind}s. "
        "Return JSON array of indices to keep."
        "\nItems:\n{items}\n"
    )
    payload = prompt.format(kind=kind, items="\n".join(preview))
    try:
        response = chat_fn(payload, temperature=0.1, max_tokens=200, model=model)
        _log_llm_output(f"{kind}_postprocess", response, logger)
        parsed = parse_json_response(response)
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ %s postprocess failed: %s", kind.title(), exc)
        return items[:max_keep]

    keep_indices: Set[int] = set()
    if isinstance(parsed, list):
        for entry in parsed:
            if isinstance(entry, int):
                keep_indices.add(entry)
            else:
                name = str(entry).strip().lower()
                if not name:
                    continue
                for idx, item in enumerate(items):
                    item_name = (item.get("name") or item.get("title") or "").strip().lower()
                    if item_name == name:
                        keep_indices.add(idx)
    if not keep_indices:
        return items[:max_keep]
    filtered = [item for idx, item in enumerate(items) if idx in keep_indices]
    return filtered[:max_keep]


def collect_top_dataset_names_from_memory(memory: Optional[Dict[str, Any]], top_k: int = 5) -> List[str]:
    if not memory:
        return []
    counts = memory.get("dataset_mentions") or {}
    display = memory.get("dataset_mentions_display") or {}
    if not counts:
        storage = memory.get("paper_contents") or {}
        for entry in storage.values():
            keynote = entry.get("keynote") if isinstance(entry, dict) else None
            for dataset_name in extract_dataset_names(keynote):
                key = dataset_name.lower()
                counts[key] = counts.get(key, 0) + 1
                display.setdefault(key, dataset_name)

    if not counts:
        return []
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [display.get(name, name) for name, _ in ordered[:top_k]]


def collect_top_baseline_names_from_memory(memory: Optional[Dict[str, Any]], top_k: int = 5) -> List[str]:
    if not memory:
        return []
    counts = memory.get("baseline_mentions") or {}
    display = memory.get("baseline_mentions_display") or {}
    if not counts:
        storage = memory.get("paper_contents") or {}
        for entry in storage.values():
            keynote = entry.get("keynote") if isinstance(entry, dict) else None
            for baseline_name in extract_baseline_names(keynote):
                key = baseline_name.lower()
                counts[key] = counts.get(key, 0) + 1
                display.setdefault(key, baseline_name)
            if isinstance(keynote, dict) and ("baseline" in keynote or "baselines" in keynote):
                paper_title = (entry.get("title") or "").strip()
                if paper_title:
                    key = paper_title.lower()
                    counts[key] = counts.get(key, 0) + 1
                    display.setdefault(key, paper_title)

    if not counts:
        return []
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [display.get(name, name) for name, _ in ordered[:top_k]]


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "POST"),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _get_http_session() -> requests.Session:
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        _HTTP_SESSION = _build_session()
    return _HTTP_SESSION


def _get_jina_session() -> requests.Session:
    global _JINA_SESSION
    if _JINA_SESSION is None:
        _JINA_SESSION = _build_session()
    return _JINA_SESSION


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _strip_html(html_text: str) -> str:
    if not html_text:
        return ""
    cleaned = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", html_text)
    cleaned = re.sub(r"(?is)<!--.*?-->", " ", cleaned)
    cleaned = re.sub(r"(?is)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return _normalize_whitespace(cleaned)


def _request_text(
    url: str,
    session: requests.Session,
    logger,
    timeout: tuple = (5, 20),
    headers: Optional[Dict[str, str]] = None,
    max_chars: int = 20000,
) -> str:
    try:
        resp = session.get(url, headers=headers or _DEFAULT_HEADERS, timeout=timeout)
        if not resp.ok:
            return ""
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "application/pdf" in content_type:
            return ""
        text = resp.text or ""
        return text[:max_chars]
    except requests.exceptions.SSLError as exc:  # pragma: no cover - network
        logger.warning("⚠️ SSL error for %s: %s", url, exc)
        return ""
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Request failed for %s: %s", url, exc)
        return ""


def _extract_arxiv_id(url: str) -> str:
    if not url:
        return ""
    patterns = [
        r"arxiv\.org/(?:abs|pdf|html)/([^?#]+)",
        r"ar5iv\.labs\.arxiv\.org/html/([^?#]+)",
        r"export\.arxiv\.org/(?:abs|pdf)/([^?#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            arxiv_id = match.group(1)
            if arxiv_id.endswith(".pdf"):
                arxiv_id = arxiv_id[: -len(".pdf")]
            return arxiv_id.strip()
    match = re.search(r"\\b(\\d{4}\\.\\d{4,5})(v\\d+)?\\b", url)
    if match:
        return (match.group(1) + (match.group(2) or "")).strip()
    match = re.search(r"\\b([a-zA-Z\\-]+/\\d{7})(v\\d+)?\\b", url)
    if match:
        return (match.group(1) + (match.group(2) or "")).strip()
    return ""


def _extract_arxiv_metadata(html_text: str) -> str:
    if not html_text:
        return ""
    title = ""
    abstract = ""
    authors: List[str] = []

    meta_title = re.search(r'name="citation_title"\\s+content="([^"]+)"', html_text, re.I)
    if meta_title:
        title = html.unescape(meta_title.group(1)).strip()
    meta_abstract = re.search(r'name="citation_abstract"\\s+content="([^"]+)"', html_text, re.I)
    if meta_abstract:
        abstract = html.unescape(meta_abstract.group(1)).strip()

    for match in re.findall(r'name="citation_author"\\s+content="([^"]+)"', html_text, re.I):
        cleaned = html.unescape(match).strip()
        if cleaned:
            authors.append(cleaned)

    if not abstract:
        block = re.search(r'<blockquote[^>]*class="abstract[^"]*"[^>]*>(.*?)</blockquote>', html_text, re.I | re.S)
        if block:
            abstract = _strip_html(block.group(1))
            abstract = re.sub(r"^Abstract:?\\s*", "", abstract).strip()

    parts: List[str] = []
    if title:
        parts.append(f"Title: {title}")
    if authors:
        parts.append(f"Authors: {', '.join(authors)}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    return "\n".join(parts).strip()


def _fetch_arxiv_abs(arxiv_id: str, logger) -> str:
    session = _get_http_session()
    for base in ("https://arxiv.org/abs/", "https://export.arxiv.org/abs/"):
        url = f"{base}{arxiv_id}"
        html_text = _request_text(url, session=session, logger=logger)
        if html_text:
            meta = _extract_arxiv_metadata(html_text)
            if meta:
                return meta
    return ""


def _fetch_ar5iv_text(arxiv_id: str, logger, max_chars: int = 20000) -> str:
    session = _get_http_session()
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    html_text = _request_text(url, session=session, logger=logger, max_chars=max_chars)
    if not html_text:
        return ""
    text = _strip_html(html_text)
    return text


def _download_arxiv_pdf(arxiv_id: str, logger, max_bytes: int = 25 * 1024 * 1024) -> str:
    session = _get_http_session()
    for base in ("https://arxiv.org/pdf/", "https://export.arxiv.org/pdf/"):
        url = f"{base}{arxiv_id}.pdf"
        try:
            resp = session.get(url, headers=_DEFAULT_HEADERS, timeout=(5, 30), stream=True)
            if not resp.ok:
                continue
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "application/pdf" not in content_type:
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                total = 0
                for chunk in resp.iter_content(chunk_size=1024 * 512):
                    if not chunk:
                        continue
                    tmp.write(chunk)
                    total += len(chunk)
                    if total >= max_bytes:
                        break
                pdf_path = tmp.name
            return pdf_path
        except requests.exceptions.SSLError as exc:  # pragma: no cover - network
            logger.warning("⚠️ SSL error fetching PDF %s: %s", url, exc)
        except Exception as exc:  # pragma: no cover - network
            logger.warning("⚠️ PDF download failed for %s: %s", url, exc)
    return ""


def _extract_pdf_text(pdf_path: str, logger, max_pages: int = 6, max_chars: int = 12000) -> str:
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    try:
        import pypdfium2 as pdfium
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("⚠️ PDF text extraction unavailable: %s", exc)
        return ""
    text_chunks: List[str] = []
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        page_count = len(pdf)
        for page_index in range(min(page_count, max_pages)):
            page = pdf.get_page(page_index)
            textpage = page.get_textpage()
            try:
                page_text = textpage.get_text_range()
            except Exception:
                page_text = ""
            try:
                textpage.close()
            except Exception:
                pass
            try:
                page.close()
            except Exception:
                pass
            if page_text:
                text_chunks.append(page_text)
            if sum(len(chunk) for chunk in text_chunks) >= max_chars:
                break
        try:
            pdf.close()
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover - pdf parsing
        logger.warning("⚠️ PDF text extraction failed: %s", exc)
        return ""
    return _normalize_whitespace(" ".join(text_chunks))[:max_chars]


def _fetch_jina_text(url: str, logger, max_chars: int = 20000) -> str:
    if not url:
        return ""
    try:
        if url.startswith("https://"):
            jina_url = "https://r.jina.ai/https://" + url[len("https://") :]
        elif url.startswith("http://"):
            jina_url = "https://r.jina.ai/http://" + url[len("http://") :]
        else:
            jina_url = "https://r.jina.ai/http://" + url
        resp = _get_jina_session().get(
            jina_url,
            headers=_DEFAULT_HEADERS,
            timeout=(5, 20),
        )
        if not resp.ok:
            return ""
        text = resp.text or ""
        text = text.replace("\r", " ").strip()
        return text[:max_chars]
    except requests.exceptions.SSLError as exc:  # pragma: no cover - network
        logger.warning("⚠️ Jina SSL error for %s: %s", url, exc)
        return ""
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Jina fetch failed for %s: %s", url, exc)
        return ""


def _fetch_arxiv_text(url: str, logger, max_chars: int = 3500) -> str:
    arxiv_id = _extract_arxiv_id(url)
    if not arxiv_id:
        return ""
    base_id = re.sub(r"v\\d+$", "", arxiv_id)
    id_candidates = [arxiv_id]
    if base_id and base_id != arxiv_id:
        id_candidates.append(base_id)

    best_text = ""
    for candidate_id in id_candidates:
        meta_text = _fetch_arxiv_abs(candidate_id, logger=logger)
        html_text = _fetch_ar5iv_text(candidate_id, logger=logger)
        combined = "\n".join([part for part in (meta_text, html_text) if part]).strip()
        if combined and len(combined) > len(best_text):
            best_text = combined
        if html_text and len(html_text) >= 800:
            return combined[:max_chars]

        pdf_path = _download_arxiv_pdf(candidate_id, logger=logger)
        if pdf_path:
            try:
                pdf_text = _extract_pdf_text(pdf_path, logger=logger)
            finally:
                try:
                    os.remove(pdf_path)
                except OSError:
                    pass
            if pdf_text:
                combined = "\n".join([part for part in (meta_text, pdf_text) if part]).strip()
                if combined and len(combined) > len(best_text):
                    best_text = combined
                return combined[:max_chars]
    return best_text[:max_chars] if best_text else ""


def fetch_url_text(url: str, logger, max_chars: int = 3500) -> str:
    if not url:
        return ""
    if "arxiv.org" in url or "ar5iv.labs.arxiv.org" in url or "export.arxiv.org" in url:
        arxiv_text = _fetch_arxiv_text(url, logger=logger, max_chars=max_chars)
        if arxiv_text:
            return arxiv_text

    direct_text = _request_text(url, session=_get_http_session(), logger=logger, max_chars=max_chars * 3)
    if direct_text:
        if "<html" in direct_text.lower() or "<body" in direct_text.lower():
            direct_text = _strip_html(direct_text)
        if direct_text:
            return direct_text[:max_chars]

    jina_text = _fetch_jina_text(url, logger=logger, max_chars=max_chars * 3)
    if jina_text:
        return jina_text[:max_chars]
    return ""


def build_baseline_idea_card(
    topic: str,
    best_entry: Dict[str, Any],
    algorithm: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> Dict[str, Any]:
    prompt = prompts.get("baseline_idea_card")
    if not prompt:
        prompt = (
            "You are structuring an idea for baseline retrieval. "
            "Return JSON with keys: task, benchmarks, method_family, key_components, evaluation_axes."
            "\nTopic: {topic}\nIdea: {idea}\nAlgorithm: {algorithm}\nReferences: {references}\n"
        )
    payload = prompt.format(
        topic=topic,
        idea=json.dumps(best_entry, ensure_ascii=False, indent=2),
        algorithm=json.dumps(algorithm, ensure_ascii=False, indent=2),
        references=json.dumps(references[:6], ensure_ascii=False, indent=2),
    )
    try:
        response = chat_fn(payload, temperature=0.1, max_tokens=1200, model=model)
        _log_llm_output("baseline_idea_card", response, logger)
        parsed = parse_json_response(response)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Baseline idea card failed: %s", exc)
    return {
        "task": topic or best_entry.get("title") or "unspecified task",
        "benchmarks": [],
        "method_family": [],
        "key_components": [],
        "evaluation_axes": [],
    }


def build_baseline_queries(
    topic: str,
    best_entry: Dict[str, Any],
    references: List[Dict[str, Any]],
    idea_card: Dict[str, Any],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> List[str]:
    idea_title = best_entry.get("title", "")
    idea_context = trim_query_context(best_entry.get("abstract") or best_entry.get("core_contribute") or "")
    task = (idea_card.get("task") or topic or idea_title).strip()
    benchmarks = idea_card.get("benchmarks") or []
    method_family = idea_card.get("method_family") or []
    key_components = idea_card.get("key_components") or []
    evaluation_axes = idea_card.get("evaluation_axes") or []

    def _sanitize_queries(values: Any, limit: int = 10) -> List[str]:
        raw_queries: List[str] = []
        if isinstance(values, list):
            raw_queries = [str(v) for v in values if v]
        elif isinstance(values, str):
            raw_queries = [line.strip() for line in re.split(r"[\n;]+", values) if line.strip()]
        elif isinstance(values, dict) and isinstance(values.get("queries"), list):
            raw_queries = [str(v) for v in values.get("queries") if v]

        cleaned: List[str] = []
        seen_local: Set[str] = set()
        for q in raw_queries:
            q = (q or "").replace("\r", " ").replace("\n", " ").strip()
            if not q:
                continue
            if len(q) > 140:
                q = q[:140].rstrip()
            key = q.lower()
            if key in seen_local:
                continue
            seen_local.add(key)
            cleaned.append(q)
            if len(cleaned) >= limit:
                break
        return cleaned

    def _heuristic_queries() -> List[str]:
        queries: List[str] = []
        if idea_title:
            queries.append(f"\"{idea_title}\" baseline")
            queries.append(f"\"{idea_title}\" implementation")
        if task:
            queries.append(f"{task} arxiv")
        for component in key_components[:2]:
            queries.append(f"{task} {component}")
        for axis in evaluation_axes[:2]:
            queries.append(f"{task} {axis}")
        for ref in references[:3]:
            ref_title = (ref.get("title") or "").strip()
            if ref_title:
                queries.append(f"\"{ref_title}\"")
        return _sanitize_queries(queries, limit=10)

    llm_prompt = prompts.get("baseline_query_generation")
    if not llm_prompt:
        llm_prompt = (
            "You are generating web search queries to find strong baselines for a research idea. "
            "Given the inputs, output JSON: {{\"queries\": [..]}}. "
            "Rules: 4-6 queries, each <= 5 words; mix broad and specific; include at most 2 queries with site: and at most 2 with github; "
            "avoid overly long keyword chains; do not include explanations.\n"
            "Topic: {topic}\nTask: {task}\nIdeaTitle: {idea_title}\nIdeaContext: {idea_context}\n"
            "Benchmarks: {benchmarks}\nMethodFamily: {method_family}\nKeyComponents: {key_components}\nEvaluationAxes: {evaluation_axes}\n"
            "ReferenceTitles: {reference_titles}\n"
        )

    reference_titles: List[str] = []
    for ref in references[:3]:
        ref_title = (ref.get("title") or "").strip()
        if ref_title:
            reference_titles.append(ref_title)

    llm_payload = llm_prompt.format(
        topic=topic or "",
        task=task or "",
        idea_title=idea_title or "",
        idea_context=idea_context or "",
        benchmarks=json.dumps(benchmarks[:3], ensure_ascii=False),
        method_family=json.dumps(method_family[:3], ensure_ascii=False),
        key_components=json.dumps(key_components[:2], ensure_ascii=False),
        evaluation_axes=json.dumps(evaluation_axes[:2], ensure_ascii=False),
        reference_titles=json.dumps(reference_titles, ensure_ascii=False),
    )
    try:
        response = chat_fn(llm_payload, temperature=0.1, max_tokens=512, model=model)
        _log_llm_output("baseline_query_generation", response, logger)
        parsed = parse_json_response(response)
        llm_queries = _sanitize_queries(parsed, limit=10)
        print(f"[Debug] {llm_queries}")
        if len(llm_queries) >= 5:
            return llm_queries
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Baseline query generation failed; falling back to templates: %s", exc)

    return _heuristic_queries()


def build_dataset_seed_queries(
    topic: str,
    best_entry: Dict[str, Any],
    idea_card: Dict[str, Any],
    references: List[Dict[str, Any]],
) -> List[str]:
    idea_title = best_entry.get("title", "")
    task = (idea_card.get("task") or topic or idea_title).strip()
    domain = str(idea_card.get("domain") or "").strip()
    queries: List[str] = []
    if task:
        queries.append(f"classic datasets for {task}")
        queries.append(f"popular datasets for {task}")
        queries.append(f"benchmark datasets {task}")
    if domain:
        queries.append(f"{domain} datasets")
    if idea_title:
        queries.append(f"{idea_title} dataset")
    for ref in references[:2]:
        ref_title = (ref.get("title") or "").strip()
        if ref_title:
            queries.append(f"{ref_title} dataset")

    seen: Set[str] = set()
    output: List[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(q)
        if len(output) >= 8:
            break
    return output


def build_baseline_seed_queries(
    topic: str,
    best_entry: Dict[str, Any],
    idea_card: Dict[str, Any],
    references: List[Dict[str, Any]],
) -> List[str]:
    idea_title = best_entry.get("title", "")
    task = (idea_card.get("task") or topic or idea_title).strip()
    queries: List[str] = []
    if task:
        queries.append(f"classic baselines for {task}")
        queries.append(f"standard baselines {task}")
        queries.append(f"benchmark methods {task}")
        queries.append(f"state of the art {task} baselines")
    if idea_title:
        queries.append(f"{idea_title} baseline")
    for ref in references[:2]:
        ref_title = (ref.get("title") or "").strip()
        if ref_title:
            queries.append(f"{ref_title} baseline")

    seen: Set[str] = set()
    output: List[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(q)
        if len(output) >= 8:
            break
    return output


def _compact_search_results(search_text: str, limit: int = 10) -> str:
    results = parse_search_results_limited(search_text, per_query=3)
    lines: List[str] = []
    for idx, item in enumerate(results[:limit], 1):
        title = (item.get("title") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if title or snippet:
            lines.append(f"{idx}. {title} — {snippet}")
    return "\n".join(lines).strip()


def parse_search_results_limited(search_text: str, per_query: int = 3) -> List[Dict[str, str]]:
    if not search_text:
        return []
    pattern = re.compile(
        r"--- search result for \\[(.*?)\\] ---\\n(.*?)\\n--- end of search result ---",
        re.DOTALL,
    )
    matches = pattern.findall(search_text)
    if not matches:
        return parse_search_results(search_text)[:per_query]

    limited: List[Dict[str, str]] = []
    for _, block in matches:
        block_results = parse_search_results(block)
        limited.extend(block_results[:per_query])
    return limited


def _normalize_candidate_name(name: str) -> str:
    cleaned = re.sub(r"[\\r\\n]+", " ", name or "").strip()
    cleaned = re.sub(r"\\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\\b(dataset|datasets|baseline|baselines|benchmark|benchmarks)\\b", "", cleaned, flags=re.I)
    cleaned = cleaned.strip(" -:;,.")
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rstrip()
    return cleaned



def _is_fragmented_name(name: str) -> bool:
    tokens = [t for t in (name or "").split() if t]
    if len(tokens) >= 3:
        short = sum(1 for t in tokens if len(t) <= 2)
        if short / len(tokens) >= 0.5:
            return True
    if re.search(r"\\b\\w\\b(?:\\s+\\w\\b){2,}", name or ""):
        return True
    return False


def extract_candidate_names(
    kind: str,
    idea_card: Dict[str, Any],
    search_text: str,
    chat_fn,
    model: str,
    logger,
    max_names: int = 8,
) -> List[str]:
    compact = _compact_search_results(search_text, limit=10)
    if not compact:
        return []
    prompt = (
        "You are extracting names from web search snippets. "
        "Return JSON array of up to {max_names} {kind} names only. "
        "Prefer canonical names, no commentary."
        "\nTask: {task}\nSearchResults:\n{results}\n"
    )
    payload = prompt.format(
        max_names=max_names,
        kind=kind,
        task=idea_card.get("task") or "",
        results=compact,
    )
    try:
        response = chat_fn(payload, temperature=0.1, max_tokens=300, model=model)
        _log_llm_output(f"{kind}_name_extraction", response, logger)
        parsed = parse_json_response(response)
        if isinstance(parsed, list):
            names = [_normalize_candidate_name(str(item)) for item in parsed if str(item).strip()]
        else:
            names = []
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ %s name extraction failed: %s", kind.title(), exc)
        names = []

    if not names:
        fallback: List[str] = []
        for item in parse_search_results_limited(search_text, per_query=3)[:12]:
            title = derive_label(item.get("title", ""))
            title = _normalize_candidate_name(title)
            if title and not _is_fragmented_name(title):
                fallback.append(title)
            if len(fallback) >= max_names:
                break
        names = fallback

    seen: Set[str] = set()
    output: List[str] = []
    for name in names:
        cleaned = _normalize_candidate_name(name)
        if not cleaned:
            continue
        if _is_fragmented_name(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= max_names:
            break
    return output


def build_dataset_followup_queries(names: List[str], topic: str, idea_card: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    for name in names:
        if name:
            queries.append(f"site:huggingface.co/datasets \"{name}\"")
            queries.append(f"site:kaggle.com/datasets \"{name}\"")
            queries.append(f"site:paperswithcode.com/dataset \"{name}\"")
            queries.append(f"\"{name}\" dataset")
    if not queries:
        task = (idea_card.get("task") or topic or "").strip()
        if task:
            queries.append(f"{task} dataset")
            queries.append(f"{task} benchmark dataset")
            queries.append(f"site:huggingface.co/datasets {task}")
            queries.append(f"site:kaggle.com/datasets {task}")
            queries.append(f"site:paperswithcode.com/dataset {task}")

    seen: Set[str] = set()
    output: List[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(q)
        if len(output) >= 10:
            break
    return output


def build_dataset_fallback_queries(names: List[str], topic: str, idea_card: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    for name in names:
        if name:
            queries.append(f"\"{name}\" dataset")
            queries.append(f"\"{name}\" benchmark dataset")
            queries.append(f"{name} data")
    if not queries:
        task = (idea_card.get("task") or topic or "").strip()
        if task:
            queries.append(f"{task} dataset")
            queries.append(f"{task} benchmark dataset")
    seen: Set[str] = set()
    output: List[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(q)
        if len(output) >= 10:
            break
    return output


def build_datasetsearch_direct_candidates(
    names: List[str],
    topic: str,
    idea_card: Dict[str, Any],
    note: str = "Dataset Search direct entry; requires verification.",
) -> List[Dict[str, Any]]:
    queries: List[str] = []
    for name in names:
        if name:
            queries.append(name)
    if not queries:
        task = (idea_card.get("task") or topic or "").strip()
        if task:
            queries.append(task)
    seen: Set[str] = set()
    output: List[Dict[str, Any]] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        url = f"https://datasetsearch.research.google.com/search?query={q.replace(' ', '%20')}"
        output.append({"title": q, "url": url, "snippet": note})
    return output


def build_baseline_followup_queries(names: List[str], topic: str, idea_card: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    for name in names:
        if name:
            queries.append(f"site:arxiv.org \"{name}\"")
            queries.append(f"\"{name}\" arxiv")
            queries.append(f"site:github.com \"{name}\"")
            queries.append(f"\"{name}\" github")
    if not queries:
        task = (idea_card.get("task") or topic or "").strip()
        if task:
            queries.append(f"site:arxiv.org {task} baseline")
            queries.append(f"{task} baseline github")

    seen: Set[str] = set()
    output: List[str] = []
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(q)
        if len(output) >= 10:
            break
    return output


def build_dataset_idea_card(
    topic: str,
    best_entry: Dict[str, Any],
    references: List[Dict[str, Any]],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> Dict[str, Any]:
    prompt = prompts.get("dataset_idea_card")
    if not prompt:
        prompt = (
            "You are structuring a dataset search intent. "
            "Return JSON with keys: task, domain, data_type, modalities, evaluation_axes, constraints."
            "\nTopic: {topic}\nIdea: {idea}\nReferences: {references}\n"
        )
    payload = prompt.format(
        topic=topic,
        idea=json.dumps(best_entry, ensure_ascii=False, indent=2),
        references=json.dumps(references[:6], ensure_ascii=False, indent=2),
    )
    try:
        response = chat_fn(payload, temperature=0.1, max_tokens=800, model=model)
        _log_llm_output("dataset_idea_card", response, logger)
        parsed = parse_json_response(response)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Dataset idea card failed: %s", exc)
    return {
        "task": topic or best_entry.get("title") or "unspecified task",
        "domain": "",
        "data_type": "",
        "modalities": [],
        "evaluation_axes": [],
        "constraints": [],
    }


def build_dataset_queries(
    topic: str,
    best_entry: Dict[str, Any],
    references: List[Dict[str, Any]],
    idea_card: Dict[str, Any],
    prompts: Dict[str, str],
    chat_fn,
    model: str,
    logger,
) -> List[str]:
    idea_title = best_entry.get("title", "")
    idea_context = trim_query_context(best_entry.get("abstract") or best_entry.get("core_contribute") or "")
    task = (idea_card.get("task") or topic or idea_title).strip()
    domain = str(idea_card.get("domain") or "").strip()
    data_type = str(idea_card.get("data_type") or "").strip()
    modalities = idea_card.get("modalities") or []
    evaluation_axes = idea_card.get("evaluation_axes") or []

    def _sanitize_queries(values: Any, limit: int = 10) -> List[str]:
        raw_queries: List[str] = []
        if isinstance(values, list):
            raw_queries = [str(v) for v in values if v]
        elif isinstance(values, str):
            raw_queries = [line.strip() for line in re.split(r"[\n;]+", values) if line.strip()]
        elif isinstance(values, dict) and isinstance(values.get("queries"), list):
            raw_queries = [str(v) for v in values.get("queries") if v]

        cleaned: List[str] = []
        seen_local: Set[str] = set()
        for q in raw_queries:
            q = (q or "").replace("\r", " ").replace("\n", " ").strip()
            if not q:
                continue
            if len(q) > 140:
                q = q[:140].rstrip()
            key = q.lower()
            if key in seen_local:
                continue
            seen_local.add(key)
            cleaned.append(q)
            if len(cleaned) >= limit:
                break
        return cleaned

    def _heuristic_queries() -> List[str]:
        queries: List[str] = []
        if task:
            queries.append(f"Dataset {task}")
            queries.append(f"Benchmark {task}")
        if data_type:
            queries.append(f"{task} {data_type} dataset")
        if evaluation_axes:
            for axis in evaluation_axes[:2]:
                queries.append(f"{task} {axis} dataset benchmark")

        return _sanitize_queries(queries, limit=10)

    llm_prompt = prompts.get("dataset_query_generation")
    if not llm_prompt:
        llm_prompt = (
            "You are generating web search queries to find datasets. "
            "Output JSON: {{\"queries\": [..]}}. "
            "Rules: 4-6 queries, each <= 6 words; all queries targeting Google Dataset Search; "
            "mix broad and specific; avoid overly long keyword chains; no explanations.\n"
            "Topic: {topic}\nTask: {task}\nIdeaTitle: {idea_title}\nIdeaContext: {idea_context}\n"
            "Domain: {domain}\nDataType: {data_type}\nModalities: {modalities}\nEvaluationAxes: {evaluation_axes}\n"
            "ReferenceTitles: {reference_titles}\n"
        )

    reference_titles: List[str] = []
    for ref in references[:3]:
        ref_title = (ref.get("title") or "").strip()
        if ref_title:
            reference_titles.append(ref_title)

    llm_payload = llm_prompt.format(
        topic=topic or "",
        task=task or "",
        idea_title=idea_title or "",
        idea_context=idea_context or "",
        domain=domain or "",
        data_type=data_type or "",
        modalities=json.dumps(modalities[:3], ensure_ascii=False),
        evaluation_axes=json.dumps(evaluation_axes[:3], ensure_ascii=False),
        reference_titles=json.dumps(reference_titles, ensure_ascii=False),
    )
    llm_queries: List[str] = []
    try:
        response = chat_fn(llm_payload, temperature=0.1, max_tokens=512, model=model)
        _log_llm_output("dataset_query_generation", response, logger)
        parsed = parse_json_response(response)
        llm_queries = _sanitize_queries(parsed, limit=10)
        print(f"[Debug] {llm_queries}")
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Dataset query generation failed; falling back to templates: %s", exc)

    base_queries = _heuristic_queries()
    ordered = []
    seen_ordered: Set[str] = set()
    for query in base_queries + llm_queries:
        if query and query not in seen_ordered:
            ordered.append(query)
            seen_ordered.add(query)
    return ordered[:12]


def merge_dataset_candidates(search_text: str) -> List[Dict[str, Any]]:
    raw = parse_search_results_limited(search_text, per_query=3)
    merged: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        title = derive_label(item.get("title", ""))
        url = item.get("url") or ""
        snippet = (item.get("snippet") or "").strip()
        if not title and not url:
            continue
        key = normalize_candidate_key(title, url)
        if key in merged:
            existing = merged[key]
            if snippet and snippet not in (existing.get("snippet") or ""):
                existing["snippet"] = (existing.get("snippet") or "") + "\n" + snippet
            if not existing.get("url") and url:
                existing["url"] = url
            continue
        merged[key] = {
            "title": title,
            "url": url,
            "snippet": snippet,
        }
    return list(merged.values())


def score_dataset_candidate(
    idea_card: Dict[str, Any],
    candidate: Dict[str, Any],
    chat_fn,
    model: str,
    logger,
) -> Dict[str, Any]:
    evidence_text = candidate.get("snippet") or ""
    page_text = fetch_url_text(candidate.get("url") or "", logger=logger, max_chars=2500)
    if page_text:
        evidence_text = f"{evidence_text}\n{page_text[:2000]}"

    prompt = (
        "You are extracting dataset evidence. Given the idea card and candidate text, "
        "return JSON with: dataset_name, usage, access, license, dataset_type, evidence_snippets (list), "
        "match_score (0-5), scale_score (0-5), availability_score (0-5)."
        "\nIdeaCard: {idea_card}\nCandidateTitle: {title}\nCandidateUrl: {url}\nText: {text}\n"
    )
    payload = prompt.format(
        idea_card=json.dumps(idea_card, ensure_ascii=False, indent=2),
        title=candidate.get("title", ""),
        url=candidate.get("url", ""),
        text=evidence_text[:3500],
    )
    try:
        response = chat_fn(payload, temperature=0.1, max_tokens=700, model=model)
        _log_llm_output("dataset_candidate_scoring", response, logger)
        parsed = parse_json_response(response)
        if isinstance(parsed, dict):
            candidate.update(parsed)
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Dataset candidate scoring failed: %s", exc)

    match = candidate.get("match_score")
    scale = candidate.get("scale_score")
    availability = candidate.get("availability_score")
    try:
        total = float(match or 0) + float(scale or 0) + float(availability or 0)
    except Exception:
        total = 0.0
    url = candidate.get("url") or ""
    if "huggingface.co/datasets" in url:
        total += 0.5
    if "kaggle.com/datasets" in url:
        total += 0.4
    if "paperswithcode.com/dataset" in url:
        total += 0.4
    if "datasetsearch.research.google.com" in url:
        total += 0.2
    candidate["total_score"] = total
    candidate["evidence_snippets"] = candidate.get("evidence_snippets") or []
    return candidate


def select_dataset_candidates(
    candidates: List[Dict[str, Any]],
    target: int = 5,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    candidates_sorted = sorted(candidates, key=lambda c: c.get("total_score", 0), reverse=True)
    output: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for cand in candidates_sorted:
        key = normalize_candidate_key(cand.get("title") or "", cand.get("url") or "")
        if key in seen:
            continue
        seen.add(key)
        output.append(cand)
        if len(output) >= target:
            break
    return output


def merge_baseline_candidates(search_text: str) -> List[Dict[str, Any]]:
    raw = parse_search_results_limited(search_text, per_query=3)
    merged: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        title = derive_label(item.get("title", ""))
        url = item.get("url") or ""
        snippet = (item.get("snippet") or "").strip()
        if not title and not url:
            continue
        key = normalize_candidate_key(title, url)
        if key in merged:
            existing = merged[key]
            if snippet and snippet not in (existing.get("snippet") or ""):
                existing["snippet"] = (existing.get("snippet") or "") + "\n" + snippet
            if not existing.get("url") and url:
                existing["url"] = url
            continue
        merged[key] = {
            "title": title,
            "url": url,
            "snippet": snippet,
        }
    return list(merged.values())


def score_baseline_candidate(
    idea_card: Dict[str, Any],
    candidate: Dict[str, Any],
    chat_fn,
    model: str,
    logger,
) -> Dict[str, Any]:
    evidence_text = candidate.get("snippet") or ""
    page_text = fetch_url_text(candidate.get("url") or "", logger=logger)
    if page_text:
        evidence_text = f"{evidence_text}\n{page_text[:2000]}"
    prompt = (
        "You are extracting baseline evidence. Given the idea card and candidate text, "
        "return JSON with: method_family, setting, reproducibility, evidence_snippets (list), "
        "match_score (0-5), representativeness_score (0-5), reproducibility_score (0-5)."
        "\nIdeaCard: {idea_card}\nCandidateTitle: {title}\nCandidateUrl: {url}\nText: {text}\n"
    )
    payload = prompt.format(
        idea_card=json.dumps(idea_card, ensure_ascii=False, indent=2),
        title=candidate.get("title", ""),
        url=candidate.get("url", ""),
        text=evidence_text[:3500],
    )
    try:
        response = chat_fn(payload, temperature=0.1, max_tokens=800, model=model)
        _log_llm_output("baseline_candidate_scoring", response, logger)
        parsed = parse_json_response(response)
        if isinstance(parsed, dict):
            candidate.update(parsed)
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Baseline candidate scoring failed: %s", exc)

    match = candidate.get("match_score")
    represent = candidate.get("representativeness_score")
    repro = candidate.get("reproducibility_score")
    try:
        total = float(match or 0) + float(represent or 0) + float(repro or 0)
    except Exception:
        total = 0.0
    url = candidate.get("url") or ""
    if "github.com" in url:
        total += 0.5
    if "arxiv.org" in url or "export.arxiv.org" in url:
        total += 0.3
    evidence_lower = evidence_text.lower()
    if "github.com" in evidence_lower and "github.com" not in url:
        total += 0.2
    if ("github.com" not in url and "arxiv.org" not in url and "export.arxiv.org" not in url and
            "github.com" not in evidence_lower and "arxiv.org" not in evidence_lower):
        total -= 0.8
    candidate["total_score"] = total
    candidate["evidence_snippets"] = candidate.get("evidence_snippets") or []
    return candidate


def select_baseline_candidates(
    candidates: List[Dict[str, Any]],
    idea_card: Dict[str, Any],
    target: int = 5,
) -> List[Dict[str, Any]]:
    axes: List[str] = []
    for key in ("benchmarks", "method_family", "key_components", "evaluation_axes"):
        for item in idea_card.get(key) or []:
            text = str(item).strip()
            if text:
                axes.append(text)
    seen: Set[str] = set()
    coverage_axes: List[str] = []
    for axis in axes:
        key = axis.lower()
        if key not in seen:
            seen.add(key)
            coverage_axes.append(axis)
    if not coverage_axes:
        return candidates[:target]

    def _covers_axis(candidate: Dict[str, Any], axis: str) -> bool:
        haystack = " ".join(
            str(candidate.get(k) or "")
            for k in ("title", "snippet", "method_family", "setting")
        ).lower()
        return axis.lower() in haystack

    selected: List[Dict[str, Any]] = []
    covered: Set[str] = set()
    remaining = candidates[:]
    while remaining and len(selected) < target:
        best_idx = None
        best_gain = -1
        best_score = -1
        for idx, cand in enumerate(remaining):
            gain = 0
            for axis in coverage_axes:
                if axis.lower() in covered:
                    continue
                if _covers_axis(cand, axis):
                    gain += 1
            score = cand.get("total_score") or 0
            if gain > best_gain or (gain == best_gain and score > best_score):
                best_idx = idx
                best_gain = gain
                best_score = score
        if best_idx is None:
            break
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        for axis in coverage_axes:
            if _covers_axis(chosen, axis):
                covered.add(axis.lower())
    return selected
