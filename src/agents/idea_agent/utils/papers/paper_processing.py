from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import diskcache as dc

from src.agents.survey_agent.modules.pe import PAPER_DEEP_READING
from src.agents.survey_agent.utils.api_call import ChatAgent
from src.agents.survey_agent.utils.utils import extract_json


def resolve_paper_records(work_collector, paper_ids: Iterable[str], logger=None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for pid in paper_ids or []:
        pid = (pid or "").strip()
        if not pid:
            continue
        metadata = None
        try:
            metadata = work_collector.semantic_scholar_api.get_paper_details(
                pid,
                fields="title,externalIds,openAccessPdf,authors",
            )
        except Exception as exc:  # pragma: no cover - defensive
            if logger:
                logger.warning("Failed to fetch metadata for %s: %s", pid, exc)

        if metadata:
            record = _normalize_paper_record(metadata, pid)
        else:
            record = _fallback_paper_record(pid)
        if record:
            record["_requested_id"] = pid
            records.append(record)
    return records


def _normalize_paper_record(
    metadata: Dict[str, object], requested_id: str
) -> Dict[str, object]:
    external_ids = metadata.get("externalIds") or {}
    if not isinstance(external_ids, dict):
        external_ids = {}
    open_access = metadata.get("openAccessPdf") or {}
    if not isinstance(open_access, dict):
        open_access = {}
    record: Dict[str, object] = {
        "paperId": metadata.get("paperId") or requested_id,
        "title": metadata.get("title") or requested_id,
        "externalIds": external_ids,
        "openAccessPdf": open_access,
    }
    if metadata.get("authors"):
        record["authors"] = metadata["authors"]

    arxiv_id = external_ids.get("ArXiv")
    if not arxiv_id and _looks_like_arxiv(record["paperId"]):
        arxiv_id = _normalize_arxiv_id(record["paperId"])
        if arxiv_id:
            external_ids = {**external_ids, "ArXiv": arxiv_id}
            record["externalIds"] = external_ids

    if arxiv_id and not open_access.get("url"):
        open_access["url"] = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"
        record["openAccessPdf"] = open_access

    return record


def _fallback_paper_record(paper_id: str) -> Dict[str, object]:
    external_ids: Dict[str, object] = {}
    open_access: Dict[str, object] = {}
    arxiv_id = _normalize_arxiv_id(paper_id) if _looks_like_arxiv(paper_id) else None
    if arxiv_id:
        external_ids["ArXiv"] = arxiv_id
        open_access["url"] = f"https://export.arxiv.org/pdf/{arxiv_id}.pdf"

    record: Dict[str, object] = {
        "paperId": paper_id,
        "title": paper_id,
        "externalIds": external_ids,
        "openAccessPdf": open_access or {},
    }
    return record


def _looks_like_arxiv(paper_id: str) -> bool:
    if not paper_id:
        return False
    lowered = paper_id.lower()
    return lowered.startswith("arxiv:") or "." in paper_id


def _normalize_arxiv_id(paper_id: str) -> str:
    if not paper_id:
        return ""
    cleaned = paper_id.strip()
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    return cleaned.strip()


class IdeaPaperParser:
    """Lightweight adapter around WorkCollector to ensure parsed markdown exists."""

    def __init__(self, config, work_collector, logger=None) -> None:
        self.config = config
        self.work_collector = work_collector
        self.logger = logger
        self.cache_path = Path(self.config.BasicInfo.cache_path).resolve()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.parsed_root = self.cache_path / "parsed_papers"
        self.parsed_root.mkdir(parents=True, exist_ok=True)
        self.id_aliases: Dict[str, str] = {}

    def download_and_parse(self, papers: Iterable) -> None:
        if not papers:
            return
        normalized: List[Dict[str, object]] = []
        missing_ids: List[str] = []
        for paper in papers:
            if isinstance(paper, dict):
                normalized.append(paper)
            elif isinstance(paper, str):
                pid = paper.strip()
                if pid:
                    missing_ids.append(pid)
            else:
                continue

        if missing_ids:
            normalized.extend(
                resolve_paper_records(self.work_collector, missing_ids, self.logger)
            )

        if not normalized:
            return

        canonical_records: List[Dict[str, object]] = []
        for record in normalized:
            canonical_id = self._canonical_paper_id(record)
            requested = record.pop("_requested_id", None) or canonical_id
            self.id_aliases[requested] = canonical_id
            canonical_records.append(record)

        try:
            self.work_collector.download_and_parse_papers(canonical_records)
        except Exception as exc:  # pragma: no cover - defensive I/O
            if self.logger:
                self.logger.warning("Failed to download/parse papers: %s", exc)

    def _markdown_path(self, paper_id: str) -> Path:
        return (
            self.cache_path
            / "parsed_papers"
            / paper_id
            / "auto"
            / f"{paper_id}.md"
        )

    def get_markdown(self, paper_id: str) -> str:
        canonical_id = self._resolve_canonical_id(paper_id)
        md_path = self._markdown_path(canonical_id)
        if not md_path.exists():
            self._clear_stale_parsed_folder(canonical_id)
            self.download_and_parse([canonical_id])
            if not md_path.exists():
                if self.logger:
                    self.logger.warning(
                        "Markdown still missing for %s even after parsing.", canonical_id
                    )
                return "Fail to Get Content"
        return md_path.read_text(encoding="utf-8")

    def _clear_stale_parsed_folder(self, paper_id: str) -> None:
        parsed_dir = self.parsed_root / paper_id
        if parsed_dir.exists():
            try:
                shutil.rmtree(parsed_dir)
            except OSError as exc:
                if self.logger:
                    self.logger.warning(
                        "Failed to clear stale parse directory %s: %s",
                        parsed_dir,
                        exc,
                    )

    def _canonical_paper_id(self, record: Dict[str, object]) -> str:
        external_ids = record.get("externalIds") or {}
        if isinstance(external_ids, dict) and external_ids.get("ArXiv"):
            return external_ids["ArXiv"]
        return record.get("paperId") or record.get("title") or ""

    def _resolve_canonical_id(self, requested_id: str) -> str:
        requested_id = (requested_id or "").strip()
        if not requested_id:
            return requested_id
        canonical = self.id_aliases.get(requested_id)
        if canonical:
            return canonical

        # Attempt to resolve via metadata lookup and cache the alias.
        records = resolve_paper_records(self.work_collector, [requested_id], self.logger)
        if records:
            record = records[0]
            canonical = self._canonical_paper_id(record)
            if canonical:
                self.id_aliases[requested_id] = canonical
                return canonical
        return requested_id


class IdeaPaperAnalyzer:
    """Summarize parsed markdown into keynotes without survey graph logic."""

    def __init__(self, config, parser: IdeaPaperParser, logger=None) -> None:
        self.config = config
        self.parser = parser
        self.logger = logger
        self.chat_agent = ChatAgent(config)
        cache_root = Path(self.config.BasicInfo.cache_path).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        self.cache = dc.Cache(str(cache_root / "idea_paper_keynotes"))

        analyzer_cfg = getattr(self.config.ModuleInfo, "WorkAnalyzer", None)
        self.temperature = getattr(analyzer_cfg, "paper_reading_temperature", 1.0)
        self.max_retry = getattr(analyzer_cfg, "paper_reading_max_retry", 3)

        max_ctx = getattr(self.config.APIInfo, "llm_max_context_length", 128000)
        overhead = getattr(
            self.config.APIInfo, "llm_max_context_overhead_length", 20000
        )
        self.allowed_tokens = max(1024, max_ctx - overhead)

    def ensure_keynotes(self, paper_ids: Iterable[str]) -> Dict[str, Dict[str, object]]:
        if not paper_ids:
            return {}
        missing: List[str] = []
        for pid in paper_ids:
            if pid not in self.cache:
                missing.append(pid)
        if missing:
            self._read_and_cache(missing)
        results: Dict[str, Dict[str, object]] = {}
        for pid in paper_ids:
            cached = self.cache.get(pid)
            if cached:
                results[pid] = cached
        return results

    def _read_and_cache(self, paper_ids: List[str], retry: int = 1) -> None:
        if not paper_ids or retry > self.max_retry:
            return

        tasks: List[tuple[str, str, str]] = []
        for pid in paper_ids:
            markdown = self.parser.get_markdown(pid)
            if not markdown or markdown.strip() == "Fail to Get Content":
                if self.logger:
                    self.logger.warning(
                        "Skipping keynote generation for %s due to missing markdown.",
                        pid,
                    )
                continue
            truncated = self.chat_agent.truncate_text(pid, markdown, self.allowed_tokens)
            prompt = PAPER_DEEP_READING.format(paper_markdown_text=truncated)
            excerpt = markdown[:2000]
            tasks.append((pid, prompt, excerpt))

        if not tasks:
            return

        prompts = [task[1] for task in tasks]
        try:
            responses = self.chat_agent.batch_remote_chat(
                prompts,
                temperature=self.temperature,
                desc="IdeaAgent paper reading",
            )
        except Exception as exc:  # pragma: no cover - network
            if self.logger:
                self.logger.error(
                    "Paper reading batch failed on attempt %d: %s", retry, exc
                )
            if retry >= self.max_retry:
                self._store_fallback_keynotes(tasks)
                return
            return self._read_and_cache(paper_ids, retry=retry + 1)

        for (pid, _, excerpt), response in zip(tasks, responses):
            keynote = self._safe_extract_json(response)
            self.cache[pid] = {"paper_id": pid, "keynote": keynote}

    def _store_fallback_keynotes(self, tasks: List[tuple[str, str, str]]) -> None:
        for pid, _, excerpt in tasks:
            self.cache[pid] = {
                "paper_id": pid,
                "keynote": self._fallback_keynote(pid, excerpt),
            }

    def _safe_extract_json(self, text: str) -> Dict[str, object]:
        try:
            payload = extract_json(text)
            return payload if isinstance(payload, dict) else {"content": payload}
        except Exception:
            if self.logger:
                self.logger.warning(
                    "Falling back to raw response for keynote due to parse error."
                )
            return {"raw_response": text.strip()[:2000]}

    def _fallback_keynote(self, paper_id: str, excerpt: str) -> Dict[str, object]:
        preview = (excerpt or "").strip()
        if not preview:
            preview = "Markdown unavailable or parsing failed."
        return {
            "tldr": (
                "LLM summarization unavailable due to API errors; using markdown excerpt."
            ),
            "paper_id": paper_id,
            "excerpt": preview,
        }
