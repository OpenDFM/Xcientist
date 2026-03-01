from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

from omegaconf import OmegaConf

from src.agents.survey_agent.modules.work_collector import WorkCollector
from src.agents.survey_agent.modules.outcome_RAG import OutcomeRAG

from src.agents.idea_agent.utils.paper_processing import (
    IdeaPaperAnalyzer,
    IdeaPaperParser,
    resolve_paper_records,
)


class PaperRepository:
    """
    Thin wrapper around the survey agent's WorkCollector plus lightweight parsing
    utilities so the idea agent can download, parse, and summarize papers without
    invoking the survey agent's graph-heavy analyzer.
    """

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        config: Optional[object] = None,
        rag_config: Optional[object] = None,
        logger=None,
    ) -> None:
        self.logger = logger
        self._config_path = self._resolve_config_path(config_path, config)
        self.config = config or OmegaConf.load(self._config_path)
        self._normalize_cache_path()

        self.work_collector = WorkCollector(self.config)
        self.paper_parser = IdeaPaperParser(self.config, self.work_collector, logger)
        self.paper_analyzer = IdeaPaperAnalyzer(
            self.config, self.paper_parser, logger
        )
        self._outcome_rag: Optional[OutcomeRAG] = None

        assert rag_config is not None, "RAG config must be provided"
        self.rag_config = OmegaConf.load(str(rag_config))

    def _resolve_config_path(self, provided_path, config):
        if config is not None:
            if provided_path is None:
                return None
            return Path(provided_path).resolve()

        env_path = os.getenv("IDEA_AGENT_SURVEY_CONFIG")
        if provided_path:
            config_path = Path(provided_path)
        elif env_path:
            config_path = Path(env_path)
        else:
            config_path = (
                Path(__file__).resolve().parents[2]
                / "survey_agent"
                / "config"
                / "deep_survey.yaml"
            )

        config_path = config_path.resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"Survey config not found at {config_path}. "
                "Set IDEA_AGENT_SURVEY_CONFIG to override the default path."
            )
        return config_path

    def _normalize_cache_path(self) -> None:
        """
        Resolve the cache path to an absolute directory so downstream helpers can
        deterministically read/write parsed papers and summaries.
        """
        cache_path = Path(self.config.BasicInfo.cache_path)
        if not cache_path.is_absolute():
            base = (
                self._config_path.parent
                if self._config_path is not None
                else Path.cwd()
            )
            cache_path = (base / cache_path).resolve()
        resolved = str(cache_path)
        self.config.BasicInfo.cache_path = resolved
        os.makedirs(resolved, exist_ok=True)

    def prepare_papers(self, paper_ids: Iterable[str]) -> Dict[str, Dict[str, object]]:
        """
        Ensure that each provided paper ID has a parsed Markdown file and a keynote summary.

        Returns a mapping of paper_id -> {"keynote": keynote_dict}
        """
        unique_ids = []
        seen = set()
        for pid in paper_ids or []:
            normalized = (pid or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_ids.append(normalized)

        if not unique_ids:
            return {}

        papers = resolve_paper_records(self.work_collector, unique_ids, self.logger)
        self.paper_parser.download_and_parse(papers)
        keynotes = self.paper_analyzer.ensure_keynotes(unique_ids)

        results: Dict[str, Dict[str, object]] = {}
        for pid in unique_ids:
            try:
                keynote_entry = keynotes.get(pid)
            except Exception as exc:  # pragma: no cover - defensive
                if self.logger:
                    self.logger.warning(
                        "Failed to generate keynote for paper %s: %s", pid, exc
                    )
                keynote_entry = None
            keynote_value = None
            if isinstance(keynote_entry, dict):
                keynote_value = keynote_entry.get("keynote") or keynote_entry
            results[pid] = {"keynote": keynote_value}
        return results

    def get_markdown(self, paper_id: str) -> str:
        """Retrieve the parsed Markdown content for a paper via the local parser."""
        return self.paper_parser.get_markdown(paper_id)

    def retrieve_outcome_rag(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "content",
        alpha: float = 0.5,
        cite_top_k: Optional[int] = None,
    ):
        rag = self._get_outcome_rag()
        if rag.embeddings is None:
            rag.build_index()
        return rag.retrieve(
            query=query,
            top_k=top_k,
            mode=mode,
            alpha=alpha,
            cite_top_k=cite_top_k,
        )

    def search_papers_by_title(
        self,
        titles: Iterable[str],
        limit_per_title: int = 1,
    ) -> list[Dict[str, object]]:
        results: list[Dict[str, object]] = []
        seen = set()
        for title in titles or []:
            query = (title or "").strip()
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                response = self.work_collector.semantic_scholar_api.search_papers(
                    query=query,
                    fields="title,abstract,authors,year,url,tldr,paperId,externalIds,openAccessPdf",
                )
            except Exception as exc:
                if self.logger:
                    self.logger.warning("Semantic Scholar title search failed for %s: %s", query, exc)
                continue
            data = response.get("data", []) if isinstance(response, dict) else []
            if not data:
                continue
            picked = self._pick_best_title_match(query, data[: max(1, limit_per_title)])
            if picked:
                record = self._normalize_search_result(picked, query)
                if record:
                    results.append(record)
        return results

    def _get_outcome_rag(self) -> OutcomeRAG:
        if self._outcome_rag is None:
            self._outcome_rag = OutcomeRAG(self.rag_config, self.work_collector)
        return self._outcome_rag

    def _pick_best_title_match(self, query: str, candidates: list[Dict[str, object]]) -> Optional[Dict[str, object]]:
        lowered = query.strip().lower()
        for item in candidates:
            title = (item.get("title") or "").strip().lower()
            if title and title == lowered:
                return item
        return candidates[0] if candidates else None

    def _normalize_search_result(self, paper: Dict[str, object], source_query: str) -> Optional[Dict[str, object]]:
        if not isinstance(paper, dict):
            return None
        title = paper.get("title") or source_query
        authors_field = paper.get("authors", []) or []
        if isinstance(authors_field, list):
            authors = [a.get("name", str(a)) for a in authors_field if a]
        elif authors_field:
            authors = [str(authors_field)]
        else:
            authors = []
        tldr = paper.get("tldr")
        if isinstance(tldr, dict):
            tldr = tldr.get("text") or tldr.get("summary")
        return {
            "title": title,
            "abstract": paper.get("abstract") or "No abstract available.",
            "authors": authors,
            "year": paper.get("year"),
            "url": paper.get("url") or (paper.get("openAccessPdf") or {}).get("url"),
            "tldr": tldr,
            "paper_id": paper.get("paperId") or paper.get("paper_id"),
            "source_keywords": source_query,
        }
