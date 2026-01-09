from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

from omegaconf import OmegaConf

from agents.survey_agent.modules.work_collector import WorkCollector

from agents.idea_agent.agent.paper_processing import (
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
