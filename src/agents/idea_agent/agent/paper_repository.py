from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional

from omegaconf import OmegaConf

from agents.survey_agent.modules.work_collector import WorkCollector
from agents.survey_agent.modules.work_analyzer import WorkAnalyzer


class PaperRepository:
    """
    Thin wrapper around the survey agent's WorkCollector and WorkAnalyzer modules.
    It exposes a lightweight interface for the idea agent to ensure that papers are
    downloaded, parsed into Markdown, and summarized via the keynote cache.
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
        self.work_analyzer = WorkAnalyzer(self.config, self.work_collector)

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
        Resolve the cache path to an absolute directory so that WorkCollector and
        WorkAnalyzer can find/download parsed papers deterministically.
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

        self.work_collector.download_and_parse_papers(unique_ids)

        results: Dict[str, Dict[str, object]] = {}
        for pid in unique_ids:
            try:
                keynote = self.work_analyzer.get_paper_keynote(pid)
            except Exception as exc:  # pragma: no cover - defensive
                if self.logger:
                    self.logger.warning(
                        "Failed to generate keynote for paper %s: %s", pid, exc
                    )
                keynote = None
            results[pid] = {"keynote": keynote}
        return results

    def get_markdown(self, paper_id: str) -> str:
        """
        Retrieve the parsed Markdown content for a paper directly from WorkAnalyzer.
        """
        return self.work_analyzer.get_paper_raw_markdown(paper_id)
