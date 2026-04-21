"""Repository for survey RAG, survey keynotes, and core-node lookup."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import diskcache as dc
from omegaconf import OmegaConf

from src.agents.survey_agent.modules.outcome_RAG import OutcomeRAG
from src.agents.survey_agent.modules.paper_graph_retriever import PaperGraphRetriever
from src.agents.survey_agent.modules.work_collector import WorkCollector
from src.agents.survey_agent.utils.topic_survey_storage import build_survey_artifact_paths
from src.agents.idea_agent.utils.core.progress import iter_with_progress
from src.agents.survey_agent.utils.utils import get_hash


_REPO_ROOT = Path(__file__).resolve().parents[5]


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _clone_config(config: Any) -> Any:
    if config is None:
        return None
    if isinstance(config, (str, Path)):
        return OmegaConf.load(str(config))
    if OmegaConf.is_config(config):
        return OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    return OmegaConf.create(config)


def _extract_survey_config(config: Any) -> Any:
    cloned = _clone_config(config)
    if cloned is None:
        return None
    survey_section = OmegaConf.select(cloned, "survey")
    if survey_section is None:
        return cloned
    return OmegaConf.create(OmegaConf.to_container(survey_section, resolve=False))


class PaperRepository:
    """Access survey RAG plus SurveyAgent keynote retrieval."""

    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        config: Optional[object] = None,
        rag_config: Optional[object] = None,
        logger=None,
    ) -> None:
        self.logger = logger
        self._config_path = self._resolve_config_path(config_path, config)
        survey_config = _extract_survey_config(config)
        self.config = survey_config if survey_config is not None else OmegaConf.load(self._config_path)
        self.work_collector = WorkCollector(self.config)
        self.rag_config = self._build_rag_config(rag_config, self.config)
        self._repair_runtime_survey_paths(self.rag_config)
        self._outcome_rag: Optional[OutcomeRAG] = None
        self._graph_server: Optional[Any] = None
        self._paper_graph_retriever: Optional[PaperGraphRetriever] = None
        self._paper_keynote_cache: Optional[dc.Cache] = None

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
                Path(__file__).resolve().parents[3]
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

    def _build_rag_config(self, rag_config: Optional[object], survey_config: Optional[object]) -> Any:
        base_config = _extract_survey_config(survey_config)
        if base_config is None:
            base_config = OmegaConf.create()
        rag_override = _extract_survey_config(rag_config)
        if rag_override is None:
            return base_config

        merged = OmegaConf.merge(base_config, rag_override)

        # Keep the active survey runtime paths from src/config/default.yaml::survey,
        # while still allowing the RAG preset to override retrieval/model knobs.
        base_basic_info = OmegaConf.select(base_config, "BasicInfo")
        if base_basic_info is not None:
            OmegaConf.update(
                merged,
                "BasicInfo",
                OmegaConf.to_container(base_basic_info, resolve=False),
                merge=False,
            )
        return merged

    def _repair_runtime_survey_paths(self, config: Any) -> None:
        current_md = OmegaConf.select(config, "BasicInfo.save_path")
        current_json = OmegaConf.select(config, "BasicInfo.save_json_path")
        if current_md and current_json:
            md_exists = Path(str(current_md)).expanduser().exists()
            json_exists = Path(str(current_json)).expanduser().exists()
            if md_exists and json_exists:
                return

        topic = _as_text(OmegaConf.select(config, "BasicInfo.topic"))
        fallback = build_survey_artifact_paths(topic, config=config)
        if not fallback.exists():
            return

        OmegaConf.update(config, "BasicInfo.base_dir", str(fallback.base_dir), merge=False)
        OmegaConf.update(config, "BasicInfo.save_path", str(fallback.markdown_path), merge=False)
        OmegaConf.update(config, "BasicInfo.save_json_path", str(fallback.json_path), merge=False)
        OmegaConf.update(config, "BasicInfo.evaluation_save_path", str(fallback.evaluation_path), merge=False)

    def _get_outcome_rag(self) -> OutcomeRAG:
        if self._outcome_rag is None:
            self._outcome_rag = OutcomeRAG(self.rag_config, self.work_collector)
        return self._outcome_rag

    def _get_graph_server(self) -> Any:
        if self._graph_server is not None:
            return self._graph_server

        server_path = (_REPO_ROOT / "graph" / "server.py").resolve()
        spec = importlib.util.spec_from_file_location("researchagent_graph_server", server_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load graph server module from {server_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._graph_server = module
        return module

    def _get_paper_graph_retriever(self) -> PaperGraphRetriever:
        if self._paper_graph_retriever is None:
            self._paper_graph_retriever = PaperGraphRetriever(
                self.config,
                data_manager=self.work_collector.data_manager,
            )
        return self._paper_graph_retriever

    def _resolve_cache_path(self) -> Path:
        cache_path = Path(str(OmegaConf.select(self.config, "BasicInfo.cache_path") or "./database")).expanduser()
        if cache_path.is_absolute():
            return cache_path.resolve()
        return (_REPO_ROOT / cache_path).resolve()

    def _get_paper_keynote_cache(self) -> dc.Cache:
        if self._paper_keynote_cache is None:
            cache_root = self._resolve_cache_path()
            cache_root.mkdir(parents=True, exist_ok=True)
            self._paper_keynote_cache = dc.Cache(str(cache_root / "paper_keynotes"))
        return self._paper_keynote_cache

    def _normalize_keynote_text(self, keynote: Any) -> str:
        if isinstance(keynote, list):
            keynote = keynote[0] if keynote else ""
        return _as_text(keynote)

    def _graph_node_id_for_paper_id(self, paper_id: str) -> str:
        keynote_retriever = self._get_paper_graph_retriever()
        try:
            if keynote_retriever.search_by_node_id(paper_id, limit=1):
                return paper_id
        except Exception:
            pass

        try:
            title = keynote_retriever.data_manager.get_paper_title(paper_id)
        except Exception:
            return ""

        try:
            return keynote_retriever.title_to_id(title)
        except Exception:
            return ""

    def _load_graph_stored_keynote(self, paper_id: str) -> str:
        keynote_retriever = self._get_paper_graph_retriever()
        node_id = self._graph_node_id_for_paper_id(paper_id)
        if not node_id:
            return ""

        cache_key = f"keynote_{node_id}"
        cached_keynote = keynote_retriever.graph_keynotes_cache.get(cache_key)
        if cached_keynote and keynote_retriever._validate_keynotes(cached_keynote, "cache"):
            return self._normalize_keynote_text(cached_keynote)

        db_keynote = keynote_retriever._read_keynote_from_db(node_id)
        if db_keynote and keynote_retriever._validate_keynotes(db_keynote, "SQL keynotes"):
            keynote_retriever.graph_keynotes_cache[cache_key] = db_keynote
            return self._normalize_keynote_text(db_keynote)

        return ""

    def _load_ds_stored_keynote(self, paper_id: str) -> str:
        cache = self._get_paper_keynote_cache()
        cache_key = get_hash(paper_id)
        if cache_key not in cache:
            return ""

        payload = cache[cache_key]
        if isinstance(payload, dict):
            return self._normalize_keynote_text(payload.get("keynote"))
        return self._normalize_keynote_text(payload)

    def _load_stored_keynote(self, paper_id: str) -> str:
        keynote = self._load_graph_stored_keynote(paper_id)
        if keynote:
            return keynote

        return self._load_ds_stored_keynote(paper_id)

    def retrieve_outcome_rag(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "content",
        alpha: float = 0.5,
        cite_top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        rag = self._get_outcome_rag()
        if rag.embeddings is None:
            rag.build_index()
        raw_hits = rag.retrieve(
            query=query,
            top_k=top_k,
            mode=mode,
            alpha=alpha,
            cite_top_k=cite_top_k,
        )
        return [self._normalize_rag_hit(hit) for hit in raw_hits or []]

    def get_core_node(self, node_id: str) -> Dict[str, Any]:
        server = self._get_graph_server()
        payload = server.get_node(str(node_id))
        if not isinstance(payload, dict) or not payload.get("found"):
            return {}
        node = payload.get("node")
        return dict(node) if isinstance(node, dict) else {}

    def retrieve_keynotes_by_paper_ids(
        self,
        references: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        keynote_references: List[Dict[str, Any]] = []
        seen_paper_ids: set[str] = set()
        reference_list = list(references or [])
        for reference in iter_with_progress(
            reference_list,
            description="Reading keynotes",
            total=len(reference_list),
        ):
            paper_id = _as_text(reference.get("paper_id"))
            title = _as_text(reference.get("title"))
            if not paper_id or paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper_id)
            keynote = self._load_stored_keynote(paper_id)
            if not keynote:
                if self.logger is not None:
                    self.logger.info(
                        "Skipping paper_id=%s because no stored survey keynote was found.",
                        paper_id,
                    )
                continue
            keynote_references.append(
                self._normalize_keynote_reference(
                    paper_id=paper_id,
                    title=title,
                    keynote=keynote,
                )
            )
        return keynote_references

    def _normalize_keynote_reference(
        self,
        *,
        paper_id: str,
        title: str,
        keynote: str,
    ) -> Dict[str, Any]:
        return {
            "paper_id": paper_id,
            "node_id": paper_id,
            "title": title,
            "paper_title": title,
            "summary": "",
            "insight": "",
            "keynote": _as_text(keynote),
            "authors": [],
            "source": "survey_keynote",
            "source_keywords": title,
            "paper_domain": "",
            "venue": "",
            "year": "",
            "reference_mode": "raw_keynote",
        }

    def _normalize_rag_hit(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(hit) if isinstance(hit, dict) else {}
        subsection_title = _as_text(normalized.get("subsection_title") or normalized.get("title"))
        citation_entries: List[Dict[str, str]] = []
        seen_paper_ids: set[str] = set()
        for raw_entry in normalized.get("citation_entries") or []:
            paper_id = _as_text(raw_entry.get("paper_id"))
            title = _as_text(raw_entry.get("title"))
            if not paper_id or not title or paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper_id)
            citation_entries.append({"paper_id": paper_id, "title": title})

        paper_titles: List[str] = []
        seen_titles: set[str] = set()
        for raw_title in normalized.get("paper_titles") or normalized.get("citations") or [entry["title"] for entry in citation_entries]:
            cleaned = _as_text(raw_title)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            paper_titles.append(cleaned)

        normalized["subsection_title"] = subsection_title
        normalized["citation_entries"] = citation_entries
        normalized["paper_titles"] = paper_titles
        # Keep backward-compatible access for prompt views / logs.
        normalized["citations"] = paper_titles
        return normalized
