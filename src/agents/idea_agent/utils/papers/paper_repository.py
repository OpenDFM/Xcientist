"""Repository for survey RAG, survey keynotes, and core-node lookup."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import OmegaConf
from tqdm import tqdm

from src.agents.survey_agent.modules.outcome_RAG import OutcomeRAG
from src.agents.survey_agent.modules.paper_graph_retriever import PaperGraphRetriever
from src.agents.survey_agent.modules.work_collector import WorkCollector


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
        memory_dir = (_REPO_ROOT / "src" / "agents" / "survey_agent" / "outputs" / "memory").resolve()
        fallback_paths = {
            "BasicInfo.base_dir": str(memory_dir),
            "BasicInfo.save_path": str(memory_dir / "survey.md"),
            "BasicInfo.save_json_path": str(memory_dir / "survey.json"),
            "BasicInfo.evaluation_save_path": str(memory_dir / "evaluation.txt"),
        }

        current_md = OmegaConf.select(config, "BasicInfo.save_path")
        current_json = OmegaConf.select(config, "BasicInfo.save_json_path")
        if current_md and current_json:
            md_exists = Path(str(current_md)).expanduser().exists()
            json_exists = Path(str(current_json)).expanduser().exists()
            if md_exists and json_exists:
                return

        fallback_md = Path(fallback_paths["BasicInfo.save_path"])
        fallback_json = Path(fallback_paths["BasicInfo.save_json_path"])
        if not (fallback_md.exists() and fallback_json.exists()):
            return

        for key, value in fallback_paths.items():
            OmegaConf.update(config, key, value, merge=False)

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
        keynote_retriever = self._get_paper_graph_retriever()
        keynote_references: List[Dict[str, Any]] = []
        seen_paper_ids: set[str] = set()
        for reference in tqdm(list(references or []), desc="Reading keynotes", dynamic_ncols=True):
            paper_id = _as_text(reference.get("paper_id"))
            title = _as_text(reference.get("title"))
            if not paper_id or paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper_id)
            keynotes, _ = keynote_retriever.get_paper_keynote([paper_id])
            keynote = keynotes[0]
            if not keynote:
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
