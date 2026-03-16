"""Graph-backed repository for survey RAG and Core-node retrieval."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from omegaconf import OmegaConf

from src.agents.survey_agent.modules.outcome_RAG import OutcomeRAG
from src.agents.survey_agent.modules.work_collector import WorkCollector


_REPO_ROOT = Path(__file__).resolve().parents[5]


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


class PaperRepository:
    """Access survey RAG plus graph.db-backed Core references."""

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
        self.work_collector = WorkCollector(self.config)
        self.rag_config = OmegaConf.load(str(rag_config))
        self._outcome_rag: Optional[OutcomeRAG] = None
        self._graph_server: Optional[Any] = None

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
        return rag.retrieve(
            query=query,
            top_k=top_k,
            mode=mode,
            alpha=alpha,
            cite_top_k=cite_top_k,
        )

    def get_core_node(self, node_id: str) -> Dict[str, Any]:
        server = self._get_graph_server()
        payload = server.get_node(str(node_id))
        if not isinstance(payload, dict) or not payload.get("found"):
            return {}
        node = payload.get("node")
        return dict(node) if isinstance(node, dict) else {}

    def search_core_nodes_by_titles(
        self,
        titles: Iterable[str],
        *,
        limit_per_title: int = 1,
    ) -> List[Dict[str, Any]]:
        server = self._get_graph_server()
        references: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for title in titles or []:
            query = _as_text(title)
            if not query:
                continue
            payload = server.search_simple(q=query, node_type="Core", limit=limit_per_title)
            results = payload.get("results") if isinstance(payload, dict) else []
            for node in results or []:
                reference = self._normalize_core_reference(
                    node,
                    source="survey_citation",
                    source_keywords=query,
                )
                node_key = reference.get("node_id")
                if node_key and node_key not in seen:
                    seen.add(node_key)
                    references.append(reference)
        return references

    def search_core_nodes_by_query(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        server = self._get_graph_server()
        payload = server.search_simple(q=query, node_type="Core", limit=limit)
        results = payload.get("results") if isinstance(payload, dict) else []

        references: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for node in results or []:
            reference = self._normalize_core_reference(
                node,
                source="graph_query",
                source_keywords=query,
            )
            node_key = reference.get("node_id")
            if node_key and node_key not in seen:
                seen.add(node_key)
                references.append(reference)
        return references

    def _normalize_core_reference(
        self,
        node: Dict[str, Any],
        *,
        source: str,
        source_keywords: str,
    ) -> Dict[str, Any]:
        node_id = _as_text(node.get("id") or node.get("node_id"))
        paper_title = _as_text(node.get("paper_title"))
        full_name = _as_text(node.get("full_name"))
        label = _as_text(node.get("label"))
        summary = _as_text(node.get("summary"))
        insight = _as_text(node.get("insight"))
        components = node.get("components") if isinstance(node.get("components"), list) else []
        component_names = [
            _as_text(component.get("name") or component.get("label"))
            for component in components[:4]
            if isinstance(component, dict) and _as_text(component.get("name") or component.get("label"))
        ]
        summary_parts = [part for part in [summary, f"Insight: {insight}" if insight else ""] if part]
        if component_names:
            summary_parts.append("Components: " + ", ".join(component_names))
        summary_text = " ".join(summary_parts) or "No summary available."
        title = paper_title or full_name or label or node_id
        return {
            "paper_id": node_id,
            "node_id": node_id,
            "title": title,
            "paper_title": paper_title or title,
            "summary": summary_text,
            "insight": insight,
            "authors": [],
            "source": source,
            "source_keywords": source_keywords,
            "paper_domain": _as_text(node.get("paper_domain")),
            "full_name": full_name,
            "label": label or title,
            "components": components,
        }
