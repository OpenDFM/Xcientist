"""Component novelty scoring utilities backed by embedding similarity."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from sentence_transformers import SentenceTransformer

from src.agents.idea_agent.agent.prompts.component_novelty_evaluation import (
    COMPONENT_NOVELTY_EVALUATION_PROMPT,
)
from src.agents.idea_agent.utils.mcts.mcts_helpers import parse_json_response
from src.agents.idea_agent.utils.papers.paper_graph_vector_store import (
    PaperGraphComponentVectorStore,
)


def _clamp_novelty_score(value: Any) -> float:
    score = float(value)
    if math.isnan(score) or math.isinf(score):
        raise ValueError(f"Invalid novelty score: {value!r}")
    return max(0.0, min(5.0, score))


def _clip_text(value: Any, limit: int = 6000) -> str:
    text = "" if value is None else str(value).strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(1, limit - 3)] + "..."


class ComponentNoveltyScorer:
    """Retrieve nearby paper-graph nodes from component explanations and ask an LLM for novelty."""

    def __init__(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        *,
        index_dir: Optional[str] = None,
        retrieval_top_k: int = 50,
        evidence_node_top_k: int = 5,
        evaluation_model: str = "gpt-5.2",
        evaluation_temperature: float = 0.001,
        evaluation_max_tokens: int = 2048,
        chat_fn: Optional[Callable[..., str]] = None,
        logger: Optional[Any] = None,
        log_sink: Optional[Callable[[str, str], None]] = None,
        vector_store: Optional[PaperGraphComponentVectorStore] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.index_dir = index_dir
        self.retrieval_top_k = max(1, int(retrieval_top_k))
        self.evidence_node_top_k = max(1, int(evidence_node_top_k))
        self.evaluation_model = evaluation_model
        self.evaluation_temperature = float(evaluation_temperature)
        self.evaluation_max_tokens = max(256, int(evaluation_max_tokens))
        self.chat_fn = chat_fn
        self.logger = logger
        self.log_sink = log_sink

        self._model: Optional[Any] = None
        self._vector_store: Optional[PaperGraphComponentVectorStore] = vector_store
        self._disabled_error: Optional[str] = None

    def _log(self, level: str, message: str, *args: Any) -> None:
        rendered = message % args if args else message
        if self.logger is not None:
            log_method = getattr(self.logger, level, None)
            if callable(log_method):
                log_method(rendered)
        if self.log_sink is not None:
            try:
                self.log_sink(level, rendered)
            except Exception:
                pass

    def _resolve_model_source(self) -> str:
        candidate = Path(str(self.model_name_or_path)).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        local_path = Path(__file__).resolve().parents[5] / "models" / str(self.model_name_or_path)
        return str(local_path) if local_path.exists() else str(self.model_name_or_path)

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._vector_store is not None:
            self._model = self._vector_store.get_model()
            return self._model
        self._model = SentenceTransformer(self._resolve_model_source())
        return self._model

    def _get_vector_store(self) -> PaperGraphComponentVectorStore:
        if self._vector_store is None:
            self._vector_store = PaperGraphComponentVectorStore(
                model_name_or_path=self.model_name_or_path,
                index_dir=self.index_dir,
                model=self._model,
            )
        if self._vector_store.size <= 0:
            self._vector_store.load(allow_stale_graph=True)
        return self._vector_store

    def _extract_components(
        self,
        state: Any,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        if hasattr(state, "component_inventory") and callable(state.component_inventory):
            components_with_explanations = state.component_inventory()
            payload = state.to_payload() if hasattr(state, "to_payload") and callable(state.to_payload) else {}
            return components_with_explanations, payload

        if isinstance(state, dict):
            components = state.get("components_with_explanations")
            if isinstance(components, list):
                return components, dict(state)

        if isinstance(state, list):
            return state, {}

        raise ValueError("Unsupported state payload for component novelty scoring")

    def _prepare_component_queries(
        self,
        components_with_explanations: Sequence[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        queries: List[Dict[str, str]] = []
        for item in components_with_explanations:
            if not isinstance(item, dict):
                continue
            component = str(item.get("component", "")).strip()
            explanation = str(item.get("explanation", "")).strip()
            query_text = explanation
            if not query_text:
                continue
            queries.append(
                {
                    "component": component or f"component_{len(queries)}",
                    "explanation": explanation,
                    "query_text": query_text,
                }
            )
        return queries

    def _embed_queries(self, query_payloads: Sequence[Dict[str, str]]) -> Any:
        model = self._get_model()
        return model.encode(
            [item["query_text"] for item in query_payloads],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _collect_evidence_nodes(
        self,
        store: PaperGraphComponentVectorStore,
        query_payloads: Sequence[Dict[str, str]],
        component_hits: Sequence[Sequence[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        node_counts: Counter[str] = Counter()
        node_best_scores: Dict[str, float] = {}
        node_support: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for query_index, hits in enumerate(component_hits):
            query_info = query_payloads[query_index] if query_index < len(query_payloads) else {}
            for hit in hits:
                node_id = str(hit.get("node_id") or "")
                if not node_id:
                    continue
                score = float(hit.get("score") or 0.0)
                node_counts[node_id] += 1
                prev = node_best_scores.get(node_id)
                if prev is None or score > prev:
                    node_best_scores[node_id] = score
                if len(node_support[node_id]) < 3:
                    node_support[node_id].append(
                        {
                            "query_component": query_info.get("component", ""),
                            "query_text": _clip_text(query_info.get("query_text", "")),
                            "matched_component": hit.get("component_name"),
                            "score": score,
                        }
                    )

        if not node_counts:
            return []

        ranked_node_ids = sorted(
            node_counts.keys(),
            key=lambda node_id: (
                -node_counts[node_id],
                -node_best_scores.get(node_id, 0.0),
                str(store.get_core_node(node_id).get("paper_title") or node_id).lower(),
            ),
        )

        evidence_nodes: List[Dict[str, Any]] = []
        for node_id in ranked_node_ids[: self.evidence_node_top_k]:
            core_node = store.get_core_node(node_id)
            evidence_nodes.append(
                {
                    "node_id": node_id,
                    "label": core_node.get("full_name") or core_node.get("label") or node_id,
                    "paper_title": core_node.get("paper_title") or "",
                    "match_count": int(node_counts[node_id]),
                    "best_score": float(node_best_scores.get(node_id, 0.0)),
                    "summary": _clip_text(core_node.get("summary") or ""),
                    "insight": _clip_text(core_node.get("insight") or ""),
                    "support": node_support.get(node_id, []),
                }
            )
        return evidence_nodes

    def _evaluate_with_llm(
        self,
        *,
        topic: str,
        idea_payload: Dict[str, Any],
        components_with_explanations: Sequence[Dict[str, str]],
        evidence_nodes: Sequence[Dict[str, Any]],
    ) -> float:
        if self.chat_fn is None:
            raise RuntimeError("chat_fn is unavailable for component novelty scoring")

        prompt = COMPONENT_NOVELTY_EVALUATION_PROMPT.format(
            topic=topic or "Unknown topic",
            idea_state=json.dumps(idea_payload, ensure_ascii=False, indent=2),
            components_with_explanations=json.dumps(
                list(components_with_explanations),
                ensure_ascii=False,
                indent=2,
            ),
            retrieval_top_k=self.retrieval_top_k,
            retrieved_nodes=json.dumps(list(evidence_nodes), ensure_ascii=False, indent=2),
        )
        response = self.chat_fn(
            prompt,
            model=self.evaluation_model,
            temperature=self.evaluation_temperature,
            max_output_tokens=self.evaluation_max_tokens,
            stage="component_novelty_evaluation",
        )
        payload = parse_json_response(response)
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        score = payload.get("rubric_score")
        if score is None:
            score = payload.get("perceived_novelty", payload.get("novelty"))
        if score is None:
            raise ValueError(f"Novelty evaluator missing rubric_score: {payload}")
        return _clamp_novelty_score(score)

    def score(self, state: Any, topic: str = "") -> float:
        if self._disabled_error:
            raise RuntimeError(self._disabled_error)

        components_with_explanations, idea_payload = self._extract_components(state)
        query_payloads = self._prepare_component_queries(components_with_explanations)
        if not query_payloads:
            raise ValueError("No component explanations were provided for novelty scoring")

        try:
            embeddings = self._embed_queries(query_payloads)
            store = self._get_vector_store()
        except Exception as exc:
            self._disabled_error = str(exc)
            raise

        component_hits = store.search_component_hits_by_vectors(
            query_embeddings=embeddings,
            top_k=self.retrieval_top_k,
        )
        evidence_nodes = self._collect_evidence_nodes(store, query_payloads, component_hits)
        if not evidence_nodes:
            raise ValueError("Component novelty retrieval returned no evidence nodes")

        self._log(
            "info",
            "[ComponentNovelty] Retrieved top evidence nodes: %s",
            ", ".join(
                f"{item['label']}#{item['match_count']}"
                for item in evidence_nodes
            ),
        )

        return self._evaluate_with_llm(
            topic=topic,
            idea_payload=idea_payload,
            components_with_explanations=query_payloads,
            evidence_nodes=evidence_nodes,
        )
