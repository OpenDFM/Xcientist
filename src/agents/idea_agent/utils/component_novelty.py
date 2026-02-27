from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.agents.idea_agent.utils.graph_baseline_search import (
    score_component_explanation_embeddings_for_novelty,
    supports_component_novelty_scoring,
)

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional runtime dependency
    SentenceTransformer = None  # type: ignore[assignment]


def _clamp_novelty_score(value: Any) -> float:
    score = float(value)
    if math.isnan(score) or math.isinf(score):
        raise ValueError(f"Invalid novelty score: {value!r}")
    return max(0.0, min(5.0, score))


class ComponentNoveltyScorer:
    """Embed component explanations and delegate novelty scoring to paper-graph logic."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._disabled_error: Optional[str] = None

    def _resolve_model_source(self) -> str:
        local_path = Path(__file__).resolve().parents[1] / ".cache" / self.model_name
        return str(local_path) if local_path.exists() else self.model_name

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if SentenceTransformer is None:
            raise RuntimeError("sentence_transformers is unavailable")
        self._model = SentenceTransformer(self._resolve_model_source())
        return self._model

    def score(self, components_with_explanations: Sequence[Dict[str, str]]) -> float:
        if self._disabled_error:
            raise RuntimeError(self._disabled_error)
        if not supports_component_novelty_scoring():
            self._disabled_error = "paper-graph novelty hook is not implemented"
            raise RuntimeError(self._disabled_error)
        if not components_with_explanations:
            raise ValueError("No component explanations were provided for novelty scoring")
        texts: List[str] = []
        for item in components_with_explanations:
            if not isinstance(item, dict):
                continue
            explanation = str(item.get("explanation", "")).strip()
            component = str(item.get("component", "")).strip()
            texts.append(explanation or component)
        texts = [text for text in texts if text]
        if not texts:
            raise ValueError("Component explanations were empty")
        try:
            model = self._get_model()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            novelty = score_component_explanation_embeddings_for_novelty(
                explanation_embeddings=embeddings.tolist()
                if hasattr(embeddings, "tolist")
                else list(embeddings),
                components_with_explanations=components_with_explanations,
            )
            return _clamp_novelty_score(novelty)
        except Exception as exc:
            self._disabled_error = str(exc)
            raise
