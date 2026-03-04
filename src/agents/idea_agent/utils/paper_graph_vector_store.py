from __future__ import annotations

import copy
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_FILE = "component_embeddings.npy"
_METADATA_FILE = "component_metadata.json"


def _load_sentence_transformer_cls():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("sentence_transformers is unavailable") from exc
    return SentenceTransformer


def _load_numpy():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("numpy is unavailable") from exc
    return np


def _default_graph_path() -> str:
    return str((Path(__file__).resolve().parent.parent / "paper_graph.gexf").resolve())


def _default_index_dir(graph_path: str) -> Path:
    graph_file = Path(graph_path).resolve()
    return graph_file.parent / f"{graph_file.stem}.component_index"


def _serialize_path_for_metadata(path_value: str, base_dir: Path) -> str:
    candidate = Path(str(path_value)).expanduser()
    if not candidate.is_absolute():
        return str(path_value)
    return os.path.relpath(str(candidate.resolve()), start=str(base_dir))


def _resolve_metadata_path(path_value: str, base_dir: Path) -> str:
    candidate = Path(str(path_value)).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((base_dir / candidate).resolve())


def _is_core_node(data: Dict[str, Any]) -> bool:
    node_type = str(data.get("node_type") or "").strip().lower()
    return node_type == "core"


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return html.unescape(value).strip()
    return html.unescape(str(value)).strip()


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = html.unescape(value).strip()
    if not text:
        return ""
    if not ((text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]"))):
        return text
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    return _normalize_value(parsed)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {_normalize_text(key): _normalize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, str):
        return _maybe_parse_json(value)
    return value


def _normalize_keywords(value: Any) -> Tuple[str, ...]:
    normalized = _normalize_value(value)
    if isinstance(normalized, list):
        items = [_normalize_text(item) for item in normalized if _normalize_text(item)]
        return tuple(items)
    text = _normalize_text(normalized)
    return (text,) if text else ()


def _normalize_matrix(matrix: Any) -> Any:
    np = _load_numpy()
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_mask = norms == 0.0
    if np.any(zero_mask):
        raise ValueError("Embedding matrix contains zero-length rows")
    if np.allclose(norms, 1.0, atol=1e-4):
        return matrix
    return matrix / norms


@dataclass(frozen=True)
class ComponentEmbeddingRecord:
    node_id: str
    component_index: int
    component_name: str
    component_summary: str
    component_keywords: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "component_index": self.component_index,
            "component_name": self.component_name,
            "component_summary": self.component_summary,
            "component_keywords": list(self.component_keywords),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ComponentEmbeddingRecord":
        return cls(
            node_id=str(payload.get("node_id") or ""),
            component_index=int(payload.get("component_index") or 0),
            component_name=_normalize_text(payload.get("component_name")),
            component_summary=_normalize_text(payload.get("component_summary")),
            component_keywords=_normalize_keywords(payload.get("component_keywords")),
        )


class PaperGraphComponentVectorStore:
    """Component-summary index over paper-graph core nodes.

    Each indexed vector corresponds to one component summary, but retrieval results
    are grouped and returned as whole core nodes.
    """

    def __init__(
        self,
        graph_path: Optional[str] = None,
        model_name_or_path: str = _DEFAULT_MODEL_NAME,
        index_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.graph_path = str(Path(graph_path or _default_graph_path()).expanduser().resolve())
        self.model_name_or_path = model_name_or_path
        self.index_dir = (
            Path(index_dir).expanduser().resolve()
            if index_dir
            else _default_index_dir(self.graph_path)
        )
        self.device = device

        self._model: Optional[Any] = None
        self._core_nodes: Dict[str, Dict[str, Any]] = {}
        self._component_records: List[ComponentEmbeddingRecord] = []
        self._embeddings: Optional[Any] = None
        self._graph_mtime: Optional[float] = None

    @property
    def size(self) -> int:
        return len(self._component_records)

    def _resolve_model_source(self) -> str:
        candidate = Path(str(self.model_name_or_path)).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
        return str(self.model_name_or_path)

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        kwargs: Dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        sentence_transformer_cls = _load_sentence_transformer_cls()
        self._model = sentence_transformer_cls(self._resolve_model_source(), **kwargs)
        return self._model

    def _collect_records(self) -> Tuple[Dict[str, Dict[str, Any]], List[ComponentEmbeddingRecord]]:
        graph = nx.read_gexf(self.graph_path)
        core_nodes: Dict[str, Dict[str, Any]] = {}
        records: List[ComponentEmbeddingRecord] = []

        for raw_node_id, raw_data in graph.nodes(data=True):
            if not _is_core_node(raw_data):
                continue
            node_id = str(raw_node_id)
            core_payload = {"node_id": node_id}
            for key, value in raw_data.items():
                core_payload[_normalize_text(key)] = _normalize_value(value)
            core_nodes[node_id] = core_payload

            components = core_payload.get("components")
            if not isinstance(components, list):
                continue

            for component_index, component in enumerate(components):
                if not isinstance(component, dict):
                    continue
                component_name = _normalize_text(component.get("name")) or f"component_{component_index}"
                component_summary = _normalize_text(component.get("summary"))
                if not component_summary:
                    continue
                records.append(
                    ComponentEmbeddingRecord(
                        node_id=node_id,
                        component_index=component_index,
                        component_name=component_name,
                        component_summary=component_summary,
                        component_keywords=_normalize_keywords(component.get("keywords")),
                    )
                )

        return core_nodes, records

    def _require_loaded_index(self) -> None:
        if self._embeddings is not None:
            return
        self.load(allow_stale_graph=True)

    def build(
        self,
        batch_size: int = 64,
        persist: bool = False,
        force_rebuild: bool = False,
    ) -> "PaperGraphComponentVectorStore":
        if self._embeddings is not None and not force_rebuild:
            return self

        self._core_nodes, self._component_records = self._collect_records()
        if not self._component_records:
            raise ValueError(f"No component summaries found in core nodes of {self.graph_path}")

        texts = [record.component_summary for record in self._component_records]
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        np = _load_numpy()
        self._embeddings = _normalize_matrix(np.asarray(embeddings, dtype=np.float32))
        self._graph_mtime = Path(self.graph_path).stat().st_mtime

        if persist:
            self.save()
        return self

    def save(self, index_dir: Optional[str] = None) -> Path:
        if self._embeddings is None:
            raise ValueError("No embeddings available. Call build() or load() first.")

        target_dir = Path(index_dir).expanduser().resolve() if index_dir else self.index_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        np = _load_numpy()
        np.save(target_dir / _EMBEDDING_FILE, self._embeddings)
        model_candidate = Path(str(self.model_name_or_path)).expanduser()
        model_is_local_path = model_candidate.is_absolute() or model_candidate.exists()
        metadata = {
            "graph_path": _serialize_path_for_metadata(self.graph_path, target_dir),
            "graph_mtime": self._graph_mtime,
            "model_name_or_path": (
                _serialize_path_for_metadata(self.model_name_or_path, target_dir)
                if model_is_local_path
                else self.model_name_or_path
            ),
            "model_name_or_path_relative_to_index": model_is_local_path,
            "core_nodes": self._core_nodes,
            "components": [record.to_dict() for record in self._component_records],
        }
        (target_dir / _METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.index_dir = target_dir
        return target_dir

    def load(
        self,
        index_dir: Optional[str] = None,
        allow_stale_graph: bool = True,
    ) -> "PaperGraphComponentVectorStore":
        target_dir = Path(index_dir).expanduser().resolve() if index_dir else self.index_dir
        embedding_path = target_dir / _EMBEDDING_FILE
        metadata_path = target_dir / _METADATA_FILE

        if not embedding_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Component index files are missing under {target_dir}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.graph_path = _resolve_metadata_path(
            str(metadata.get("graph_path") or self.graph_path),
            target_dir,
        )
        model_name_or_path = str(metadata.get("model_name_or_path") or self.model_name_or_path)
        if metadata.get("model_name_or_path_relative_to_index"):
            self.model_name_or_path = _resolve_metadata_path(model_name_or_path, target_dir)
        else:
            model_candidate = Path(model_name_or_path).expanduser()
            self.model_name_or_path = (
                str(model_candidate.resolve()) if model_candidate.is_absolute() else model_name_or_path
            )
        self._graph_mtime = metadata.get("graph_mtime")
        self._core_nodes = {
            str(node_id): _normalize_value(payload)
            for node_id, payload in (metadata.get("core_nodes") or {}).items()
        }
        self._component_records = [
            ComponentEmbeddingRecord.from_dict(item)
            for item in (metadata.get("components") or [])
            if isinstance(item, dict)
        ]
        np = _load_numpy()
        self._embeddings = _normalize_matrix(np.asarray(np.load(embedding_path), dtype=np.float32))
        self.index_dir = target_dir

        if not allow_stale_graph and Path(self.graph_path).exists() and self._graph_mtime is not None:
            current_mtime = Path(self.graph_path).stat().st_mtime
            if abs(float(self._graph_mtime) - current_mtime) > 1e-6:
                raise RuntimeError("Stored component index is stale relative to paper_graph.gexf")

        if self._embeddings.shape[0] != len(self._component_records):
            raise RuntimeError("Component metadata and embedding row count do not match")
        return self

    def get_core_node(self, node_id: str) -> Dict[str, Any]:
        self._require_loaded_index()
        payload = self._core_nodes.get(str(node_id), {"node_id": str(node_id)})
        return copy.deepcopy(payload)

    def ensure_index(
        self,
        batch_size: int = 64,
        persist: bool = True,
        force_rebuild: bool = False,
    ) -> "PaperGraphComponentVectorStore":
        if self._embeddings is not None and not force_rebuild:
            return self
        if not force_rebuild:
            try:
                return self.load(allow_stale_graph=False)
            except (FileNotFoundError, RuntimeError, ValueError):
                pass
        return self.build(batch_size=batch_size, persist=persist, force_rebuild=True)

    def search(
        self,
        query: str,
        top_k: int = 5,
        component_hits_per_core: int = 1,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        text = _normalize_text(query)
        if not text:
            return []
        self.ensure_index()
        model = self._get_model()
        query_embedding = model.encode(
            [text],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return self.search_by_vectors(
            query_embeddings=query_embedding,
            top_k=top_k,
            component_hits_per_core=component_hits_per_core,
        )

    def search_by_vector(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        component_hits_per_core: int = 1,
    ) -> List[Dict[str, Any]]:
        return self.search_by_vectors(
            query_embeddings=[query_embedding],
            top_k=top_k,
            component_hits_per_core=component_hits_per_core,
        )

    def search_component_hits_by_vectors(
        self,
        query_embeddings: Sequence[Sequence[float]],
        top_k: int = 50,
    ) -> List[List[Dict[str, Any]]]:
        if top_k <= 0:
            return []

        self._require_loaded_index()
        if self._embeddings is None:
            raise ValueError("Index is not available")

        np = _load_numpy()
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[0] == 0:
            raise ValueError("query_embeddings must contain at least one vector")

        queries = _normalize_matrix(queries)
        score_matrix = self._embeddings @ queries.T
        max_hits = min(int(top_k), self._embeddings.shape[0])
        all_hits: List[List[Dict[str, Any]]] = []

        for query_index in range(queries.shape[0]):
            component_scores = score_matrix[:, query_index]
            ranked_indices = np.argsort(-component_scores)[:max_hits]
            hits: List[Dict[str, Any]] = []
            for raw_index in ranked_indices.tolist():
                index = int(raw_index)
                record = self._component_records[index]
                hits.append(
                    {
                        "query_index": query_index,
                        "node_id": record.node_id,
                        "score": float(component_scores[index]),
                        "component_index": record.component_index,
                        "component_name": record.component_name,
                        "component_summary": record.component_summary,
                        "component_keywords": list(record.component_keywords),
                    }
                )
            all_hits.append(hits)
        return all_hits

    def search_by_vectors(
        self,
        query_embeddings: Sequence[Sequence[float]],
        top_k: int = 5,
        component_hits_per_core: int = 1,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        self.ensure_index()
        if self._embeddings is None:
            raise ValueError("Index is not available")

        np = _load_numpy()
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[0] == 0:
            raise ValueError("query_embeddings must contain at least one vector")

        queries = _normalize_matrix(queries)
        component_scores = np.max(self._embeddings @ queries.T, axis=1)
        ranked_indices = np.argsort(-component_scores)

        grouped: Dict[str, Dict[str, Any]] = {}
        component_cap = max(0, int(component_hits_per_core))

        for raw_index in ranked_indices.tolist():
            index = int(raw_index)
            record = self._component_records[index]
            score = float(component_scores[index])

            entry = grouped.get(record.node_id)
            if entry is None:
                entry = {
                    "node_id": record.node_id,
                    "score": score,
                    "core_node": copy.deepcopy(self._core_nodes.get(record.node_id, {"node_id": record.node_id})),
                    "matched_components": [],
                }
                grouped[record.node_id] = entry

            if component_cap and len(entry["matched_components"]) < component_cap:
                entry["matched_components"].append(
                    {
                        "component_index": record.component_index,
                        "component_name": record.component_name,
                        "component_summary": record.component_summary,
                        "component_keywords": list(record.component_keywords),
                        "score": score,
                    }
                )

        results = sorted(
            grouped.values(),
            key=lambda item: (
                -float(item["score"]),
                _normalize_text(
                    item["core_node"].get("paper_title")
                    or item["core_node"].get("full_name")
                    or item["node_id"]
                ),
            ),
        )
        return results[:top_k]
