"""Vector-store utilities over the prebuilt Core-component FAISS index."""

from __future__ import annotations

import copy
import html
import importlib.util
import json
import faiss
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agents.idea_agent.utils.core.json_utils import read_json_file

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_MODEL_NAME = "bge-m3"
_FAISS_FILE = "faiss.index"
_METADATA_FILE = "meta.json"
_GRAPH_SERVER_MODULE: Optional[Any] = None



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
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    zero_mask = norms == 0.0
    if np.any(zero_mask):
        raise ValueError("Embedding matrix contains zero-length rows")
    if np.allclose(norms, 1.0, atol=1e-4):
        return matrix
    return matrix / norms


def _resolve_repo_path(path_value: Any) -> str:
    raw = _normalize_text(path_value)
    if not raw:
        return raw

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve()) if candidate.exists() else raw

    search_roots = (
        _REPO_ROOT,
        _REPO_ROOT / "models",
    )
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return raw


def _load_graph_server_module() -> Any:
    global _GRAPH_SERVER_MODULE
    if _GRAPH_SERVER_MODULE is not None:
        return _GRAPH_SERVER_MODULE

    server_path = (_REPO_ROOT / "graph" / "server.py").resolve()
    spec = importlib.util.spec_from_file_location("researchagent_graph_server", server_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load graph server module from {server_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _GRAPH_SERVER_MODULE = module
    return module


def _default_db_path() -> Path:
    graph_server = _load_graph_server_module()
    return Path(graph_server.DB_PATH).expanduser().resolve()


def _default_index_dir() -> Path:
    graph_server = _load_graph_server_module()
    base_dir = Path(graph_server.BASE_DIR).expanduser().resolve()
    return (base_dir / "core_component_summary_vector_store").resolve()


@dataclass(frozen=True)
class ComponentEmbeddingRecord:
    record_id: str
    node_id: str
    component_index: int
    component_name: str
    component_summary: str
    component_keywords: Tuple[str, ...]
    node_label: str = ""
    paper_title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.record_id,
            "node_id": self.node_id,
            "component_index": self.component_index,
            "component_name": self.component_name,
            "component_summary": self.component_summary,
            "component_keywords": list(self.component_keywords),
            "node_label": self.node_label,
            "paper_title": self.paper_title,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        *,
        component_index: int = 0,
    ) -> "ComponentEmbeddingRecord":
        return cls(
            record_id=_normalize_text(payload.get("id")),
            node_id=_normalize_text(payload.get("node_id")),
            component_index=int(payload.get("component_index", component_index) or component_index),
            component_name=_normalize_text(payload.get("component_name")),
            component_summary=_normalize_text(payload.get("component_summary")),
            component_keywords=_normalize_keywords(payload.get("component_keywords")),
            node_label=_normalize_text(payload.get("node_label")),
            paper_title=_normalize_text(payload.get("paper_title")),
        )


class PaperGraphComponentVectorStore:
    """Read-only wrapper over the prebuilt Core-component FAISS vector store."""

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL_NAME,
        index_dir: Optional[str] = None,
        device: Optional[str] = None,
        db_path: Optional[str] = None,
        model: Optional[Any] = None,
        disabled_error: Optional[str] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.index_dir = (
            Path(index_dir).expanduser().resolve()
            if index_dir
            else _default_index_dir()
        )
        server_db_path = _default_db_path()
        if db_path is not None:
            candidate = Path(db_path).expanduser().resolve()
            if candidate != server_db_path:
                raise ValueError(
                    "PaperGraphComponentVectorStore only supports DB access through graph/server.py. "
                    f"Expected db_path={server_db_path}, got {candidate}."
                )
            self.db_path = candidate
        else:
            self.db_path = server_db_path
        self.device = device

        self._model: Optional[Any] = model
        self._index: Optional[Any] = None
        self._core_nodes: Dict[str, Dict[str, Any]] = {}
        self._core_node_fallbacks: Dict[str, Dict[str, Any]] = {}
        self._component_records: Dict[int, ComponentEmbeddingRecord] = {}
        self._disabled_error: Optional[str] = str(disabled_error).strip() if disabled_error else None

    @property
    def size(self) -> int:
        return len(self._component_records)

    def _resolve_model_source(self) -> str:
        return _resolve_repo_path(self.model_name_or_path)

    def _ensure_available(self) -> None:
        if self._disabled_error:
            raise RuntimeError(self._disabled_error)

    def _get_model(self) -> Any:
        self._ensure_available()
        if self._model is not None:
            return self._model
        kwargs: Dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self._resolve_model_source(), **kwargs)
        return self._model

    def get_model(self) -> Any:
        return self._get_model()

    def warmup(
        self,
        index_dir: Optional[str] = None,
        allow_stale_graph: bool = True,
    ) -> "PaperGraphComponentVectorStore":
        self.load(index_dir=index_dir, allow_stale_graph=allow_stale_graph)
        self._get_model()
        return self

    def _require_loaded_index(self) -> None:
        self._ensure_available()
        if self._index is not None and self._component_records:
            return
        self.load()

    def _load_core_node_from_graph_server(self, node_id: str) -> Dict[str, Any]:
        graph_server = _load_graph_server_module()
        payload = graph_server.get_node(str(node_id))
        if not isinstance(payload, dict) or not payload.get("found"):
            return {}
        node = payload.get("node")
        if not isinstance(node, dict):
            return {}
        normalized = {
            _normalize_text(key): _normalize_value(value)
            for key, value in node.items()
        }
        normalized.setdefault("node_id", str(node_id))
        normalized.setdefault("label", _normalize_text(node.get("label")) or str(node_id))
        return normalized

    def _load_core_node_from_db(self, node_id: str) -> Dict[str, Any]:
        return self._load_core_node_from_graph_server(node_id)

    def load(
        self,
        index_dir: Optional[str] = None,
        allow_stale_graph: bool = True,
    ) -> "PaperGraphComponentVectorStore":
        self._ensure_available()
        del allow_stale_graph
        target_dir = Path(index_dir).expanduser().resolve() if index_dir else self.index_dir
        faiss_path = target_dir / _FAISS_FILE
        metadata_path = target_dir / _METADATA_FILE

        if not faiss_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Component index files are missing under {target_dir}")

        metadata = read_json_file(metadata_path)

        self._index = faiss.read_index(str(faiss_path))
        self.index_dir = target_dir
        stored_model = _normalize_text(metadata.get("model_name_or_path"))
        if stored_model:
            self.model_name_or_path = _resolve_repo_path(stored_model)

        raw_meta = metadata.get("meta") or {}
        if not isinstance(raw_meta, dict):
            raise RuntimeError(f"Invalid component metadata under {metadata_path}")

        self._component_records = {}
        self._core_node_fallbacks = {}
        component_counts: Dict[str, int] = {}

        for fid in sorted(int(key) for key in raw_meta.keys()):
            payload = raw_meta.get(str(fid))
            if not isinstance(payload, dict):
                continue
            node_id = _normalize_text(payload.get("node_id"))
            component_index = component_counts.get(node_id, 0)
            component_counts[node_id] = component_index + 1
            record = ComponentEmbeddingRecord.from_dict(payload, component_index=component_index)
            self._component_records[fid] = record
            self._core_node_fallbacks.setdefault(
                record.node_id,
                {
                    "node_id": record.node_id,
                    "label": record.node_label or record.node_id,
                    "full_name": record.node_label or record.node_id,
                    "paper_title": record.paper_title,
                },
            )

        return self

    def get_core_node(self, node_id: str) -> Dict[str, Any]:
        key = str(node_id)
        if key not in self._core_nodes:
            payload = self._load_core_node_from_db(key)
            if not payload:
                payload = copy.deepcopy(self._core_node_fallbacks.get(key, {"node_id": key}))
            self._core_nodes[key] = payload
        return copy.deepcopy(self._core_nodes[key])

    def ensure_index(
        self,
        batch_size: int = 64,
        persist: bool = True,
        force_rebuild: bool = False,
    ) -> "PaperGraphComponentVectorStore":
        del batch_size, persist
        if self._index is not None and self._component_records and not force_rebuild:
            return self
        if force_rebuild:
            raise NotImplementedError(
                "force_rebuild is unsupported here. Rebuild the index externally, then reload."
            )
        return self.load()

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

    def _prepare_query_embeddings(self, query_embeddings: Sequence[Sequence[float]]) -> Any:
        queries = np.asarray(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[0] == 0:
            raise ValueError("query_embeddings must contain at least one vector")
        return _normalize_matrix(queries)

    def search_component_hits_by_vectors(
        self,
        query_embeddings: Sequence[Sequence[float]],
        top_k: int = 50,
    ) -> List[List[Dict[str, Any]]]:
        if top_k <= 0:
            return []

        self._require_loaded_index()
        if self._index is None:
            raise ValueError("Index is not available")

        queries = self._prepare_query_embeddings(query_embeddings)
        max_hits = min(int(top_k), int(self._index.ntotal))
        scores, ids = self._index.search(queries, max_hits)

        all_hits: List[List[Dict[str, Any]]] = []
        for query_index in range(scores.shape[0]):
            hits: List[Dict[str, Any]] = []
            for score, raw_id in zip(scores[query_index], ids[query_index]):
                fid = int(raw_id)
                if fid == -1:
                    continue
                record = self._component_records.get(fid)
                if record is None:
                    continue
                hits.append(
                    {
                        "query_index": query_index,
                        "node_id": record.node_id,
                        "score": float(score),
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
        if self._index is None:
            raise ValueError("Index is not available")

        component_cap = max(0, int(component_hits_per_core))
        per_query_hits = max(int(top_k) * max(component_cap, 1) * 8, int(top_k))
        all_component_hits = self.search_component_hits_by_vectors(
            query_embeddings=query_embeddings,
            top_k=per_query_hits,
        )

        best_by_component: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for hits in all_component_hits:
            for hit in hits:
                key = (
                    str(hit.get("node_id") or ""),
                    int(hit.get("component_index") or 0),
                )
                existing = best_by_component.get(key)
                if existing is None or float(hit.get("score") or 0.0) > float(existing.get("score") or 0.0):
                    best_by_component[key] = hit

        grouped: Dict[str, Dict[str, Any]] = {}
        ranked_hits = sorted(
            best_by_component.values(),
            key=lambda item: (
                -float(item.get("score") or 0.0),
                _normalize_text(item.get("component_name") or ""),
                _normalize_text(item.get("node_id") or ""),
            ),
        )

        for hit in ranked_hits:
            node_id = str(hit.get("node_id") or "")
            if not node_id:
                continue
            score = float(hit.get("score") or 0.0)
            entry = grouped.get(node_id)
            if entry is None:
                core_node = self.get_core_node(node_id)
                entry = {
                    "node_id": node_id,
                    "score": score,
                    "core_node": core_node,
                    "matched_components": [],
                }
                grouped[node_id] = entry

            if component_cap and len(entry["matched_components"]) < component_cap:
                entry["matched_components"].append(
                    {
                        "component_index": int(hit.get("component_index") or 0),
                        "component_name": hit.get("component_name"),
                        "component_summary": hit.get("component_summary"),
                        "component_keywords": list(hit.get("component_keywords") or []),
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
                    or item["core_node"].get("label")
                    or item["node_id"]
                ),
            ),
        )
        return results[:top_k]
