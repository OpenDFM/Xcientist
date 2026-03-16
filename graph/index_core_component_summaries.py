from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_DB_PATH = BASE_DIR / "graph.db"
DEFAULT_STORE_PATH = BASE_DIR / "core_component_summary_vector_store"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "bge-m3"
DEFAULT_FETCH_SIZE = 16
DEFAULT_BATCH_SIZE = 8


def now_iso() -> str:
    return datetime.now().isoformat()


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("numpy is required to build the component vector store") from exc
    return np


def _load_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("faiss is required to build the component vector store") from exc
    return faiss


def _load_sentence_transformer_cls():
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence_transformers is required to build the component vector store"
        ) from exc
    return SentenceTransformer


def _load_tqdm():
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return None
    return tqdm


@dataclass
class ComponentVectorRecord:
    id: str
    node_id: str
    node_label: str
    paper_title: str
    component_name: str
    component_summary: str
    component_keywords: tuple[str, ...]


    def with_timestamps(self, timestamp: Optional[str] = None) -> "ComponentVectorRecord":
        return ComponentVectorRecord(
            id=self.id,
            node_id=self.node_id,
            node_label=self.node_label,
            paper_title=self.paper_title,
            component_name=self.component_name,
            component_summary=self.component_summary,
            component_keywords=tuple(self.component_keywords),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "node_label": self.node_label,
            "paper_title": self.paper_title,
            "component_name": self.component_name,
            "component_summary": self.component_summary,
            "component_keywords": list(self.component_keywords),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ComponentVectorRecord":
        return cls(
            id=_normalize_text(payload.get("id")),
            node_id=_normalize_text(payload.get("node_id")),
            node_label=_normalize_text(payload.get("node_label")),
            paper_title=_normalize_text(payload.get("paper_title")),
            component_name=_normalize_text(payload.get("component_name")),
            component_summary=_normalize_text(payload.get("component_summary")),
            component_keywords=_normalize_keywords(payload.get("component_keywords")),
        )


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
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _normalize_keywords(value: Any) -> tuple[str, ...]:
    parsed = _maybe_parse_json(value)
    if isinstance(parsed, list):
        keywords = [_normalize_text(item) for item in parsed if _normalize_text(item)]
        return tuple(keywords)
    text = _normalize_text(parsed)
    return (text,) if text else ()


def _stable_component_record_id(node_id: str, component_name: str, component_summary: str) -> str:
    normalized_key = "\n".join(
        [
            _normalize_text(node_id).lower(),
            _normalize_text(component_name).lower(),
            _normalize_text(component_summary).lower(),
        ]
    )
    digest = hashlib.sha1(normalized_key.encode("utf-8")).hexdigest()[:20]
    return f"sem_graph_component_{digest}"


def _store_files_exist(store_path: Path) -> bool:
    return (store_path / "faiss.index").exists() and (store_path / "meta.json").exists()


class ComponentSummaryFaissStore:
    def __init__(self, model_name_or_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_name_or_path = str(model_name_or_path).strip() or DEFAULT_MODEL_PATH
        self.index = None
        self.dim: Optional[int] = None
        self.meta: dict[int, ComponentVectorRecord] = {}
        self.fidmap2mid: dict[int, str] = {}
        self._next_id = 0
        self._model = None

    @property
    def size(self) -> int:
        return len(self.meta)

    @property
    def existing_record_ids(self) -> set[str]:
        return {str(mid) for mid in self.fidmap2mid.values()}

    def _get_model(self):
        if self._model is None:
            sentence_transformer_cls = _load_sentence_transformer_cls()
            self._model = sentence_transformer_cls(self.model_name_or_path)
        return self._model

    def _embed(self, texts: list[str]):
        np = _load_numpy()
        model = self._get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embedding output must be a 2D matrix")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("Embedding matrix contains zero-length rows")
        if not np.allclose(norms, 1.0, atol=1e-4):
            matrix = matrix / norms
        return matrix

    def _ensure_index(self, dim: int) -> None:
        if self.index is not None:
            return
        faiss = _load_faiss()
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.dim = dim

    def add(self, records: list[ComponentVectorRecord]) -> int:
        if not records:
            return 0

        np = _load_numpy()
        vectors = self._embed([record.component_summary for record in records])
        self._ensure_index(vectors.shape[1])

        ids = np.arange(self._next_id, self._next_id + len(records), dtype="int64")
        self.index.add_with_ids(vectors, ids)

        for fid, record in zip(ids.tolist(), records):
            self.meta[int(fid)] = record
            self.fidmap2mid[int(fid)] = record.id
        self._next_id += len(records)
        return len(records)

    def save(self, store_path: Path) -> None:
        if self.index is None:
            raise ValueError("No vectors available to save")

        faiss = _load_faiss()
        target = Path(store_path).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(target / "faiss.index"))
        payload = {
            "model_name_or_path": self.model_name_or_path,
            "next_id": self._next_id,
            "fidmap2mid": {str(fid): mid for fid, mid in self.fidmap2mid.items()},
            "meta": {str(fid): record.to_dict() for fid, record in self.meta.items()},
        }
        with (target / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load(self, store_path: Path) -> None:
        faiss = _load_faiss()
        target = Path(store_path).expanduser().resolve()
        self.index = faiss.read_index(str(target / "faiss.index"))
        self.dim = int(self.index.d)

        with (target / "meta.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        loaded_model_name = str(payload.get("model_name_or_path") or "").strip()
        if loaded_model_name and loaded_model_name != self.model_name_or_path:
            logging.info(
                "Existing vector store uses model %s; reusing it instead of requested model %s",
                loaded_model_name,
                self.model_name_or_path,
            )
        self.model_name_or_path = loaded_model_name or self.model_name_or_path
        self._next_id = int(payload.get("next_id", self.index.ntotal))
        self.fidmap2mid = {
            int(fid): str(mid)
            for fid, mid in dict(payload.get("fidmap2mid") or {}).items()
        }
        self.meta = {
            int(fid): ComponentVectorRecord.from_dict(record_payload)
            for fid, record_payload in dict(payload.get("meta") or {}).items()
        }

class GraphCoreComponentIndexer:
    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        store_path: Path = DEFAULT_STORE_PATH,
        model_path: str = DEFAULT_MODEL_PATH,
        fetch_size: int = DEFAULT_FETCH_SIZE,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.store_path = Path(store_path).expanduser().resolve()
        self.model_path = str(model_path).strip() or DEFAULT_MODEL_PATH
        self.fetch_size = max(int(fetch_size), 1)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _iter_core_rows(self, limit: Optional[int] = None) -> Iterator[sqlite3.Row]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            if limit is None:
                cursor.execute(
                    """
                    select id, raw_json
                    from nodes
                    where node_type = 'Core'
                    order by id
                    """
                )
            else:
                cursor.execute(
                    """
                    select id, raw_json
                    from nodes
                    where node_type = 'Core'
                    order by id
                    limit ?
                    """,
                    (int(limit),),
                )
            while True:
                rows = cursor.fetchmany(self.fetch_size)
                if not rows:
                    break
                for row in rows:
                    yield row
        finally:
            conn.close()

    def _count_core_rows(self, limit: Optional[int] = None) -> int:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            total = int(cursor.execute("select count(*) from nodes where node_type = 'Core'").fetchone()[0])
        finally:
            conn.close()
        if limit is None:
            return total
        return min(total, max(int(limit), 0))

    def _extract_components_from_row(
        self,
        row: sqlite3.Row,
    ) -> tuple[list[ComponentVectorRecord], int]:
        node_id = _normalize_text(row["id"])
        raw_json = _normalize_text(row["raw_json"])
        if not raw_json:
            return [], 0

        payload = _maybe_parse_json(raw_json)
        if not isinstance(payload, dict):
            logging.warning("Skip core node %s because raw_json is not an object", node_id)
            return [], 0

        node_label = _normalize_text(payload.get("label")) or node_id
        paper_title = _normalize_text(payload.get("paper_title"))
        raw_components = _maybe_parse_json(payload.get("components"))
        if not isinstance(raw_components, list):
            return [], 0

        extracted: list[ComponentVectorRecord] = []
        skipped_empty = 0
        for component_index, component in enumerate(raw_components):
            if not isinstance(component, dict):
                continue
            component_name = _normalize_text(component.get("name")) or f"component_{component_index}"
            component_summary = _normalize_text(component.get("summary"))
            if not component_summary:
                skipped_empty += 1
                continue
            extracted.append(
                ComponentVectorRecord(
                    id=_stable_component_record_id(node_id, component_name, component_summary),
                    node_id=node_id,
                    node_label=node_label,
                    paper_title=paper_title,
                    component_name=component_name,
                    component_summary=component_summary,
                    component_keywords=_normalize_keywords(component.get("keywords")),
                )
            )
        return extracted, skipped_empty

    def _load_existing_store(self, rebuild: bool) -> tuple[ComponentSummaryFaissStore, set[str]]:
        store = ComponentSummaryFaissStore(model_name_or_path=self.model_path)
        if rebuild or not _store_files_exist(self.store_path):
            return store, set()

        store.load(self.store_path)
        existing_ids = store.existing_record_ids
        logging.info(
            "Loaded existing vector store from %s with %d records",
            self.store_path,
            len(existing_ids),
        )
        return store, existing_ids

    def build(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        limit: Optional[int] = None,
        rebuild: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_path}")

        batch_size = max(int(batch_size), 1)
        total_core_nodes = self._count_core_rows(limit=limit)
        store, existing_ids = self._load_existing_store(rebuild=rebuild)
        seen_ids = set(existing_ids)
        pending_records: list[ComponentVectorRecord] = []

        stats = {
            "db_path": str(self.db_path),
            "store_path": str(self.store_path),
            "model_path": self.model_path,
            "rebuild": bool(rebuild),
            "total_core_nodes": total_core_nodes,
            "core_nodes_scanned": 0,
            "component_summaries_seen": 0,
            "component_summaries_added": 0,
            "duplicates_skipped": 0,
            "empty_summaries_skipped": 0,
        }

        tqdm_cls = _load_tqdm()
        progress = (
            tqdm_cls(
                total=total_core_nodes,
                desc="Parsing Core nodes",
                unit="node",
                dynamic_ncols=True,
            )
            if show_progress and tqdm_cls is not None
            else None
        )

        try:
            for row in self._iter_core_rows(limit=limit):
                stats["core_nodes_scanned"] += 1
                components, skipped_empty = self._extract_components_from_row(row)
                stats["empty_summaries_skipped"] += skipped_empty

                for component in components:
                    stats["component_summaries_seen"] += 1
                    if component.id in seen_ids:
                        stats["duplicates_skipped"] += 1
                        continue

                    pending_records.append(component.with_timestamps())
                    seen_ids.add(component.id)

                    if len(pending_records) >= batch_size:
                        added_count = store.add(pending_records)
                        stats["component_summaries_added"] += added_count
                        logging.info(
                            "Indexed %d new component summaries after scanning %d core nodes",
                            stats["component_summaries_added"],
                            stats["core_nodes_scanned"],
                        )
                        pending_records.clear()

                if progress is not None:
                    progress.update(1)
                    if stats["core_nodes_scanned"] % 100 == 0 or stats["core_nodes_scanned"] == total_core_nodes:
                        progress.set_postfix(
                            added=stats["component_summaries_added"],
                            dup=stats["duplicates_skipped"],
                            empty=stats["empty_summaries_skipped"],
                            refresh=False,
                        )
        finally:
            if progress is not None:
                progress.close()

        if pending_records:
            added_count = store.add(pending_records)
            stats["component_summaries_added"] += added_count

        if store.size == 0:
            raise ValueError(f"No Core component summaries were extracted from {self.db_path}")

        if stats["component_summaries_added"] > 0 or rebuild or not _store_files_exist(self.store_path):
            store.save(self.store_path)

        stats["vector_store_size"] = store.size
        self._write_build_stats(stats)
        return stats

    def _write_build_stats(self, stats: dict[str, Any]) -> None:
        payload = dict(stats)
        payload["built_at"] = now_iso()
        target = self.store_path / "build_stats.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_component_summary_vector_store(
    db_path: Path = DEFAULT_DB_PATH,
    store_path: Path = DEFAULT_STORE_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: Optional[int] = None,
    rebuild: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    indexer = GraphCoreComponentIndexer(
        db_path=Path(db_path),
        store_path=Path(store_path),
        model_path=model_path,
    )
    return indexer.build(
        batch_size=batch_size,
        limit=limit,
        rebuild=rebuild,
        show_progress=show_progress,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incrementally index every Core node's component.summary from graph.db into a FAISS vector store."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="Path to graph.db")
    parser.add_argument(
        "--store-path",
        default=str(DEFAULT_STORE_PATH),
        help="Directory used to persist faiss.index, meta.json, and build_stats.json",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="SentenceTransformer model path or model name used for embeddings",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--fetch-size", type=int, default=DEFAULT_FETCH_SIZE, help="SQLite fetch-many size")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on scanned Core nodes")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore any existing vector store and rebuild from scratch",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the parsing progress bar",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _build_arg_parser().parse_args(argv)

    indexer = GraphCoreComponentIndexer(
        db_path=Path(args.db_path),
        store_path=Path(args.store_path),
        model_path=args.model_path,
        fetch_size=args.fetch_size,
    )
    stats = indexer.build(
        batch_size=args.batch_size,
        limit=args.limit,
        rebuild=args.rebuild,
        show_progress=not args.no_progress,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
