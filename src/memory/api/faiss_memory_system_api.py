from typing import Dict, List, Optional, Tuple, Union

from src.memory.api.base_vector_memory_system_api import (
    BaseVectorMemorySystem,
    EpisodicRecordPayload,
    ProceduralRecordPayload,
    SemanticRecordPayload,
    VectorMemorySystemConfig,
)
from src.memory.memory_system import (
    EpisodicRecord,
    FaissVectorStore,
    ProceduralRecord,
    SemanticRecord,
)
from src.memory.memory_system.utils import _transfer_dict_to_semantic_text, new_id, now_iso


class FAISSMemorySystem(BaseVectorMemorySystem):
    def __init__(self, embedding_model=None, **kwargs):
        cfg = VectorMemorySystemConfig(**kwargs)

        if FaissVectorStore is None:
            raise ModuleNotFoundError(
                "Vector-memory dependencies are missing. Install `faiss`, "
                "`sentence_transformers`, and `rank_bm25` to enable the FAISS backend."
            )

        self.memory_type = cfg.memory_type
        self.vector_store = FaissVectorStore(
            cfg.model_path,
            self.memory_type,
            model=embedding_model,
        )

    def instantiate_sem_record(self, **kwargs) -> SemanticRecord:
        payload = SemanticRecordPayload(**kwargs)
        record = SemanticRecord(
            id=new_id("sem"),
            summary=payload.summary,
            detail=payload.detail,
            tags=payload.tags,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        return record

    def instantiate_epi_record(self, **kwargs) -> EpisodicRecord:
        payload = EpisodicRecordPayload(**kwargs)
        record = EpisodicRecord(
            id=new_id("epi"),
            stage=payload.stage,
            summary=payload.summary,
            detail=payload.detail,
            tags=payload.tags,
            created_at=now_iso(),
        )
        record.embedding = self.vector_store._embed(_transfer_dict_to_semantic_text(record.detail))
        return record

    def instantiate_proc_record(self, **kwargs) -> ProceduralRecord:
        payload = ProceduralRecordPayload(**kwargs)
        record = ProceduralRecord(
            id=new_id("proc"),
            name=payload.name,
            description=payload.description,
            steps=payload.steps,
            code=payload.code,
            tags=payload.tags,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        return record

    @property
    def size(self) -> int:
        return self.vector_store._get_record_nums()

    def get_records_by_ids(
        self,
        mids: List[str],
    ) -> Union[List[SemanticRecord], List[EpisodicRecord], List[ProceduralRecord]]:
        reverse_map = {mid: fid for fid, mid in self.vector_store.fidmap2mid.items()}
        records = []
        for mid in mids:
            fid = reverse_map.get(mid, None)
            try:
                record = self.vector_store.meta[fid]
            except KeyError as e:
                print(f"Record with id {mid} not found: {e}")
                continue
            records.append(record)
        return records

    def get_last_k_records(
        self,
        k: int,
    ) -> Tuple[Union[List[SemanticRecord], List[EpisodicRecord], List[ProceduralRecord]], int]:
        if k >= self.size:
            return ([record.to_dict() for record in self.vector_store.meta.values()], self.size)

        sorted_fids = sorted(self.vector_store.fidmap2mid.keys(), reverse=True)
        return ([self.vector_store.meta[fid].to_dict() for fid in sorted_fids[:k]], k)

    def is_exists(self, mids: List[str]) -> List[bool]:
        reverse_map = {mid: fid for fid, mid in self.vector_store.fidmap2mid.items()}
        results = []
        for mid in mids:
            fid = reverse_map.get(mid, None)
            if fid is not None and fid in self.vector_store.meta:
                results.append(True)
            else:
                results.append(False)
        return results

    def add(self, memories: List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]] = None, agent_id: str = "") -> bool:
        try:
            self.vector_store.add(memories, agent_id=agent_id)
            return True
        except Exception as e:
            print(f"Error adding memories: {e}")
            return False

    def update(self, memories: List[Union[SemanticRecord, ProceduralRecord]] = None) -> bool:
        try:
            self.vector_store.update(memories)
            return True
        except Exception as e:
            print(f"Error updating memories: {e}")
            return False

    def delete(self, mids: List[str]) -> bool:
        try:
            self.vector_store.delete(mids)
            return True
        except Exception as e:
            print(f"Error deleting memories: {e}")
            return False

    def upsert_normal_records(self, records: List[Union[SemanticRecord, ProceduralRecord]], agent_id: str = "") -> None:
        for record in records:
            result = self.get_nearest_k_records(record, k=1, threshold=0.8, agent_id=agent_id)
            if result is not None and len(result) > 0:
                nearest_record = result[0][1]
                if isinstance(record, SemanticRecord):
                    nearest_record.update(
                        summary=record.summary,
                        detail=record.detail,
                        tags=record.tags,
                    )
                elif isinstance(record, ProceduralRecord):
                    nearest_record.update(
                        name=record.name,
                        description=record.description,
                        steps=record.steps,
                        code=record.code,
                        tags=record.tags,
                    )
            else:
                self.add([record], agent_id=agent_id)

    def query(
        self,
        query_text: str,
        method: str = "embedding",
        limit: int = 5,
        filters: Optional[Dict] = None,
        threshold: float = 0.0,
        agent_id: str = "",
    ) -> List[Tuple[float, Union[SemanticRecord, EpisodicRecord, ProceduralRecord]]]:
        limit = min(limit, self.size)
        try:
            results = self.vector_store.query(
                query_text,
                method=method,
                limit=limit,
                filters=filters,
                threshold=threshold,
                agent_id=agent_id,
            )
        except Exception as e:
            print(f"Error querying memories: {e}")
            results = []
        return results

    def get_nearest_k_records(
        self,
        record: Union[SemanticRecord, EpisodicRecord, ProceduralRecord],
        method: str = "embedding",
        k: int = 5,
        filters: Optional[Dict] = None,
        threshold: float = 0.0,
        agent_id: str = "",
    ) -> List[Tuple[float, Union[SemanticRecord, EpisodicRecord, ProceduralRecord]]]:
        if isinstance(record, SemanticRecord):
            query_text = record.detail
        elif isinstance(record, EpisodicRecord):
            query_text = record.summary
        elif isinstance(record, ProceduralRecord):
            query_text = record.description

        try:
            results = self.vector_store.query(
                query_text,
                method=method,
                limit=k,
                filters=filters,
                threshold=threshold,
                agent_id=agent_id,
            )
        except Exception as e:
            print(f"Error querying nearest records: {e}")
            results = []
        return results

    def save(self, path: str) -> bool:
        try:
            self.vector_store.save(path)
            return True
        except Exception:
            return False

    def load(self, path: str) -> bool:
        try:
            self.vector_store.load(path)
            return True
        except Exception:
            return False
