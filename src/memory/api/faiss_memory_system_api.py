import os
import shutil
import sys

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, validate_call
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
from memory_system import (
    FaissVectorStore,
    SemanticRecord,
    EpisodicRecord,
    ProceduralRecord,
    OpenAIClient,
    ABSTRACT_EPISODIC_TO_SEMANTIC_PROMPT,
)
from memory_system.utils import now_iso, new_id, _transfer_dict_to_semantic_text
from memory_system.denstream import DenStream
from .base_memory_system_api import MemorySystem, MemorySystemConfig, MemoryRecordPayload
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class FAISSMemorySystem(MemorySystem):
    def __init__(self, **kwargs):
        cfg = MemorySystemConfig(**kwargs)

        self.memory_type = cfg.memory_type
        self.vector_store = FaissVectorStore(cfg.model_path, self.memory_type)
        self.llm = OpenAIClient(model=cfg.llm_name)

    def instantiate_sem_record(self, **kwargs) -> SemanticRecord:
        cfg = MemoryRecordPayload(**kwargs)
        record = SemanticRecord(
            id=new_id("sem"),
            summary=cfg.summary,
            detail=cfg.detail,
            tags=cfg.tags,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        return record
    
    def instantiate_epi_record(self, eps: float = 0.6, beta: float = 0.5, mu: float = 4, **kwargs) -> EpisodicRecord:
        cfg = MemoryRecordPayload(**kwargs)
        record = EpisodicRecord(
            id=new_id("epi"),
            stage=cfg.stage,
            summary=cfg.summary,
            detail=cfg.detail,
            tags=cfg.tags,
            created_at=now_iso(),
        )
        record.embedding = self.vector_store._embed(_transfer_dict_to_semantic_text(record.detail))
        self.cluster_machine = DenStream(eps=eps, beta=beta, mu=mu)        
        return record

    def instantiate_proc_record(self, **kwargs) -> ProceduralRecord:
        cfg = MemoryRecordPayload(**kwargs)
        record = ProceduralRecord(
            id=new_id("proc"),
            name=cfg.name,
            description=cfg.description,
            steps=cfg.steps,
            code=cfg.code,
            tags=cfg.tags,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        return record

    def size(self) -> int:
        return self.vector_store._get_record_nums()

    def get_records_by_ids(self, mids: List[str]) -> Union[List[SemanticRecord], List[EpisodicRecord], List[ProceduralRecord]]:
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
    
    def get_last_k_records(self, k: int) -> Tuple[Union[List[SemanticRecord], List[EpisodicRecord], List[ProceduralRecord]], int]:
        if k >= self.size():
            return [record for record in self.vector_store.meta.values()], self.size()
        
        else:
            sorted_fids = sorted(self.vector_store.fidmap2mid.keys(), reverse=True)
            return [self.vector_store.meta[fid] for fid in sorted_fids[:real_k]], k
        
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
        
    def add(self, memories: List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]] = None) -> bool:
        try:
            self.vector_store.add(memories) # Add new memory to FAISS vectorstore.
            return True
        except Exception as e:
            print(f"Error adding memories: {e}")
            return False
    
    def update(self, memories: List[Union[SemanticRecord, ProceduralRecord]] = None) -> bool:
        try:
            self.vector_store.update(memories) # Update new memory to FAISS vectorstore.
            return True
        except Exception as e:
            print(f"Error updating memories: {e}")
            return False

    def batch_memory_process(self, memories: List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]] = None) -> bool:
        '''If you can not distinguish memories need to be add or update, use this method.'''
        try:
            self.vector_store.batch_memory_process(memories)
            return True
        except Exception as e:
            print(f"Error processing memories: {e}")
            return False
    
    def delete(self, mids: List[str]) -> bool:
        try:
            self.vector_store.delete(mids)
            if self.memory_type == "episodic":
                self.cluster_machine = DenStream() # Reset clustering machine
            return True
        except Exception as e:
            print(f"Error deleting memories: {e}")
            return False
    
    def query(self, query_text: str, method: str = "embedding", limit: int = 5, filters: Dict | None = None) -> List[Tuple[float, Dict]]:
        try:
            results = self.vector_store.query(query_text, method=method, limit=limit, filters=filters)
        except Exception as e:
            print(f"Error querying memories: {e}")
            results = []
        return results

    async def abstract_episodic_records(
            self, 
            epi_records: List[EpisodicRecord], 
            consistency_threshold: float = 0.8) -> Tuple[List[SemanticRecord], Dict[int, SemanticRecord]]:
        assert self.memory_type == "episodic", "Clustering is only supported for episodic memory type."

        cidmap2mid: Dict[int, List] = defaultdict(list) # {cluster_id: episodic_record_id}
        midmap2record: Dict[str, EpisodicRecord] = {} # {episodic_record_id: EpisodicRecord}
        cidmap2semrec: Dict[int, SemanticRecord] = {} # {cluster_id: SemanticRecord}

        abstract_result: List[SemanticRecord] = []
        updated_cluster_id: List[int] = []

        for epi in epi_records:
            mid2record[epi.id] = epi
            info = denstream.process(point=epi.embedding, now=epi.created_at)
            cidmap2mid[info['cluster_id']].append(epi.id)
            updated_cluster_id.append(info['cluster_id'])
            result.update({epi: info})
        
        score = dict(sorted(
            self.cluster_machine.cidmap2cluster.items(),
            key=lambda item: item[1].avg_pairwise_cos(), 
            reverse=True
        ))

        for cl in score.values():
            # Only abstract clusters that meet the PMC and consistency thresholds, and have been updated in this batch
            if cl.kind.value == "PMC" and cl.avg_pairwise_cos() >= consistency_threshold and cl.id in set(updated_cluster_id):
                member_ids = cidmap2mid.get(cl.id, [])
                if not member_ids:
                    continue

                episodic_notes = []
                for mid in member_ids:
                    record = midmap2record.get(mid)
                    if not record:
                        continue
                    if isinstance(record.detail, dict):
                        detail_text = _transfer_dict_to_semantic_text(record.detail)
                    else:
                        detail_text = str(record.detail)
                    tags_text = ", ".join(record.tags) if record.tags else "None"
                    episodic_notes.append(
                        "\n".join([
                            f"[EpisodicRecord {record.id}]",
                            f"Stage: {record.stage}",
                            f"Summary: {record.summary}",
                            f"Detail: {detail_text},"
                            f"Tags: {tags_text}",
                        ])
                    )

                if not episodic_notes:
                    continue

                system_prompt = "You are an expert at summarizing episodic memories into concise semantic records."
                user_prompt = ABSTRACT_EPISODIC_TO_SEMANTIC_PROMPT.format(
                    episodic_notes="\n\n".join(episodic_notes)
                )
                response = await self.llm.complete(system_prompt=system_prompt, user_prompt=user_prompt)
                sem_record_dict = json.loads(response)
                sem_record_dict['id'] = new_id("sem")
                sem_record = SemanticRecord.from_dict(sem_record_dict)
                abstract_result.append(sem_record)
                cidmap2semrec[cl.id] = sem_record
        
        return abstract_result, cidmap2semrec

    def get_nearest_k_records(self, 
            record: Union[SemanticRecord, EpisodicRecord, ProceduralRecord], 
            method: str = "embedding", 
            k: int = 5,
            filters: Dict | None = None) -> List[Tuple[float, Union[SemanticRecord, EpisodicRecord, ProceduralRecord]]]:
        if isinstance(record, SemanticRecord):
            query_text = record.detail
        elif isinstance(record, EpisodicRecord):
            query_text = record.summary
        elif isinstance(record, ProceduralRecord):
            query_text = record.description
        
        try:
            results = self.vector_store.query(query_text, method=method, limit=k, filters=filters)
        except Exception as e:
            print(f"Error querying nearest records: {e}")
            results = []
        return results

    def save(self, path: str) -> bool:
        try:
            self.vector_store.save(path)
            return True
        except Exception as e:
            return False
    
    def load(self, path: str) -> bool:
        try:
            self.vector_store.load(path)
            return True
        except Exception as e:
            return False

        
