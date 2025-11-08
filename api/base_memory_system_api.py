from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, validate_call
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

from memory_system import (
    FaissVectorStore,
    SemanticRecord,
    EpisodicRecord,
    ProceduralRecord,
)


class MemorySystemConfig(BaseModel):
    memory_type: Literal["semantic", "episodic", "procedural", "working"] = "semantic"
    model_path: str = Field("./.cache/all-MiniLM-L6-v2", description="Path to the model used for vector embeddings.")
    llm_name: str = Field("gpt-4.1-mini", description="Name of the LLM model to be used.")

class MemoryRecordPayload(BaseModel):
    name: str = Field("", description="Name of the ProceduralRecord.")
    description: str = Field("", description="Description of the ProceduralRecord.")
    summary: str = Field("", description="A brief summary of the SemanticRecord/EpisodicRecord.")
    detail: Union[str, dict] = Field("", description="Detailed information about the SemanticRecord/EpisodicRecord.")
    steps: Optional[List[str]] = Field(None, description="Steps for the ProceduralRecord.")
    code: Optional[str] = Field(None, description="Code snippet for the ProceduralRecord.")
    stage: str = Field("", description="Stage of the memory.")
    tags: Optional[Iterable[str]] = Field(None, description="Tags associated with the memory.")

class MemorySystem(ABC):
    @abstractmethod
    def instantiate_sem_record(self, **kwargs) -> SemanticRecord:
        ...
    
    @abstractmethod
    def instantiate_epi_record(self, **kwargs) -> EpisodicRecord:
        ...
    
    @abstractmethod
    def instantiate_proc_record(self, **kwargs) -> ProceduralRecord:
        ...
    
    @abstractmethod
    def size(self) -> int:
        ...
    
    @abstractmethod
    def get_records_by_ids(self, mids: List[str]) -> Union[List[SemanticRecord], List[EpisodicRecord], List[ProceduralRecord]]:
        ...
    
    @abstractmethod
    def get_last_k_records(self, k: int) -> Tuple[List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]], int]:
        ...

    @abstractmethod
    def is_exists(self, mids: List[str]) -> List[bool]:
        ...

    @abstractmethod
    def add(self, memories: List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]]) -> bool:
        ...
    
    @abstractmethod
    def update(self, memories: List[Union[SemanticRecord, ProceduralRecord]]) -> bool:
        ...
    
    @abstractmethod
    def batch_memory_process(self, memories: List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]]) -> bool:
        ...
    
    @abstractmethod
    def delete(self, mids: List[str]) -> bool:
        ...
    
    @abstractmethod
    def query(self, query_text: str, method: str = "embedding", limit: int = 5, filters: Dict | None = None) -> List[Tuple[float, List[Union[SemanticRecord, EpisodicRecord, ProceduralRecord]]]]:
        ...
    
    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...