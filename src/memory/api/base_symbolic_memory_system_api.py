from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_symbolic_id(prefix: str = "sym") -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


class SymbolicMemorySystemConfig(BaseModel):
    upsert_threshold: float = Field(
        0.82,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for upsert deduplication.",
    )


class SymbolicRecordPayload(BaseModel):
    component: str = Field(
        ...,
        description="Ablated component name from the experiment report.",
    )
    component_family: str = Field(
        ...,
        description="Derived component family used for hierarchical retrieval.",
    )
    result: str = Field(
        "inconclusive",
        description="Qualitative ablation result label: positive, negative, or inconclusive.",
    )
    metric: str = Field(
        "",
        description="Metric name reported for this ablation result.",
    )
    value: str = Field(
        "",
        description="Raw metric value or textual measurement from the ablation report.",
    )
    analysis: str = Field(
        "",
        description="Free-text interpretation of the ablation result.",
    )
    method_context: str = Field(
        "",
        description=(
            "Idea introduction used in the ablation experiment after removing "
            "this component."
        ),
    )
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence attached to the ablation result.",
    )


class SymbolicRecord(BaseModel):
    id: str = Field(default_factory=_new_symbolic_id)
    component: str
    component_family: str
    result: str = "inconclusive"
    metric: str = ""
    value: str = ""
    analysis: str = ""
    method_context: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SymbolicRecord":
        return cls(**payload)

    def update(
        self,
        component: Optional[str] = None,
        component_family: Optional[str] = None,
        result: Optional[str] = None,
        metric: Optional[str] = None,
        value: Optional[str] = None,
        analysis: Optional[str] = None,
        method_context: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        if component is not None:
            self.component = component
        if component_family is not None:
            self.component_family = component_family
        if result is not None:
            self.result = result
        if metric is not None:
            self.metric = metric
        if value is not None:
            self.value = value
        if analysis is not None:
            self.analysis = analysis
        if method_context is not None:
            self.method_context = method_context
        if confidence is not None:
            self.confidence = max(0.0, min(1.0, float(confidence)))


class SymbolicMemorySystem(ABC):
    @abstractmethod
    def instantiate_symbolic_record(self, **kwargs) -> SymbolicRecord:
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def get_records_by_ids(self, mids: List[str]) -> List[SymbolicRecord]:
        ...

    @abstractmethod
    def get_last_k_records(self, k: int) -> Tuple[List[Dict[str, Any]], int]:
        ...

    @abstractmethod
    def is_exists(self, mids: List[str]) -> List[bool]:
        ...

    @abstractmethod
    def add(self, memories: List[SymbolicRecord], agent_id: str = "") -> bool:
        ...

    @abstractmethod
    def update(self, memories: List[SymbolicRecord]) -> bool:
        ...

    @abstractmethod
    def delete(self, mids: List[str]) -> bool:
        ...

    @abstractmethod
    def upsert_normal_records(
        self,
        records: List[SymbolicRecord],
        agent_id: str = "",
    ) -> None:
        ...

    @abstractmethod
    def query(
        self,
        query_text: str,
        method: str = "lexical",
        limit: int = 5,
        filters: Optional[Dict] = None,
        threshold: float = 0.0,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        ...

    @abstractmethod
    def get_nearest_k_records(
        self,
        record: SymbolicRecord,
        method: str = "lexical",
        k: int = 5,
        filters: Optional[Dict] = None,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        ...

    def retrieve_hierarchical(
        self,
        target_component: str = "",
        target_family: str = "",
        limit: int = 2,
        threshold: float = 0.0,
        agent_id: str = "",
        query_context: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> bool:
        ...

    @abstractmethod
    def load(self, path: str) -> bool:
        ...
