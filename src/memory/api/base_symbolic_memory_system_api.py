from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now().isoformat()


def _new_symbolic_id(prefix: str = "sym") -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


class SymbolicMemorySystemConfig(BaseModel):
    lexical_weight: float = Field(
        0.45,
        ge=0.0,
        le=1.0,
        description="Weight for lexical matching in hybrid query.",
    )
    rule_weight: float = Field(
        0.45,
        ge=0.0,
        le=1.0,
        description="Weight for symbolic rule matching in hybrid query.",
    )
    recency_weight: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Weight for recency in hybrid query.",
    )
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
    op: str = Field(
        "remove",
        description=(
            "Component-level operation represented by the ablation evidence. "
            "For ablation_results.json this is usually 'remove', because the "
            "experiment evaluates the idea after removing the component."
        ),
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
    run_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Top-level summary block from ablation_results.json.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata for debugging and filtering.",
    )
    support_count: int = Field(
        1,
        ge=1,
        description="How many times this evidence pattern has been observed.",
    )


class SymbolicRecord(BaseModel):
    id: str = Field(default_factory=_new_symbolic_id)
    component: str
    component_family: str
    op: str = "remove"
    result: str = "inconclusive"
    metric: str = ""
    value: str = ""
    analysis: str = ""
    method_context: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    run_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    support_count: int = Field(1, ge=1)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SymbolicRecord":
        return cls(**payload)

    def update(
        self,
        component: Optional[str] = None,
        component_family: Optional[str] = None,
        op: Optional[str] = None,
        result: Optional[str] = None,
        metric: Optional[str] = None,
        value: Optional[str] = None,
        analysis: Optional[str] = None,
        method_context: Optional[str] = None,
        confidence: Optional[float] = None,
        run_summary: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        support_count: Optional[int] = None,
    ) -> None:
        if component is not None:
            self.component = component
        if component_family is not None:
            self.component_family = component_family
        if op is not None:
            self.op = op
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
        if run_summary is not None:
            self.run_summary = dict(run_summary)
        if metadata is not None:
            self.metadata = dict(metadata)
        if support_count is not None:
            self.support_count = max(1, int(support_count))
        self.updated_at = _now_iso()


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
        method: str = "hybrid",
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
        method: str = "hybrid",
        k: int = 5,
        filters: Optional[Dict] = None,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        ...

    def retrieve_hierarchical(
        self,
        target_family: str = "",
        context_sig: Optional[Any] = None,
        op: str = "",
        limit: int = 10,
        threshold: float = 0.0,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> bool:
        ...

    @abstractmethod
    def load(self, path: str) -> bool:
        ...
