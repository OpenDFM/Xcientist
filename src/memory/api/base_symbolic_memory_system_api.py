from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now().isoformat()


def _new_symbolic_id(prefix: str = "sym") -> str:
    return f"{prefix}_{uuid4().hex[:8]}"


class SymbolicMemorySystemConfig(BaseModel):
    memory_type: Literal["symbolic"] = "symbolic"
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
    summary: str = Field("", description="Short summary for the symbolic prior.")
    pattern: str = Field(
        "",
        description="Problem pattern or decision context where this prior applies.",
    )
    conditions: Optional[Iterable[str]] = Field(
        None,
        description="Preconditions for applying this symbolic prior.",
    )
    actions: Optional[Iterable[str]] = Field(
        None,
        description="Recommended action sequence or operator hints.",
    )
    rationale: str = Field(
        "",
        description="Reasoning why this prior is expected to work.",
    )
    expected_outcomes: Optional[Iterable[str]] = Field(
        None,
        description="Expected outcomes after applying actions.",
    )
    anti_patterns: Optional[Iterable[str]] = Field(
        None,
        description="Known failure modes to avoid during expansion.",
    )
    tags: Optional[Iterable[str]] = Field(
        None,
        description="Tags for retrieval and filtering.",
    )
    priority: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Relative priority when multiple priors match.",
    )
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Estimated confidence score for this symbolic memory.",
    )
    source: str = Field("", description="Source of this symbolic memory.")
    support_count: int = Field(
        1,
        ge=1,
        description="How many times this prior has been validated.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata for filtering and debugging.",
    )


class SymbolicRecord(BaseModel):
    id: str = Field(default_factory=_new_symbolic_id)
    summary: str
    pattern: str
    conditions: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    rationale: str = ""
    expected_outcomes: List[str] = Field(default_factory=list)
    anti_patterns: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    priority: float = Field(0.5, ge=0.0, le=1.0)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    source: str = ""
    support_count: int = Field(1, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SymbolicRecord":
        return cls(**payload)

    def update(
        self,
        summary: Optional[str] = None,
        pattern: Optional[str] = None,
        conditions: Optional[Iterable[str]] = None,
        actions: Optional[Iterable[str]] = None,
        rationale: Optional[str] = None,
        expected_outcomes: Optional[Iterable[str]] = None,
        anti_patterns: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        priority: Optional[float] = None,
        confidence: Optional[float] = None,
        source: Optional[str] = None,
        support_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if summary is not None:
            self.summary = summary
        if pattern is not None:
            self.pattern = pattern
        if conditions is not None:
            self.conditions = list(conditions)
        if actions is not None:
            self.actions = list(actions)
        if rationale is not None:
            self.rationale = rationale
        if expected_outcomes is not None:
            self.expected_outcomes = list(expected_outcomes)
        if anti_patterns is not None:
            self.anti_patterns = list(anti_patterns)
        if tags is not None:
            self.tags = list(tags)
        if priority is not None:
            self.priority = max(0.0, min(1.0, float(priority)))
        if confidence is not None:
            self.confidence = max(0.0, min(1.0, float(confidence)))
        if source is not None:
            self.source = source
        if support_count is not None:
            self.support_count = max(1, int(support_count))
        if metadata is not None:
            self.metadata = dict(metadata)
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

    @abstractmethod
    def retrieve_priors_for_expand(
        self,
        topic: str,
        operator: str = "",
        defects: Optional[List[str]] = None,
        limit: int = 5,
        threshold: float = 0.0,
        agent_id: str = "",
    ) -> List[Tuple[float, SymbolicRecord]]:
        ...

    @abstractmethod
    def save(self, path: str) -> bool:
        ...

    @abstractmethod
    def load(self, path: str) -> bool:
        ...
