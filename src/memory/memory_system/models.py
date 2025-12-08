from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterable
from .utils import now_iso


class EpisodicRecord(object):
    def __init__(
        self,
        id: str,
        stage: str, 
        summary: str, 
        detail: Dict[str, Any], 
        tags: Optional[Iterable[str]] = None,  
        created_at: str = "", 
    ):
        self.id = id
        self.stage = stage
        self.summary = summary
        self.detail = detail or {}
        self.embedding = None  # to be set in the vectorstore
        self.tags = tags or []
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "summary": self.summary,
            "detail": self.detail,
            "embedding": self.embedding,
            "tags": list(self.tags),
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> EpisodicRecord:
        return cls(
            id=payload.get("id", ""),
            stage=payload.get("stage", ""),
            summary=payload.get("summary", ""),
            detail=payload.get("detail", {}),
            tags=payload.get("tags"),
            created_at=payload.get("created_at", ""),
        )


class SemanticRecord(object):
    def __init__(
        self,
        id: str, 
        summary: str,
        detail: str, 
        tags: Optional[Iterable[str]] = None,
        is_abstracted: bool = False,
        created_at: str = "", 
        updated_at: str = "",
    ):
        self.id = id
        self.summary = summary
        self.detail = detail
        self.tags = tags or []
        self.is_abstracted = is_abstracted
        self.cluster_id = None  # to be set in the denstream
        self.created_at = created_at
        self.updated_at = updated_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "summary": self.summary,
            "detail": self.detail,
            "tags": list(self.tags),
            "cluster_id": self.cluster_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def update(
        self,
        summary: Optional[str] = None,
        detail: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        if summary is not None:
            self.summary = summary
        if detail is not None:
            self.detail = detail
        if tags is not None:
            self.tags = list(tags)
        self.updated_at = now_iso()
        
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> SemanticRecord:
        return cls(
            id=payload.get("id", ""),
            summary=payload.get("summary", ""),
            detail=payload.get("detail", ""),
            tags=payload.get("tags"),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
        )


class ProceduralRecord(object):
    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        steps: List[str],
        code: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        created_at: str = "",
        updated_at: str = "",
    ):
        self.id = id
        self.name = name
        self.description = description
        self.steps = steps
        self.code = code
        self.tags = tags
        self.created_at = created_at
        self.updated_at = updated_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": list(self.steps),
            "code": self.code,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> ProceduralRecord:
        return cls(
            id=payload.get("id", ""),
            name=payload.get("name", ""),
            description=payload.get("description", ""),
            steps=payload.get("steps", []),
            code=payload.get("code"),
            tags=payload.get("tags"),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
        )
    
    def update(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        steps: Optional[Iterable[str]] = None,
        code: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if steps is not None:
            self.steps = list(steps)
        if code is not None:
            self.code = code
        if tags is not None:
            self.tags = list(tags)
        self.updated_at = now_iso()