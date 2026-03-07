"""Explicit stage contracts for LigAgent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


StageStatus = Literal[
    "success",
    "degraded",
    "retryable_failure",
    "terminal_failure",
    "skipped",
]


@dataclass
class ArtifactPatch:
    """Typed artifact mutations applied by the workflow executor."""

    replace: Dict[str, Any] = field(default_factory=dict)
    append: Dict[str, List[Any]] = field(default_factory=dict)
    merge: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.replace or self.append or self.merge)


@dataclass
class StageContext:
    """Per-stage execution context."""

    agent: Any
    artifact: Dict[str, Any]
    workflow_name: str
    stage_name: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    logger: Any = None
    attempt: int = 1

    def for_stage(
        self,
        stage_name: str,
        *,
        state: Dict[str, Any],
        attempt: int,
    ) -> "StageContext":
        return StageContext(
            agent=self.agent,
            artifact=self.artifact,
            workflow_name=self.workflow_name,
            stage_name=stage_name,
            state=state,
            inputs=self.inputs,
            logger=self.logger,
            attempt=attempt,
        )


@dataclass
class StageResult:
    """Canonical output contract for all workflow stages."""

    status: StageStatus = "success"
    artifact_patch: ArtifactPatch = field(default_factory=ArtifactPatch)
    state_patch: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    step_summary: Optional[str] = None
    next_stage: Optional[str] = None
    error: Optional[str] = None
