import hashlib
import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionKind(str, Enum):
    CODE_CHANGE = "CODE_CHANGE"
    SCIENCE_CHANGE = "SCIENCE_CHANGE"
    VERIFY = "VERIFY"
    NOOP = "NOOP"
    STOP = "STOP"


class ActionSource(BaseModel):
    namespace: str = Field(
        default="", description="Source layer namespace, e.g. science/code"
    )
    step_index: int = Field(
        default=0, description="Source step index that triggered this action plan"
    )
    blueprint_id: str = Field(
        default="", description="Source blueprint id (plan id / blueprint hash)"
    )
    iteration: int = Field(
        default=0, description="Optional iteration counter in the source loop"
    )


class Action(BaseModel):
    action_id: str = Field(description="Stable action id for idempotent resume")
    kind: ActionKind = Field(description="Action type")
    depends_on: List[str] = Field(
        default_factory=list, description="Upstream action_ids"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Action parameters"
    )
    expected_artifacts: List[str] = Field(
        default_factory=list, description="Optional expected outputs"
    )


class ActionPlan(BaseModel):
    plan_id: str = Field(description="Stable plan id (content-addressed)")
    source: ActionSource = Field(description="Where this plan came from")
    policy_snapshot: Dict[str, Any] = Field(
        default_factory=dict, description="Decision policy metadata for audit"
    )
    actions: List[Action] = Field(
        default_factory=list, description="Ordered actions to execute"
    )
    stop: bool = Field(default=False, description="Whether to stop the outer loop")


def _stable_hash8(data: Any) -> str:
    blob = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:8]


def action_task_id(action_id: str) -> str:
    return f"__action__/{str(action_id)}"


def build_action_plan_from_science(
    analysis: Any,
    results: Optional[List[Any]] = None,
    source: Optional[ActionSource] = None,
    policy_snapshot: Optional[Dict[str, Any]] = None,
) -> ActionPlan:
    """
    Convert a ScienceAnalysis into a unified ActionPlan.

    Notes:
    - This is intentionally conservative: it only encodes actions that already exist in the system
      (next_experiments and optimization_tickets). It does NOT invent new actions.
    - Idempotency is guaranteed by stable action_id derived from action content.
    """
    _ = results
    src = source or ActionSource(
        namespace="science", step_index=0, blueprint_id="", iteration=0
    )
    policy = dict(policy_snapshot or {})

    try:
        analysis_data = (
            analysis.model_dump() if hasattr(analysis, "model_dump") else dict(analysis)
        )
    except Exception:
        analysis_data = {"analysis": str(analysis)}

    actions: List[Action] = []

    success = bool(analysis_data.get("success", False))
    next_experiments = analysis_data.get("next_experiments")
    tickets = analysis_data.get("optimization_tickets") or []

    if next_experiments:
        payload = {"next_experiments": next_experiments}
        actions.append(
            Action(
                action_id=f"science_change_{_stable_hash8(payload)}",
                kind=ActionKind.SCIENCE_CHANGE,
                payload=payload,
            )
        )

    if tickets:
        payload = {"tickets": tickets}
        actions.append(
            Action(
                action_id=f"code_change_{_stable_hash8(payload)}",
                kind=ActionKind.CODE_CHANGE,
                payload=payload,
            )
        )

    if success:
        actions.append(
            Action(
                action_id=f"stop_{_stable_hash8({'stop': True})}",
                kind=ActionKind.STOP,
                payload={"reason": "analysis.success"},
            )
        )
        stop = True
    else:
        stop = False

    plan_id = f"ap_{src.namespace}_{src.step_index:04d}_{_stable_hash8({'source': src.model_dump(), 'analysis': analysis_data, 'policy': policy})}"
    return ActionPlan(
        plan_id=plan_id, source=src, policy_snapshot=policy, actions=actions, stop=stop
    )
