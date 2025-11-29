from datetime import datetime, timezone
from uuid import uuid4
from typing import Iterable, Optional, Tuple, Any, Dict
#from src.agents.experiment_agent.sub_agents.experiment_master.workflow_state_machine import WorkflowContext


import json, re
import numpy as np


def now_iso() -> str:
    return datetime.now().isoformat()


def new_id(prefix: str) -> str:
    uuid_hex = uuid4().hex[:8]
    return f"{prefix}_{uuid_hex}"


def compute_overlap_score(text: str, query: str, keywords: Optional[Iterable[str]] = None) -> float:
    """Cheap lexical relevance score in [0, 1]."""
    if not text or not query:
        return 0.0
    text_lower = text.lower()
    query_lower = query.lower()
    overlap = sum(1 for word in query_lower.split() if word in text_lower)
    base_score = overlap / max(len(query_lower.split()), 1)
    if keywords:
        hit_bonus = sum(0.1 for keyword in keywords if keyword.lower() in text_lower)
    else:
        hit_bonus = 0.0
    return min(1.0, base_score + hit_bonus)


def ensure_tuple(value: Optional[Iterable]) -> Tuple:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    return tuple(value)


def _nomralize_embedding(emb: np.float32) -> np.float32:
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm 


def _jsonable_meta(meta: dict) -> dict:
    output = {}
    for k, v in meta.items():
        output[k] = v.to_dict()
    return output


def dump_slot_json(slot) -> str:
    payload = {
        "id": slot.id,
        "stage": slot.stage,
        "topic": slot.topic,
        "summary": slot.summary,
        "attachments": slot.attachments,
        "tags": slot.tags
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_json_between(text: str, open_tag: str, close_tag: str) -> Dict[str, Any]:
    m = re.search(rf"<{re.escape(open_tag)}>\s*(\{{.*\}})\s*</{re.escape(close_tag)}>", text, flags=re.S)
    if not m:
        raise ValueError(f"Missing <{open_tag}> JSON block.")
    try:
        return json.loads(m.group(1))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}")


def _hard_validate_slot_keys(payload: Dict[str, Any], allowed_keys: Iterable[str]) -> None:
    extra = set(payload.keys()) - allowed_keys
    if extra:
        raise ValueError(f"Unexpected keys in slot payload: {extra}")


def _transfer_dict_to_semantic_text(d: Dict[str, Any], prefix: str = "") -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(_transfer_dict_to_semantic_text(v, prefix + " "))
        elif isinstance(v, list):
            joined = ", ".join(str(item) for item in v)
            lines.append(f"{prefix}{k}: {joined}")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


def _build_context_snapshot(self, context, state: str, char_limit: int = 4000) -> str:
    attr = state + "_output"

    snapshot = {
        "input": {
            "type": context.input_type,
            "research_excerpt": self._truncate_text(context.research_input),
        },
        "state": {
            "current_state": context.current_state.value,
            "iteration": context.iteration_count,
            "max_iterations": context.max_iterations,
            "retry_count": context.retry_count,
            "last_error": context.last_error,
        },
        "outputs": self._safe_dump(getattr(context, attr)),
        "history": [
            {
                "from": transition.from_state.value,
                "to": transition.to_state.value,
                "reason": transition.reason,
            }
            for transition in (context.state_history or [])
        ],
    }

    serialized = json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)
    return self._truncate_text(serialized, limit=char_limit)


def _safe_dump(self, value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            pass
    if isinstance(value, list):
        return [self._safe_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: self._safe_dump(v) for k, v in value.items()}
    return self._truncate_text(str(value))


def _truncate_text(self, text: Optional[str], limit: int = 1500) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 12] + "... <truncated>"