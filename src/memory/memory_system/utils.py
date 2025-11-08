from datetime import datetime, timezone
from uuid import uuid4
from typing import Iterable, Optional, Tuple, Any, Dict

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
    