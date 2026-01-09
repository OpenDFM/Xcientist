from datetime import datetime, timezone
from uuid import uuid4
from typing import Iterable, Optional, Tuple, Any, Dict, List
from pathlib import Path
from tqdm import tqdm

import logging
import json, re
import numpy as np
import concurrent.futures


def setup_logger(name: str, log_path: str, level=logging.INFO) -> logging.Logger:
    log_path = Path(log_path)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    logger.addHandler(fh)

    logger.propagate = False

    logger.info(f"logger[{name}] initialized, log file = {log_path}")

    return logger


def now_iso() -> str:
    return datetime.now().isoformat()


def new_id(prefix: str) -> str:
    uuid_hex = uuid4().hex[:8]
    return f"{prefix}_{uuid_hex}"


def compute_overlap_score(
    text: str, query: str, keywords: Optional[Iterable[str]] = None
) -> float:
    """Cheap lexical relevance score in [0, 1]."""
    if not text or not query:
        return 0.0

    STOPWORDS = {
        "a",
        "an",
        "the",
        "of",
        "and",
        "or",
        "to",
        "in",
        "on",
        "for",
        "with",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "as",
        "into",
        "up",
        "down",
    }

    text_lower = text.lower()
    query_lower = query.lower()

    query_words = [w for w in query_lower.split() if w not in STOPWORDS]
    if not query_words:
        return 0.0

    overlap = sum(1 for word in query_words if word in text_lower)
    base_score = overlap / len(query_words)

    if keywords:
        filtered_keywords = [
            kw for kw in keywords if kw and kw.lower() not in STOPWORDS
        ]
        hit_bonus = sum(
            0.1 for keyword in filtered_keywords if keyword.lower() in text_lower
        )
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
        d = v.to_dict()
        # Embeddings are stored in the FAISS index; do not persist them in meta.json.
        # This keeps meta.json small and avoids partial writes when large vectors are dumped.
        if isinstance(d, dict) and "embedding" in d:
            d.pop("embedding", None)
        output[k] = d
    return output


def dump_slot_json(slot) -> str:
    payload = {
        "id": slot.id,
        "stage": slot.stage,
        "topic": slot.topic,
        "summary": slot.summary,
        "attachments": slot.attachments,
        "tags": slot.tags,
    }
    return json.dumps(payload, ensure_ascii=False)


def _extract_json_between(text: str, open_tag: str, close_tag: str) -> Dict[str, Any]:
    m = re.search(
        rf"<{re.escape(open_tag)}>\s*(\{{.*\}})\s*</{re.escape(close_tag)}>",
        text,
        flags=re.S,
    )
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}")


def _hard_validate_slot_keys(
    payload: Dict[str, Any], allowed_keys: Iterable[str]
) -> None:
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


def _build_context_snapshot(context, state: str, char_limit: int = 4000) -> str:
    def _state_to_str(s) -> str:
        if s is None:
            return ""
        if hasattr(s, "value"):
            try:
                return str(getattr(s, "value"))
            except Exception:
                pass
        return str(s)

    state_str = _state_to_str(state)
    attr = state_str + "_output" if state_str else ""

    input_type = getattr(context, "input_type", "") if context is not None else ""
    research_input = (
        getattr(context, "research_input", "") if context is not None else ""
    )
    current_state = (
        getattr(context, "current_state", None) if context is not None else None
    )

    iteration_count = (
        getattr(context, "iteration_count", None) if context is not None else None
    )
    max_iterations = (
        getattr(context, "max_iterations", None) if context is not None else None
    )
    retry_count = getattr(context, "retry_count", None) if context is not None else None
    last_error = getattr(context, "last_error", None) if context is not None else None

    outputs_obj = None
    if attr and context is not None and hasattr(context, attr):
        try:
            outputs_obj = getattr(context, attr)
        except Exception:
            outputs_obj = None
    if outputs_obj is None and context is not None:
        # Fallbacks for the new experiment agent (step/meta-centric).
        outputs_obj = getattr(context, "meta", None)
        if outputs_obj is None:
            outputs_obj = getattr(context, "current_data", None)

    history_obj = (
        getattr(context, "state_history", None) if context is not None else None
    )
    history_items = []
    if isinstance(history_obj, list):
        for transition in history_obj:
            try:
                history_items.append(
                    {
                        "from": _state_to_str(getattr(transition, "from_state", "")),
                        "to": _state_to_str(getattr(transition, "to_state", "")),
                        "reason": str(getattr(transition, "reason", "")),
                    }
                )
            except Exception:
                continue

    snapshot = {
        "input": {
            "type": str(input_type or type(context).__name__),
            "research_excerpt": _truncate_text(str(research_input or "")),
        },
        "state": {
            "requested_state": state_str,
            "current_state": _state_to_str(current_state),
            "iteration": iteration_count,
            "max_iterations": max_iterations,
            "retry_count": retry_count,
            "last_error": last_error,
        },
        "outputs": _safe_dump(outputs_obj),
        "context": _safe_dump(context),
        "history": history_items,
    }

    serialized = json.dumps(snapshot, ensure_ascii=False, indent=2, default=str)
    return _truncate_text(serialized, limit=char_limit)


def _safe_dump(value):
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
        return [_safe_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_dump(v) for k, v in value.items()}
    return _truncate_text(str(value))


def _safe_dump_str(value) -> str:
    dumped = _safe_dump(value)
    try:
        text = json.dumps(dumped, ensure_ascii=False)
    except TypeError:
        text = str(dumped)
    return _truncate_text(text)


def _truncate_text(text: Optional[str], limit: int = 1500) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 12] + "... <truncated>"


def _push_event(event_buffer: List[str], tag: str, text: str, max_chars: int = 1500):
    text = (text or "").strip()
    if not text:
        return
    if len(text) > max_chars:
        text = text[:max_chars] + "...(truncated)"
    event_buffer.append(f"[{tag}]\n{text}")


def _drain_snapshot(event_buffer: List[str], max_chars: int = 4000) -> str:
    snapshot = "\n\n".join(event_buffer)
    if len(snapshot) > max_chars:
        snapshot = snapshot[-max_chars:]  # keep the last max_chars
    event_buffer.clear()
    return snapshot


def _multi_thread_run(func, row_data: List[Tuple], max_workers: int = 20):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(func, row_data), total=len(row_data)))


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
