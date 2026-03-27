from __future__ import annotations

from typing import Optional


DEFAULT_PROMPT_MODE = "default"
CONCEPTUAL_SURPRISE_PROMPT_MODE = "conceptual_surprise"


def normalize_prompt_mode(mode: Optional[str]) -> str:
    value = str(mode or DEFAULT_PROMPT_MODE).strip().lower()
    return value or DEFAULT_PROMPT_MODE


def is_conceptual_surprise_mode(mode: Optional[str]) -> bool:
    return normalize_prompt_mode(mode) == CONCEPTUAL_SURPRISE_PROMPT_MODE
