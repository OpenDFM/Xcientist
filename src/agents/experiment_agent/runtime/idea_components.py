"""
Canonical idea-component helpers for experiment-agent.

The experiment pipeline treats `idea.json.components` as the authoritative
source for component identity and order. These helpers intentionally stay small
and deterministic so every phase can share the same component contract.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


IDEA_COMPONENTS_HEADING = "## Canonical Idea Components"
IDEA_JSON_CANDIDATE_FILENAMES = ("idea.json", "idea_result.json")


def find_idea_json_path(workspace_root: str) -> Optional[str]:
    workspace_dir = os.path.realpath(workspace_root)
    for filename in IDEA_JSON_CANDIDATE_FILENAMES:
        candidate = os.path.join(workspace_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def load_idea_json(workspace_root: str, idea_json_path: Optional[str] = None) -> Dict[str, Any]:
    path = idea_json_path or find_idea_json_path(workspace_root)
    if not path:
        raise FileNotFoundError(
            f"idea.json not found under {workspace_root}; tried {IDEA_JSON_CANDIDATE_FILENAMES}"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Idea JSON at {path} must be an object.")
    return payload


def load_canonical_components(
    workspace_root: str, idea_json_path: Optional[str] = None
) -> List[Dict[str, str]]:
    payload = load_idea_json(workspace_root, idea_json_path=idea_json_path)
    raw_components = payload.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("idea.json.components must be a non-empty list.")

    normalized: List[Dict[str, str]] = []
    seen = set()
    for index, item in enumerate(raw_components, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"idea.json.components[{index - 1}] must be an object.")
        name = str(item.get("component") or "").strip()
        explanation = str(item.get("explanation") or "").strip()
        if not name:
            raise ValueError(f"idea.json.components[{index - 1}] is missing `component`.")
        if name in seen:
            raise ValueError(f"idea.json.components contains duplicate component `{name}`.")
        seen.add(name)
        normalized.append(
            {
                "component": name,
                "explanation": explanation,
                "index": str(index),
            }
        )
    return normalized


def canonical_component_names(
    workspace_root: str, idea_json_path: Optional[str] = None
) -> List[str]:
    return [
        item["component"]
        for item in load_canonical_components(workspace_root, idea_json_path=idea_json_path)
    ]


def format_canonical_components_markdown(components: List[Dict[str, str]]) -> str:
    lines = []
    for item in components:
        index = item.get("index", "")
        name = item["component"]
        explanation = item.get("explanation", "")
        prefix = f"{index}. " if index else "- "
        lines.append(f"{prefix}`{name}`")
        if explanation:
            lines.append(f"   - {explanation}")
    return "\n".join(lines)


__all__ = [
    "IDEA_COMPONENTS_HEADING",
    "IDEA_JSON_CANDIDATE_FILENAMES",
    "canonical_component_names",
    "find_idea_json_path",
    "format_canonical_components_markdown",
    "load_canonical_components",
    "load_idea_json",
]
