"""Canonical idea-contract normalization and legacy alias migration helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping


LEGACY_IDEA_ALIASES = {
    "core_contribute": "core_contribution",
    "methodology": "method",
}

IDEA_CONTRACT_FIELDS = (
    "title",
    "abstract",
    "core_contribution",
    "method",
    "risks",
    "tags",
    "operator",
    "target_defects",
    "rationale",
    "memory_refs",
    "components",
    "component_explanations",
    "root_domains",
    "paper_graph_context",
    "edit_plan",
    "skill_metrics",
)

_REQUIRED_TEXT_FIELDS = (
    "title",
    "abstract",
    "core_contribution",
    "method",
)


def _cmp_key(value: Any) -> str:
    return str(value).strip()


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_str_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key).strip(): str(item).strip()
        for key, item in value.items()
        if str(key).strip() and str(item).strip()
    }


def normalize_idea_contract(
    payload: Any,
    *,
    allow_legacy: bool = False,
    keep_extra: bool = False,
) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Idea contract must be a mapping.")

    raw = dict(payload)
    raw.pop("experiments", None)
    raw.pop("experiment_design", None)
    for legacy_key, canonical_key in LEGACY_IDEA_ALIASES.items():
        if legacy_key not in raw:
            continue
        if not allow_legacy:
            raise ValueError(f"Legacy idea key is not allowed: {legacy_key}")
        legacy_value = raw.pop(legacy_key)
        if canonical_key in raw and _cmp_key(raw[canonical_key]) != _cmp_key(legacy_value):
            raise ValueError(f"Conflicting idea fields: {canonical_key} vs {legacy_key}")
        raw.setdefault(canonical_key, legacy_value)

    idea = {
        "title": _as_text(raw.get("title")),
        "abstract": _as_text(raw.get("abstract")),
        "core_contribution": _as_text(raw.get("core_contribution")),
        "method": _as_text(raw.get("method")),
        "risks": _as_text(raw.get("risks")),
        "tags": _as_list(raw.get("tags")),
        "operator": _as_text(raw.get("operator")),
        "target_defects": _as_list(raw.get("target_defects")),
        "rationale": _as_text(raw.get("rationale")),
        "memory_refs": _as_list(raw.get("memory_refs")),
        "components": _as_list(raw.get("components")),
        "component_explanations": _as_str_dict(raw.get("component_explanations")),
        "root_domains": _as_list(raw.get("root_domains")),
        "paper_graph_context": _as_text(raw.get("paper_graph_context")),
        "edit_plan": raw.get("edit_plan") if isinstance(raw.get("edit_plan"), Mapping) else None,
        "skill_metrics": _as_dict(raw.get("skill_metrics")),
    }
    missing = [field for field in _REQUIRED_TEXT_FIELDS if not idea[field]]
    if missing:
        raise ValueError(f"Idea contract missing required fields: {', '.join(missing)}")

    if not keep_extra:
        return idea

    extras = {key: value for key, value in raw.items() if key not in IDEA_CONTRACT_FIELDS}
    return {**idea, **extras}
