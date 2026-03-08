"""Shared helper functions for MCTS parsing, normalization, fallback logic, and formatting."""

import json
import re

from typing import Any, Dict, List, Sequence, Set, Tuple

from src.agents.idea_agent.utils.prompting.prompt_views import format_analysis_prompt_view


def parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Strip potential code fences and capture the first JSON object/array.
    Fallbacks to incremental decoding if the model prepends commentary.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty response")
    if text.startswith("```"):
        fence_end = text.find("\n")
        if fence_end != -1:
            text = text[fence_end + 1 :]
        if text.endswith("```"):
            text = text[: -3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch in "{[":
                try:
                    parsed, _ = decoder.raw_decode(text[idx:])
                    return parsed
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"Unable to parse JSON from response: {text[:200]}")


def format_analysis_blob(analysis: List[Any]) -> str:
    return format_analysis_prompt_view(analysis)


def clip_text(value: Any, limit: int = 800) -> str:
    text = "" if value is None else str(value).strip()
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "..."


def _humanize_component_name(component: str) -> str:
    raw = str(component).strip()
    if not raw:
        return "component"
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
    raw = re.sub(r"[^A-Za-z0-9]+", " ", raw)
    tokens = [token for token in raw.split() if token]
    return " ".join(tokens) if tokens else "component"


def _coerce_component_name(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        preferred_keys = ("component", "name", "target", "value", "label", "title", "id")
        for key in preferred_keys:
            text = _coerce_component_name(value.get(key))
            if text:
                return text
        for item in value.values():
            text = _coerce_component_name(item)
            if text:
                return text
        return ""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            text = _coerce_component_name(item)
            if text:
                return text
        return ""
    return str(value).strip()


def _normalize_component_mapping(raw_mapping: Any) -> Dict[str, str]:
    if not isinstance(raw_mapping, dict):
        return {}

    normalized: Dict[str, str] = {}
    for raw_key, raw_value in raw_mapping.items():
        key = _coerce_component_name(raw_key)
        value = _coerce_component_name(raw_value)
        if key and value:
            normalized[key] = value
    return normalized


def _clean_component_explanation(explanation: Any, fallback_component: str) -> str:
    fallback_label = _humanize_component_name(fallback_component)

    def _flatten(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            preferred_keys = (
                "explanation",
                "description",
                "summary",
                "detail",
                "details",
                "role",
                "rationale",
                "purpose",
            )
            ordered_chunks: List[str] = []
            seen_chunks: Set[str] = set()
            for key in preferred_keys:
                raw = value.get(key)
                text = _flatten(raw).strip()
                norm = text.lower()
                if text and norm not in seen_chunks:
                    ordered_chunks.append(text)
                    seen_chunks.add(norm)
            if ordered_chunks:
                return " ".join(ordered_chunks)
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        if isinstance(value, (list, tuple, set)):
            parts: List[str] = []
            seen_parts: Set[str] = set()
            for item in value:
                text = _flatten(item).strip()
                norm = text.lower()
                if text and norm not in seen_parts:
                    parts.append(text)
                    seen_parts.add(norm)
            return " ".join(parts)
        return str(value)

    text = _flatten(explanation).strip()
    placeholder_values = {
        "",
        "n/a",
        "na",
        "none",
        "null",
        "unknown",
        "unspecified",
        "not provided",
        "no explanation",
        "no explanation provided",
        "no specific explanation provided",
        "tbd",
    }
    if text.lower() in placeholder_values:
        return f"No specific explanation provided for {fallback_label}."

    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n-*:;,.")
    text = re.sub(r"^['\"`]+|['\"`]+$", "", text).strip()

    prefix_patterns = [
        rf"^{re.escape(str(fallback_component).strip())}\s*[:\-]\s*",
        rf"^{re.escape(fallback_label)}\s*[:\-]\s*",
        r"^(component|module|role|purpose|description|explanation)\s*[:\-]\s*",
    ]
    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    if not text:
        return f"No specific explanation provided for {fallback_label}."

    text = clip_text(text, limit=3000)
    if text[-1] not in ".!?":
        text += "."
    return text


def normalize_component_explanations(
    components: Sequence[str],
    raw_explanations: Any,
) -> Dict[str, str]:
    normalized_components: List[str] = []
    seen: Set[str] = set()
    for component in components:
        name = str(component).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        normalized_components.append(name)

    lookup: Dict[str, str] = {}
    if isinstance(raw_explanations, dict):
        for key, value in raw_explanations.items():
            name = str(key).strip()
            if not name:
                continue
            lookup[name] = _clean_component_explanation(value, fallback_component=name)
    elif isinstance(raw_explanations, list):
        for item in raw_explanations:
            if not isinstance(item, dict):
                continue
            name = str(item.get("component") or item.get("name") or "").strip()
            if not name:
                continue
            lookup[name] = _clean_component_explanation(
                item.get("explanation", ""),
                fallback_component=name,
            )

    return {
        component: lookup.get(component, _clean_component_explanation("", fallback_component=component))
        for component in normalized_components
    }


def component_inventory_payload(
    components: Sequence[str],
    component_explanations: Any,
) -> List[Dict[str, str]]:
    normalized = normalize_component_explanations(components, component_explanations)
    return [
        {
            "component": component,
            "explanation": normalized.get(component, _clean_component_explanation("", component)),
        }
        for component in normalized
    ]


def parse_component_bundle_payload(
    payload: Any,
    *,
    max_components: int,
) -> Tuple[List[str], Dict[str, str]]:
    if not isinstance(payload, dict):
        return [], {}

    raw_components = payload.get("components", [])
    ordered_components: List[str] = []
    explanations: Dict[str, str] = {}
    seen: Set[str] = set()

    if isinstance(raw_components, list):
        for item in raw_components:
            if len(ordered_components) >= max_components:
                break
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("component") or "").strip()
                explanation = item.get("explanation", "")
            else:
                name = str(item).strip()
                explanation = ""
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)
            ordered_components.append(name)
            if explanation:
                explanations[name] = _clean_component_explanation(
                    explanation,
                    fallback_component=name,
                )

    external_explanations = normalize_component_explanations(
        ordered_components,
        payload.get("component_explanations"),
    )
    explanations = {
        component: explanations.get(component)
        or external_explanations.get(component)
        or _clean_component_explanation("", fallback_component=component)
        for component in ordered_components
    }
    return ordered_components, explanations


def _dedupe_keep_order_strings(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        text = str(item).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _safe_pretty_json(value: Any, pretty_json: Any = None) -> str:
    if pretty_json:
        try:
            return pretty_json(value)
        except Exception:
            pass
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _safe_float_default(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalize_budget_dict(budget: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(budget, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in budget.items():
        if isinstance(value, (int, float)):
            cleaned[str(key)] = float(value)
        else:
            try:
                cleaned[str(key)] = float(value)
            except (TypeError, ValueError):
                cleaned[str(key)] = value
    return cleaned


def apply_budget_delta_to_parent(
    parent_budget: Dict[str, Any],
    delta: Dict[str, Any],
) -> Dict[str, Any]:
    next_budget = normalize_budget_dict(parent_budget)
    for key, val in (delta or {}).items():
        if key not in next_budget:
            next_budget[key] = _safe_float_default(val, 0.0)
            continue
        base = next_budget.get(key)
        if isinstance(base, (int, float)):
            next_budget[key] = round(float(base) + _safe_float_default(val, 0.0), 4)
    return next_budget


def plan_to_method_text(plan: Any) -> str:
    lines: List[str] = []
    for idx, edit in enumerate(getattr(plan, "component_edits", []) or [], start=1):
        op = getattr(getattr(edit, "op", None), "value", getattr(edit, "op", ""))
        target = f" -> {getattr(edit, 'target', '')}" if getattr(edit, "target", "") else ""
        condition = (
            f" [condition: {getattr(edit, 'condition', '')}]"
            if getattr(edit, "condition", "")
            else ""
        )
        details = f"; {getattr(edit, 'details', '')}" if getattr(edit, "details", "") else ""
        lines.append(f"{idx}. {op}({getattr(edit, 'component', '')}{target}){condition}{details}")
    return "\n".join(lines)


def plan_to_experiment_text(plan: Any) -> str:
    blocks: List[str] = []
    validation = getattr(plan, "validation", None)
    regression_tests = getattr(validation, "regression_tests", []) if validation else []
    ablation_tests = getattr(validation, "ablation_tests", []) if validation else []
    stress_tests = getattr(validation, "stress_tests", []) if validation else []
    if regression_tests:
        blocks.append("Regression:\n- " + "\n- ".join(regression_tests))
    if ablation_tests:
        blocks.append("Ablation:\n- " + "\n- ".join(ablation_tests))
    if stress_tests:
        blocks.append("Stress:\n- " + "\n- ".join(stress_tests))
    return "\n\n".join(blocks)
