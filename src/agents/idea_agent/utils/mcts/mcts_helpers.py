"""Shared helper functions for MCTS parsing, normalization, fallback logic, and formatting."""

import re

from typing import Any, Dict, List, Sequence, Set, Tuple

from src.agents.idea_agent.utils.core.json_utils import compact_json
from src.agents.idea_agent.utils.prompting.prompt_views import format_analysis_prompt_view


ROOT_DOMAIN_CATALOG: Dict[str, str] = {
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.DS": "Data Structures and Algorithms",
    "cs.GT": "Computer Science and Game Theory",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "cs.SI": "Social and Information Networks",
    "stat.ML": "Machine Learning (Statistics)",
}
DEFAULT_ROOT_DOMAIN = "cs.LG"
_CONTROL_CORE_RE = re.compile(
    r"\b(threshold\w*|gat\w*|suppress\w*|quota\w*|controller\w*|control\w*)\b",
    re.IGNORECASE,
)
_FORBIDDEN_QUERY_TERM_RE = re.compile(
    r"\b(threshold\w*|gat\w*|suppress\w*|quota\w*)\b",
    re.IGNORECASE,
)
_EXECUTION_PATH_SCOPE_RE = re.compile(
    r"\b(path|branch|fallback|route|routing|pipeline|repair|recovery|speculat\w*)\b",
    re.IGNORECASE,
)
_BROAD_ARCHITECTURE_SCOPE_RE = re.compile(
    r"\b(hierarch\w*|architecture|system|multi scale|multiscale|coordinat\w*|global)\b",
    re.IGNORECASE,
)
_MULTI_PATH_SHAPE_RE = re.compile(
    r"\b(path|branch|fallback|repair|hierarch\w*|multi scale|multiscale|feedback|monitor|speculat\w*)\b",
    re.IGNORECASE,
)
_TRAINING_FREE_RE = re.compile(
    r"\b(training free|inference time|no training|without training|frozen)\b",
    re.IGNORECASE,
)
_SCOPE_KIND_BY_DEFECT: Dict[str, Set[str]] = {
    "existing_component": {
        "stagnant_novelty",
        "unclear_mechanism",
        "validation_gap",
        "feature_dumping",
        "harder_to_ablate",
    },
    "existing_subsystem": {
        "theory_gap",
        "weak_generalization",
    },
    "execution_path": {
        "brittle_single_path",
        "rare_regime_failure",
        "weak_fallback_behavior",
        "silent_failure",
        "drift",
        "open_loop_fragility",
        "over_conservative_execution",
        "rollback_blindspot",
        "latency_bottleneck",
    },
    "broad_architecture": {
        "monolithic_design",
        "scale_mismatch",
        "coordination_failure",
        "responsibility_entanglement",
    },
}
_SCOPE_KIND_PRIORITY: Dict[str, int] = {
    "existing_subsystem": 0,
    "existing_component": 1,
    "execution_path": 2,
    "broad_architecture": 3,
}
_ROOT_DOMAIN_FALLBACK_RULES: Tuple[Tuple[str, str], ...] = (
    ("cs.CV", "vision image video segmentation detection visual multimodal camera pixel"),
    ("cs.CL", "language text llm nlp translation summarization dialogue speech token prompt"),
    ("cs.RO", "robot robotics slam manipulation control navigation drone embodied"),
    ("cs.CR", "security privacy cryptography secure attack adversarial authentication"),
    ("cs.DS", "algorithm algorithms graph shortest path combinatorial data structure"),
    ("cs.GT", "game theory auction bandit equilibrium mechanism design"),
    ("cs.SI", "social network information diffusion recommendation influence graph"),
    ("cs.NE", "neural evolutionary neuroscience spike spiking brain"),
    ("stat.ML", "bayesian posterior statistical statistics estimator inference uncertainty"),
    ("cs.LG", "learning machine learning training optimization representation model"),
    ("cs.AI", "reasoning planning agent artificial intelligence knowledge"),
)


def normalize_term_scan_text(text: str) -> str:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    normalized = normalized.replace("_", " ").replace("-", " ")
    return normalized.lower()


def _parent_state_text(parent_state: Any) -> str:
    return "\n".join(
        [
            parent_state.title,
            parent_state.abstract,
            parent_state.core_contribution,
            parent_state.method,
            " ".join(parent_state.components or []),
            " ".join((parent_state.component_explanations or {}).values()),
        ]
    )


def parent_centers_control_surface(parent_state: Any) -> bool:
    return bool(_CONTROL_CORE_RE.search(normalize_term_scan_text(_parent_state_text(parent_state))))


def text_uses_forbidden_query_terms(text: str) -> bool:
    return bool(_FORBIDDEN_QUERY_TERM_RE.search(normalize_term_scan_text(text)))


def _infer_scope_kind_from_defects(defect_tags: Sequence[str]) -> str:
    counts = {scope_kind: 0 for scope_kind in _SCOPE_KIND_BY_DEFECT}
    for defect in defect_tags or []:
        normalized = str(defect).strip().lower()
        if not normalized:
            continue
        for scope_kind, tagged_defects in _SCOPE_KIND_BY_DEFECT.items():
            if normalized in tagged_defects:
                counts[scope_kind] += 1
                break

    ranked = [
        (count, _SCOPE_KIND_PRIORITY[scope_kind], scope_kind)
        for scope_kind, count in counts.items()
        if count > 0
    ]
    if not ranked:
        return ""
    return max(ranked)[2]


def build_structural_profile(
    parent_state: Any,
    *,
    refinement_scope: str,
    mature_idea: str,
    defect_tags: Sequence[str] = (),
) -> Dict[str, Any]:
    scope_text = normalize_term_scan_text(refinement_scope or "")
    component_names = [
        normalize_term_scan_text(component)
        for component in parent_state.components
        if str(component).strip()
    ]
    parent_text = normalize_term_scan_text(_parent_state_text(parent_state))
    if scope_text and any(name and name in scope_text for name in component_names):
        scope_kind = "existing_component"
    elif scope_text and _BROAD_ARCHITECTURE_SCOPE_RE.search(scope_text):
        scope_kind = "broad_architecture"
    elif scope_text and _EXECUTION_PATH_SCOPE_RE.search(scope_text):
        scope_kind = "execution_path"
    elif scope_text:
        scope_kind = "existing_subsystem"
    else:
        defect_scope_kind = _infer_scope_kind_from_defects(defect_tags)
        if defect_scope_kind:
            scope_kind = defect_scope_kind
        elif _MULTI_PATH_SHAPE_RE.search(parent_text):
            scope_kind = "execution_path"
        else:
            scope_kind = "existing_subsystem"
    return {
        "control_centered": parent_centers_control_surface(parent_state),
        "has_multi_path_shape": bool(_MULTI_PATH_SHAPE_RE.search(parent_text)),
        "scope_kind": scope_kind,
        "training_free_like": bool(
            _TRAINING_FREE_RE.search(normalize_term_scan_text(mature_idea or parent_state.method))
        ),
    }


def format_analysis_blob(analysis: List[Any]) -> str:
    return format_analysis_prompt_view(analysis)


def clip_text(value: Any, limit: int = 800) -> str:
    text = "" if value is None else str(value).strip()
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "..."


def clip_metric_score(value: Any, *, lower: float = 0.0, upper: float = 5.0) -> float:
    return max(lower, min(upper, _safe_float_default(value, lower)))


def coerce_integer_metric_score(
    value: Any,
    *,
    lower: int = 0,
    upper: int = 5,
) -> int:
    clipped = clip_metric_score(value, lower=float(lower), upper=float(upper))
    return max(lower, min(upper, int(round(clipped))))


def normalize_score_weights(
    weights: Dict[str, Any],
    fields: Sequence[str],
) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for field_name in fields:
        try:
            cleaned[field_name] = max(0.0, float(weights.get(field_name, 0.0)))
        except (TypeError, ValueError):
            cleaned[field_name] = 0.0

    total = sum(cleaned.values())
    if total <= 1e-9:
        uniform = 1.0 / max(len(fields), 1)
        return {field_name: uniform for field_name in fields}
    return {field_name: value / total for field_name, value in cleaned.items()}


def apply_normalized_score_weights(target: Any, fields: Sequence[str]) -> None:
    normalized = normalize_score_weights(
        {field_name: getattr(target, field_name, 0.0) for field_name in fields},
        fields,
    )
    for field_name, value in normalized.items():
        setattr(target, field_name, value)


def _normalize_root_domains(items: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for item in items:
        code = str(item).strip()
        if not code or code not in ROOT_DOMAIN_CATALOG or code in seen:
            continue
        seen.add(code)
        ordered.append(code)
    return ordered[:2]


def _format_root_domains_for_prompt(domains: Sequence[str]) -> str:
    normalized = _normalize_root_domains(domains)
    if not normalized:
        return "Unspecified"
    return ", ".join(f"{code} ({ROOT_DOMAIN_CATALOG[code]})" for code in normalized)


def _infer_root_domains_heuristically(topic: str, text: str) -> List[str]:
    haystack = f"{topic}\n{text}".lower()
    scores: Dict[str, int] = {code: 0 for code in ROOT_DOMAIN_CATALOG}
    for code, keywords in _ROOT_DOMAIN_FALLBACK_RULES:
        for token in keywords.split():
            if token in haystack:
                scores[code] += 1

    ranked = [
        code
        for code, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if score > 0
    ]
    if not ranked:
        return [DEFAULT_ROOT_DOMAIN]

    chosen = ranked[:2]
    if "cs.LG" in ranked and "cs.LG" not in chosen and len(chosen) < 2:
        chosen.append("cs.LG")
    return _normalize_root_domains(chosen or [DEFAULT_ROOT_DOMAIN])

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


def _extract_component_mapping_keys_from_plan(plan: Any) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for edit in getattr(plan, "component_edits", []) or []:
        for raw_name in (
            getattr(edit, "component", ""),
            getattr(edit, "target", ""),
        ):
            name = _coerce_component_name(raw_name)
            lowered = name.lower()
            if not name or lowered in seen:
                continue
            seen.add(lowered)
            ordered.append(name)
    return ordered


def _filter_component_mapping_to_plan_keys(raw_mapping: Any, plan: Any) -> Dict[str, str]:
    normalized = _normalize_component_mapping(raw_mapping)
    allowed_keys = {
        key.lower()
        for key in _extract_component_mapping_keys_from_plan(plan)
        if str(key).strip()
    }
    if not allowed_keys:
        return {}
    return {
        key: value
        for key, value in normalized.items()
        if key.lower() in allowed_keys
    }


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
            return compact_json(value)
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

def _safe_float_default(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalize_theory_transfer_query_payload(
    payload: Any,
    *,
    fallback: Dict[str, str],
    clip_limit: int,
) -> Dict[str, str]:
    if not isinstance(payload, dict):
        return fallback

    query = clip_text(str(payload.get("query") or "").strip(), clip_limit)
    needed_content = clip_text(str(payload.get("needed_content") or "").strip(), clip_limit)
    expected_role = clip_text(str(payload.get("expected_role") or "").strip(), clip_limit)
    if not query:
        return fallback
    return {
        "query": query,
        "needed_content": needed_content or fallback["needed_content"],
        "expected_role": expected_role or fallback["expected_role"],
    }


def normalize_mechanism_commit_query_payload(
    payload: Any,
    *,
    fallback: Dict[str, str],
    clip_limit: int,
) -> Dict[str, str]:
    if not isinstance(payload, dict):
        return fallback

    query = clip_text(str(payload.get("query") or "").strip(), clip_limit)
    mechanism_gap = clip_text(str(payload.get("mechanism_gap") or "").strip(), clip_limit)
    expected_role = clip_text(str(payload.get("expected_role") or "").strip(), clip_limit)
    if not query:
        return fallback
    return {
        "query": query,
        "mechanism_gap": mechanism_gap or fallback["mechanism_gap"],
        "expected_role": expected_role or fallback["expected_role"],
    }


def format_theory_transfer_references(
    query_payload: Dict[str, str],
    hits: Sequence[Dict[str, Any]],
    *,
    clip_limit: int,
) -> str:
    if not hits:
        return "None"
    lines = [
        f"Transfer need: {query_payload.get('needed_content') or 'Unspecified'}",
        f"Expected role: {query_payload.get('expected_role') or 'Unspecified'}",
        "Cross-domain core references:",
    ]
    for idx, hit in enumerate(hits, start=1):
        core_node = hit.get("core_node") if isinstance(hit.get("core_node"), dict) else {}
        label = (
            core_node.get("full_name")
            or core_node.get("paper_title")
            or hit.get("node_id")
            or f"core_{idx}"
        )
        paper_title = str(core_node.get("paper_title") or "").strip()
        paper_domain = str(core_node.get("paper_domain") or "").strip() or "unknown"
        lines.append(
            f"{idx}. {label} | domain={paper_domain} | score={float(hit.get('score') or 0.0):.3f}"
        )
        if paper_title:
            lines.append(f"   paper: {clip_text(paper_title, clip_limit)}")
        summary = clip_text(str(core_node.get("summary") or "").strip(), clip_limit)
        insight = clip_text(str(core_node.get("insight") or "").strip(), clip_limit)
        if summary:
            lines.append(f"   summary: {summary}")
        if insight:
            lines.append(f"   insight: {insight}")
        matched_components = (
            hit.get("matched_components") if isinstance(hit.get("matched_components"), list) else []
        )
        for matched in matched_components[:2]:
            component_name = clip_text(
                str(matched.get("component_name") or "").strip(),
                clip_limit,
            )
            component_summary = clip_text(
                str(matched.get("component_summary") or "").strip(),
                clip_limit,
            )
            if component_name or component_summary:
                lines.append(
                    f"   matched_component: {component_name or 'unknown'} | {component_summary}"
                )
    lines.append(
        "Use these nodes only as references for transferable mechanisms. Do not copy them verbatim and do not change the root domain(s)."
    )
    return "\n".join(lines)


def format_mechanism_commit_references(
    query_payload: Dict[str, str],
    hits: Sequence[Dict[str, Any]],
    *,
    clip_limit: int,
) -> str:
    if not hits:
        return "None"
    lines = [
        f"Mechanism gap: {query_payload.get('mechanism_gap') or 'Unspecified'}",
        f"Expected role: {query_payload.get('expected_role') or 'Unspecified'}",
        "Retrieved core references for mechanism grounding:",
    ]
    for idx, hit in enumerate(hits, start=1):
        core_node = hit.get("core_node") if isinstance(hit.get("core_node"), dict) else {}
        label = (
            core_node.get("full_name")
            or core_node.get("paper_title")
            or hit.get("node_id")
            or f"core_{idx}"
        )
        paper_title = str(core_node.get("paper_title") or "").strip()
        paper_domain = str(core_node.get("paper_domain") or "").strip() or "unknown"
        lines.append(
            f"{idx}. {label} | domain={paper_domain} | score={float(hit.get('score') or 0.0):.3f}"
        )
        if paper_title:
            lines.append(f"   paper: {clip_text(paper_title, clip_limit)}")
        summary = clip_text(str(core_node.get("summary") or "").strip(), clip_limit)
        insight = clip_text(str(core_node.get("insight") or "").strip(), clip_limit)
        if summary:
            lines.append(f"   summary: {summary}")
        if insight:
            lines.append(f"   insight: {insight}")
        matched_components = (
            hit.get("matched_components") if isinstance(hit.get("matched_components"), list) else []
        )
        for matched in matched_components[:2]:
            component_name = clip_text(
                str(matched.get("component_name") or "").strip(),
                clip_limit,
            )
            component_summary = clip_text(
                str(matched.get("component_summary") or "").strip(),
                clip_limit,
            )
            if component_name or component_summary:
                lines.append(
                    f"   matched_component: {component_name or 'unknown'} | {component_summary}"
                )
    lines.append(
        "Use these nodes to ground the concrete mechanism choice and execution path details."
    )
    lines.append(
        "Use them as grounding patterns, not as modules to copy verbatim."
    )
    return "\n".join(lines)


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
