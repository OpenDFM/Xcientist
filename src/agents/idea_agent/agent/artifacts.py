from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


ARTIFACT_NAMESPACES = (
    "run",
    "retrieval",
    "analysis",
    "ideation",
    "persistence",
)


FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    "topic": {"namespace": "run", "path": "topic", "kind": "list"},
    "run_topic": {"namespace": "run", "path": "run_topic", "kind": "str"},
    "mature_idea": {"namespace": "run", "path": "mature_idea", "kind": "str"},
    "dialogue": {"namespace": "run", "path": "dialogue", "kind": "dict"},
    "steps": {"namespace": "run", "path": "steps", "kind": "list"},
    "workflow_trace": {"namespace": "run", "path": "workflow_trace", "kind": "list"},
    "workflow_state": {"namespace": "run", "path": "workflow_state", "kind": "dict"},
    "context_slots": {"namespace": "run", "path": "context_slots", "kind": "dict"},
    "operation_trace": {"namespace": "run", "path": "operation_trace", "kind": "list"},
    "idea_taste_mode": {"namespace": "run", "path": "idea_taste_mode", "kind": "str"},
    "idea_taste_label": {"namespace": "run", "path": "idea_taste_label", "kind": "str"},
    "survey": {"namespace": "retrieval", "path": "survey", "kind": "str"},
    "retrieval_keywords": {"namespace": "retrieval", "path": "retrieval_keywords", "kind": "list"},
    "references": {"namespace": "retrieval", "path": "references", "kind": "list"},
    "rag_query": {"namespace": "retrieval", "path": "rag_query", "kind": "list"},
    "rag_hits": {"namespace": "retrieval", "path": "rag_hits", "kind": "list"},
    "rag_contents": {"namespace": "retrieval", "path": "rag_contents", "kind": "list"},
    "analysis": {"namespace": "analysis", "path": "entries", "kind": "list"},
    "root_idea": {"namespace": "analysis", "path": "root_idea", "kind": "dict"},
    "background_knowledge": {"namespace": "analysis", "path": "background_knowledge", "kind": "list"},
    "component_decisions": {"namespace": "analysis", "path": "component_decisions", "kind": "list"},
    "latest_candidate": {"namespace": "ideation", "path": "latest_candidate", "kind": "dict"},
    "evaluations": {"namespace": "ideation", "path": "evaluations", "kind": "list"},
    "ligagent_pro_candidates": {"namespace": "ideation", "path": "ligagent_pro_candidates", "kind": "list"},
    "fusion_result": {"namespace": "ideation", "path": "fusion_result", "kind": "dict"},
    "ablation_results": {"namespace": "ideation", "path": "ablation_results", "kind": "list"},
    "ablation_results_raw": {"namespace": "ideation", "path": "ablation_results_raw", "kind": "dict"},
    "ltm_experiences": {"namespace": "ideation", "path": "ltm_experiences", "kind": "list"},
    "idea_result": {"namespace": "persistence", "path": "idea_result", "kind": "dict"},
    "artifact_structure": {"namespace": "persistence", "path": "artifact_structure", "kind": "dict"},
}


ARTIFACT_FORMAT = {
    "run": dict,
    "retrieval": dict,
    "analysis": dict,
    "ideation": dict,
    "persistence": dict,
}


class ArtifactContainer(dict):
    """Restrict top-level storage to namespaces only."""

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in ARTIFACT_NAMESPACES:
            raise KeyError(
                f"Top-level artifact writes must target a namespace, got '{key}'."
            )
        super().__setitem__(key, value)

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in ARTIFACT_NAMESPACES:
            raise KeyError(
                f"Top-level artifact writes must target a namespace, got '{key}'."
            )
        return super().setdefault(key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        incoming = dict(*args, **kwargs)
        invalid = [key for key in incoming if key not in ARTIFACT_NAMESPACES]
        if invalid:
            raise KeyError(
                "Top-level artifact writes must target namespaces only: "
                + ", ".join(sorted(invalid))
            )
        super().update(incoming)


def _namespace_defaults() -> Dict[str, Dict[str, Any]]:
    return {
        "run": {
            "run_topic": "",
            "topic": [],
            "mature_idea": "",
            "dialogue": {},
            "steps": [],
            "workflow_trace": [],
            "workflow_state": {},
            "context_slots": {},
            "operation_trace": [],
            "idea_taste_mode": "",
            "idea_taste_label": "",
        },
        "retrieval": {
            "survey": "",
            "retrieval_keywords": [],
            "references": [],
            "rag_query": [],
            "rag_hits": [],
            "rag_contents": [],
        },
        "analysis": {
            "entries": [],
            "root_idea": {},
            "background_knowledge": [],
            "component_decisions": [],
        },
        "ideation": {
            "latest_candidate": {},
            "evaluations": [],
            "ligagent_pro_candidates": [],
            "fusion_result": {},
            "ablation_results": [],
            "ablation_results_raw": {},
            "ltm_experiences": [],
        },
        "persistence": {
            "idea_result": {},
            "artifact_structure": {},
            "schema_version": 4,
        },
    }


def _default_for_kind(kind: str) -> Any:
    if kind == "list":
        return []
    if kind == "dict":
        return {}
    if kind == "str":
        return ""
    raise ValueError(f"Unsupported artifact kind: {kind}")


def _matches_kind(value: Any, kind: str) -> bool:
    if kind == "list":
        return isinstance(value, list)
    if kind == "dict":
        return isinstance(value, dict)
    if kind == "str":
        return isinstance(value, str)
    return False


def artifact_namespace_for(key: str) -> str:
    spec = FIELD_SPECS.get(key)
    if not spec:
        raise KeyError(f"Unknown artifact field: {key}")
    return str(spec["namespace"])


def artifact_kind_for(key: str) -> str:
    spec = FIELD_SPECS.get(key)
    if not spec:
        raise KeyError(f"Unknown artifact field: {key}")
    return str(spec["kind"])


def _field_path(key: str) -> str:
    spec = FIELD_SPECS.get(key)
    if not spec:
        raise KeyError(f"Unknown artifact field: {key}")
    return str(spec["path"])


def artifact_namespace(artifact: Dict[str, Any], namespace: str) -> Dict[str, Any]:
    ensure_artifact_structure(artifact)
    if namespace not in ARTIFACT_NAMESPACES:
        raise KeyError(f"Unknown artifact namespace: {namespace}")
    payload = artifact.get(namespace)
    if not isinstance(payload, dict):
        raise TypeError(f"Artifact namespace '{namespace}' is not a dict.")
    return payload


def ensure_artifact_structure(artifact: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(artifact)
    normalized = _namespace_defaults()

    for key, spec in FIELD_SPECS.items():
        namespace = str(spec["namespace"])
        path = str(spec["path"])
        kind = str(spec["kind"])

        existing_namespace = raw.get(namespace)
        namespaced_value = (
            existing_namespace.get(path)
            if isinstance(existing_namespace, dict)
            else None
        )
        if _matches_kind(namespaced_value, kind):
            value = namespaced_value
        else:
            legacy_value = raw.get(key)
            if key == "latest_candidate" and not _matches_kind(legacy_value, kind):
                legacy_pool = raw.get("idea_pool")
                if isinstance(existing_namespace, dict) and isinstance(existing_namespace.get("idea_pool"), list):
                    legacy_pool = existing_namespace.get("idea_pool")
                if isinstance(legacy_pool, list) and legacy_pool:
                    last_entry = legacy_pool[-1]
                    if isinstance(last_entry, dict):
                        legacy_value = last_entry
            if _matches_kind(legacy_value, kind):
                value = legacy_value
            else:
                value = deepcopy(normalized[namespace].get(path, _default_for_kind(kind)))
        normalized[namespace][path] = value

    normalized["persistence"]["schema_version"] = 4
    artifact.clear()
    for namespace in ARTIFACT_NAMESPACES:
        artifact[namespace] = normalized[namespace]
    return artifact


def artifact_init() -> Dict[str, Any]:
    return ensure_artifact_structure(ArtifactContainer())


def artifact_get(artifact: Dict[str, Any], key: str, default: Optional[Any] = None) -> Any:
    ensure_artifact_structure(artifact)
    if key in FIELD_SPECS:
        namespace = artifact_namespace_for(key)
        path = _field_path(key)
        return artifact[namespace].get(path, default)
    if key in ARTIFACT_NAMESPACES:
        return artifact.get(key, default)
    return artifact.get(key, default)


def _validate_value(key: str, value: Any, expected_kind: str) -> None:
    if not _matches_kind(value, expected_kind):
        raise TypeError(
            f"Artifact field '{key}' expects {expected_kind}, got {type(value).__name__}."
        )


def artifact_set(artifact: Dict[str, Any], key: str, value: Any) -> None:
    ensure_artifact_structure(artifact)
    _validate_value(key, value, artifact_kind_for(key))
    namespace = artifact_namespace_for(key)
    path = _field_path(key)
    artifact[namespace][path] = value


def artifact_append(artifact: Dict[str, Any], key: str, items: List[Any]) -> None:
    ensure_artifact_structure(artifact)
    if artifact_kind_for(key) != "list":
        raise TypeError(f"Artifact field '{key}' is not appendable.")
    if not isinstance(items, list):
        raise TypeError(f"Artifact append on '{key}' expects a list payload.")
    target = artifact_get(artifact, key)
    if not isinstance(target, list):
        target = []
        namespace = artifact_namespace_for(key)
        path = _field_path(key)
        artifact[namespace][path] = target
    target.extend(items)


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    for child_key, child_value in incoming.items():
        if isinstance(base.get(child_key), dict) and isinstance(child_value, dict):
            _deep_merge(base[child_key], deepcopy(child_value))
            continue
        base[child_key] = deepcopy(child_value)
    return base


def artifact_merge(artifact: Dict[str, Any], key: str, incoming: Dict[str, Any]) -> None:
    ensure_artifact_structure(artifact)
    if artifact_kind_for(key) != "dict":
        raise TypeError(f"Artifact field '{key}' is not mergeable.")
    if not isinstance(incoming, dict):
        raise TypeError(f"Artifact merge on '{key}' expects a dict payload.")
    target = artifact_get(artifact, key)
    if not isinstance(target, dict):
        target = {}
        namespace = artifact_namespace_for(key)
        path = _field_path(key)
        artifact[namespace][path] = target
    _deep_merge(target, incoming)
