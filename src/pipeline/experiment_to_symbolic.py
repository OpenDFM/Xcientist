"""Convert ablation experiment results to symbolic memory format."""

import json
import os
from typing import Any, Dict, List, Optional

from src.memory.api.base_symbolic_memory_system_api import SymbolicRecordPayload
from src.memory.api.symbolic_memory_system_api import SymbolicMemorySystem
from src.memory.memory_system.component_taxonomy import extract_component_families


def _default_macro_roles(config: Any) -> List[str]:
    value = getattr(config, "default_macro_roles", None)
    if value is None:
        pipeline_cfg = getattr(config, "pipeline", None)
        if pipeline_cfg is not None:
            if hasattr(pipeline_cfg, "get"):
                value = pipeline_cfg.get("default_macro_roles")
            else:
                value = getattr(pipeline_cfg, "default_macro_roles", None)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _config_lookup(config: Any, path: str, default: Any = None) -> Any:
    current = config
    for part in str(path or "").split("."):
        if current is None:
            return default
        if hasattr(current, "get"):
            try:
                current = current.get(part)
            except Exception:
                current = None
        else:
            current = getattr(current, part, None)
    return default if current is None else current


def _configured_symbolic_memory_path(config: Any) -> str:
    for path in (
        "pipeline.symbolic_memory_path",
        "flow.memory.symbolic_memory_path",
        "symbolic_memory_path",
    ):
        value = str(_config_lookup(config, path, "") or "").strip()
        if value:
            return value
    return "idea_skill_priors"


def normalize_component_family(
    component_name: str,
    known_roles: Optional[List[str]] = None,
    method_context: str = "",
) -> str:
    del known_roles
    families = extract_component_families([component_name], method_context or "")
    if families:
        family = str(families[0].get("family") or "").strip()
        if family:
            return family
    fallback = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(component_name or "").strip())
    fallback = "_".join(part for part in fallback.split("_") if part)
    return f"component.{fallback or 'unknown'}"

def load_ablation_results(ablation_path: str) -> Dict[str, Any]:
    with open(ablation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_ablation_to_symbolic_memory(
    ablation_path: str,
    idea_components: Optional[List[Dict[str, Any]]] = None,
    experiment_id: str = "",
    symbolic_memory_path: Optional[str] = None,
    config: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    from src.config import load_config

    config = config or load_config()
    default_roles = _default_macro_roles(config)
    if symbolic_memory_path is None:
        symbolic_memory_path = _configured_symbolic_memory_path(config)

    # Load ablation results
    ablation_data = load_ablation_results(ablation_path)

    # Initialize symbolic memory system
    memory = SymbolicMemorySystem()
    if os.path.exists(os.path.join(symbolic_memory_path, "symbolic_memory.json")):
        memory.load(symbolic_memory_path)

    # Create records for each component
    created_records = []
    components_data = ablation_data.get("components", {})
    for comp_id, comp_result in components_data.items():
        if not isinstance(comp_result, dict):
            continue
        result_type = comp_result.get("result", "inconclusive")

        value = comp_result.get("value", "0%")
        confidence = comp_result.get("confidence", 0.5)
        method_context = str(comp_result.get("method_context", "") or "")

        # Normalize component family
        component_family = normalize_component_family(
            comp_id,
            default_roles,
            method_context=method_context,
        )

        # Build the payload
        payload = SymbolicRecordPayload(
            component=comp_id,
            component_family=component_family,
            result=str(result_type or "inconclusive"),
            metric=str(comp_result.get("metric", "") or ""),
            value=str(value or ""),
            analysis=str(comp_result.get("analysis", "") or ""),
            method_context=method_context,
            confidence=float(confidence),
        )

        # Create symbolic record
        record = memory.instantiate_symbolic_record(**payload.model_dump())
        memory.upsert_normal_records([record])
        created_records.append(record.to_dict())

    # Save the updated memory
    if not memory.save(symbolic_memory_path):
        raise RuntimeError(f"failed to save symbolic memory at `{symbolic_memory_path}`")

    return created_records
