"""Convert ablation experiment results to symbolic memory format."""

import json
import os
from typing import Any, Dict, List, Optional

from memory.memory_system.component_taxonomy import extract_component_families
from memory.api.symbolic_memory_system_api import SymbolicMemorySystem
from memory.api.base_symbolic_memory_system_api import SymbolicRecordPayload

from .config import PipelineConfig, get_default_config


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
    config: Optional[PipelineConfig] = None,
) -> List[Dict[str, Any]]:
    config = config or get_default_config()
    if symbolic_memory_path is None:
        symbolic_memory_path = str(config.symbolic_memory_path)

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
            config.default_macro_roles,
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
    memory.save(symbolic_memory_path)

    return created_records
