"""Convert ablation experiment results to symbolic memory format."""

import json
import os
import re
from typing import Any, Dict, List, Optional

from memory.api.symbolic_memory_system_api import SymbolicMemorySystem
from memory.api.base_symbolic_memory_system_api import SymbolicRecordPayload

from src.config import load_config


def normalize_component_family(
    component_name: str,
    known_roles: Optional[List[str]] = None,
) -> str:
    if not known_roles:
        config = load_config()
        known_roles = list(config.pipeline.get("default_macro_roles", []))

    # Try to extract macro_role from suffix (e.g., 'flow_matching_generator' -> 'generator')
    parts = component_name.lower().split("_")

    # First, check if any known role is a suffix
    for role in known_roles:
        if parts[-1] == role:
            sub_type = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
            return f"{role}.{sub_type}"

    # Check if any known role is in the parts
    for i, part in enumerate(parts):
        if part in known_roles:
            sub_type = "_".join(parts[:i]) if i > 0 else parts[0]
            return f"{part}.{sub_type}"

    # Fallback: use 'component' as macro_role
    return f"component.{component_name.lower()}"


def compute_delta_score(result: str, value: str, confidence: float) -> float:
    # Parse the value to get a numeric delta
    delta = 0.0
    if result == "positive":
        # Try to extract percentage
        match = re.search(r"([+-]?\d+\.?\d*)%", value)
        if match:
            percent = float(match.group(1))
            delta = min(1.0, percent / 100.0)  # Cap at 1.0
        else:
            delta = 0.3  # Default positive
    elif result == "negative":
        # Try to extract percentage (negative values)
        match = re.search(r"([+-]?\d+\.?\d*)%", value)
        if match:
            percent = float(match.group(1))
            delta = max(-1.0, percent / 100.0)  # Cap at -1.0
        else:
            delta = -0.3  # Default negative
    else:
        delta = 0.0  # Inconclusive

    # Weight by confidence
    return delta * confidence


def infer_main_op(result: str) -> str:
    if result == "positive":
        return "add"  # The component was beneficial
    elif result == "negative":
        return "remove"  # The component was harmful
    else:
        return "tune"  # Needs parameter tuning


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
    config = config or load_config()
    pipeline_cfg = config.pipeline
    if symbolic_memory_path is None:
        symbolic_memory_path = str(pipeline_cfg.get("symbolic_memory_path", "idea_skill_priors"))

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
        # Determine main_op from result
        result_type = comp_result.get("result", "inconclusive")
        main_op = infer_main_op(result_type)

        # Compute delta_score
        value = comp_result.get("value", "0%")
        confidence = comp_result.get("confidence", 0.5)
        delta_score = compute_delta_score(result_type, value, confidence)

        # Normalize component family
        component_family = normalize_component_family(
            comp_id,
            list(pipeline_cfg.get("default_macro_roles", [])),
        )

        # Build the payload
        payload = SymbolicRecordPayload(
            summary=f"Ablation result for {comp_id}: {result_type} ({value})",
            pattern=f"Experiment result: {comp_result.get('metric', 'N/A')} = {value}",
            conditions=[],
            actions=[f"ablation_{result_type}"],
            rationale=(
                comp_result.get("analysis", "")
                + (
                    f"\n\nMethod context: {comp_result.get('method_context', '')}"
                    if comp_result.get("method_context")
                    else ""
                )
            ),
            expected_outcomes=[],
            anti_patterns=[] if result_type == "positive" else [comp_id],
            tags=["ablation", "experiment_result", comp_id, result_type],
            priority=confidence,
            confidence=confidence,
            source="experiment",
            support_count=1,
            metadata={
                "idea_id": experiment_id,
                "experiment_id": experiment_id,
                "metric": comp_result.get("metric"),
                "value": value,
                "component_id": comp_id,
                "method_context": comp_result.get("method_context"),
                "supporting_diagnostics": comp_result.get("supporting_diagnostics"),
            },
            component_family=component_family,
            family_pair="",
            main_op=main_op,
            context_signature={},
            delta_score=delta_score,
        )

        # Create symbolic record
        record = memory.instantiate_symbolic_record(**payload.model_dump())
        memory.add([record])
        created_records.append(record.to_dict())

    # Add summary as well
    summary = ablation_data.get("summary", {})
    if summary:
        payload = SymbolicRecordPayload(
            summary=f"Experiment summary: feasible={summary.get('feasible', False)}, confidence={summary.get('confidence', 0)}",
            pattern="Overall experiment evaluation",
            conditions=[],
            actions=["experiment_complete"],
            rationale="; ".join(summary.get("key_findings", [])),
            expected_outcomes=[],
            anti_patterns=[],
            tags=["experiment_summary", "experiment_complete"],
            priority=summary.get("confidence", 0.5),
            confidence=summary.get("confidence", 0.5),
            source="experiment",
            support_count=1,
            metadata={
                "idea_id": experiment_id,
                "experiment_id": experiment_id,
                "feasible": summary.get("feasible", False),
            },
            component_family="experiment.summary",
            family_pair="",
            main_op="evaluate",
            context_signature={},
            delta_score=0.5 if summary.get("feasible", False) else -0.5,
        )
        record = memory.instantiate_symbolic_record(**payload.model_dump())
        memory.add([record])
        created_records.append(record.to_dict())

    # Save the updated memory
    memory.save(symbolic_memory_path)

    return created_records
