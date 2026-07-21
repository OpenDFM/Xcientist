---
name: hierarchical-decomposition
description: Replace a flat control or execution flow with a hierarchy that assigns responsibilities across explicit levels.
---

## defect_tags
- responsibility_entanglement
- monolithic_design
- scale_mismatch

## guardrails
- Separate responsibilities cleanly across levels.
- Avoid duplicating the same control logic in every layer.
- Show that the hierarchy improves control, analysis, or scaling rather than adding ceremony.

## structural_mode
- hierarchical_reorg

## scope_preference
- broad_architecture

## requires_control_centered_parent
- false

## atomic_blueprint
- REPLACE_COMPONENT(flat_pipeline -> hierarchical_pipeline)
- REWIRE(hierarchical_pipeline -> execution_path)
- ADD_PROTOCOL(ablation)

## required_protocols
- ablation

## avoid_combinations
- ADD_COMPONENT(parallel_shadow_pipeline) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
