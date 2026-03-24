---
name: mechanism-commit-innovation
description: Commit to one concrete mechanism-level innovation by strengthening one weak existing component and keep scaffolding secondary to the task-solving path.
---

## defect_tags
- stagnant_novelty
- unclear_mechanism
- validation_gap

## guardrails
- Name the exact mechanism being introduced and the existing component it strengthens or replaces.
- Keep one main mechanism; move supporting implementation details to notes or protocols.
- Prefer direct replacement of the weak internal block over adding a broader coordination structure.
- Use the smallest validation suite that can falsify the core mechanism.

## atomic_blueprint
- REPLACE_COMPONENT(weak_internal_component -> refined_internal_component)
- ADD_PROTOCOL(ablation)

## required_protocols
- ablation

## avoid_combinations
- ADD_COMPONENT(refined_internal_component) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
