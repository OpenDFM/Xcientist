---
name: mechanism-commit-innovation
description: Commit to one concrete mechanism-level innovation and keep scaffolding secondary to the task-solving path.
---

## defect_tags
- stagnant_novelty
- unclear_mechanism
- validation_gap

## guardrails
- Name the exact mechanism being introduced and the primary execution path it changes.
- Keep one main mechanism; move supporting implementation details to notes or protocols.
- Use the smallest validation suite that can falsify the core mechanism.

## atomic_blueprint
- ADD_COMPONENT(core_mechanism_module)
- REWIRE(core_mechanism_module -> primary_execution_path)
- ADD_PROTOCOL(ablation)

## required_protocols
- ablation

## avoid_combinations
- REMOVE_COMPONENT(core_mechanism_module) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
