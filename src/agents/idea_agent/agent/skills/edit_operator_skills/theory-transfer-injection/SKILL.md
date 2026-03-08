---
name: theory-transfer-injection
description: Inject a theory-backed module from another domain while keeping validation and gating secondary to the transferred mechanism.
---

## defect_tags
- stagnant_novelty
- theory_gap
- weak_generalization

## guardrails
- Name the transferred mechanism source and integration point.
- Use gating only if transfer reliability is itself the bottleneck.
- Add the minimum validation needed to test transfer value and negative transfer risk.

## atomic_blueprint
- ADD_COMPONENT(theory_transfer_module)
- REWIRE(theory_transfer_module -> training_objective)
- ADD_PROTOCOL(ablation,stress)

## required_protocols
- ablation
- stress

## avoid_combinations
- ADD_COMPONENT(unrelated_parallel_module) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
