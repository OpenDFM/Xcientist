---
name: theory-transfer-injection
description: Inject a theory-backed mechanism or invariant from another domain and wire it into the core objective or mechanism rule.
---

## defect_tags
- stagnant_novelty
- theory_gap
- weak_generalization

## guardrails
- Name the transferred principle and where it enters the core path.
- Limit transfer to one main mechanism instead of a bundle of unrelated ideas.
- Add the minimum stress coverage needed to detect negative transfer.

## atomic_blueprint
- ADD_COMPONENT(theory_transfer_module)
- REWIRE(theory_transfer_module -> core_objective)
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
