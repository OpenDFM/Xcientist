---
name: counterfactual-contrast
description: Inject counterfactual data components and rewire train/eval flow so edge-case behavior is explicitly learned and tested.
---

## defect_tags
- missing_edge_cases
- weak_generalization
- dataset_bias

## guardrails
- Limit synthetic channels to avoid unbounded complexity.
- Rewire training and evaluation so the counterfactual signal is first-class.
- Keep validation focused on proving the counterfactual channel matters.

## atomic_blueprint
- ADD_COMPONENT(counterfactual_sampler)
- REWIRE(counterfactual_sampler -> training_pipeline)
- ADD_PROTOCOL(ablation)

## required_protocols
- ablation

## avoid_combinations
- REMOVE_COMPONENT(training_pipeline) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
