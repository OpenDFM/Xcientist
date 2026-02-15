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
- Add a stress protocol that targets boundary cases.

## atomic_blueprint
- ADD_COMPONENT(counterfactual_sampler)
- REWIRE(counterfactual_sampler -> training_pipeline)
- ADD_PROTOCOL(regression,ablation,stress)

## required_protocols
- regression
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(training_pipeline) in the same plan
