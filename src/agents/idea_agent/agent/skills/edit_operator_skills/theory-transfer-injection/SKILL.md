---
name: theory-transfer-injection
description: Inject a theory-backed module from another domain and validate both gains and transfer risks.
---

## defect_tags
- stagnant_novelty
- theory_gap
- weak_generalization

## guardrails
- Name the transferred mechanism source and integration point.
- Gate transfer module activation under reliability criteria.
- Add stress tests for negative transfer.

## atomic_blueprint
- ADD_COMPONENT(theory_transfer_module)
- REWIRE(theory_transfer_module -> training_objective)
- GATE_COMPONENT(theory_transfer_module, when_transfer_signal_reliable)
- ADD_PROTOCOL(regression,ablation,stress)

## required_protocols
- regression
- ablation
- stress

## avoid_combinations
- ADD_COMPONENT(unrelated_parallel_module) in the same plan
