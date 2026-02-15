---
name: mechanism-commit-innovation
description: Commit to a concrete mechanism-level innovation and compile it into component edits with budget gates and full validation protocols.
---

## defect_tags
- stagnant_novelty
- unclear_mechanism
- validation_gap

## guardrails
- Name exactly which component is introduced or rewired and link it to a defect.
- Keep one main mechanism and move extra implementation details to notes.
- Always add regression, ablation, and stress protocols.

## atomic_blueprint
- ADD_COMPONENT(core_mechanism_module)
- REWIRE(backbone_model -> core_mechanism_module)
- GATE_COMPONENT(core_mechanism_module, if_compute_budget_allows)
- ADD_PROTOCOL(regression,ablation,stress)

## required_protocols
- regression
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(core_mechanism_module) in the same plan
