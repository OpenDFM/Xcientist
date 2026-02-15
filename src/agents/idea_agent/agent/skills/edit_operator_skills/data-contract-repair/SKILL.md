---
name: data-contract-repair
description: Repair data/supervision contract first, then add minimal components only when protocol evidence requires it.
---

## defect_tags
- data_quality
- label_noise
- missing_contracts

## guardrails
- Lead with protocol repair before model growth.
- Keep contract checks measurable and reproducible.
- Gate any added checker to avoid unnecessary cost.

## atomic_blueprint
- ADD_PROTOCOL(regression,ablation,stress)
- ADD_COMPONENT(contract_checker)
- GATE_COMPONENT(contract_checker, when_contract_violation_detected)

## required_protocols
- regression
- ablation
- stress

## avoid_combinations
- ADD_COMPONENT(large_new_backbone) in the same plan
