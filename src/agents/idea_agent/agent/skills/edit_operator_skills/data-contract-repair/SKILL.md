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
- Add a checker only if the contract cannot be enforced with lighter-weight instrumentation.

## atomic_blueprint
- ADD_PROTOCOL(regression,ablation)
- ADD_COMPONENT(contract_checker)

## required_protocols
- regression
- ablation

## avoid_combinations
- ADD_COMPONENT(large_new_backbone) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
