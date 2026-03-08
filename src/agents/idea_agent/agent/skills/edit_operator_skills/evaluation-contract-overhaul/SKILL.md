---
name: evaluation-contract-overhaul
description: Redesign evaluation contracts to support mechanism diagnosis, while keeping evaluator changes auxiliary to the core system.
---

## defect_tags
- evaluation_blindspot
- weak_accountability
- missing_contracts

## guardrails
- Use protocol redesign to expose mechanism failures, not to replace mechanism work.
- Tie every added check to a specific failure mode.
- Do not let the evaluator become the whole contribution unless the defect is purely contractual.

## atomic_blueprint
- ADD_PROTOCOL(regression,ablation)
- REPLACE_COMPONENT(legacy_evaluator -> contract_evaluator)

## required_protocols
- regression
- ablation

## avoid_combinations
- REMOVE_COMPONENT(contract_evaluator) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
