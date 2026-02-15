---
name: evaluation-contract-overhaul
description: Redesign evaluation contracts as first-class edits, then minimally adjust components to satisfy new checks.
---

## defect_tags
- evaluation_blindspot
- weak_accountability
- missing_contracts

## guardrails
- Make protocol redesign the primary action.
- Tie every added check to a specific failure mode.
- Gate heavy evaluators to keep budget under control.

## atomic_blueprint
- ADD_PROTOCOL(regression,ablation,stress)
- REPLACE_COMPONENT(legacy_evaluator -> contract_evaluator)
- GATE_COMPONENT(contract_evaluator, when_budget_allows)

## required_protocols
- regression
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(contract_evaluator) in the same plan
