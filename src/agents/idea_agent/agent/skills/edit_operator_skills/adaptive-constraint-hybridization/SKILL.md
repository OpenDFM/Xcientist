---
name: adaptive-constraint-hybridization
description: Add adaptive constraint modules, fuse them into the objective, and gate by confidence or budget.
---

## defect_tags
- constraint_drift
- physical_invalidity
- weak_regularization

## guardrails
- Keep the new constraint measurable and tied to one objective path.
- Use gating for low-confidence regimes.
- Require ablations separating constraint effect from baseline behavior.

## atomic_blueprint
- ADD_COMPONENT(constraint_penalty_module)
- REWIRE(constraint_penalty_module -> objective)
- GATE_COMPONENT(constraint_penalty_module, when_constraint_confident)
- ADD_PROTOCOL(regression,ablation,stress)

## required_protocols
- regression
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(objective) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`