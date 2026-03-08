---
name: self-supervised-corrector
description: Attach a self-supervised correction component and inject residual corrections without over-privileging gating logic.
---

## defect_tags
- systematic_bias
- silent_failure
- drift

## guardrails
- Define correction signal and injection point.
- Keep anti-overcorrection controls lightweight and subordinate to the correction mechanism.
- Validate the correction mechanism under known drift regimes.

## atomic_blueprint
- ADD_COMPONENT(self_supervised_corrector)
- REWIRE(self_supervised_corrector -> prediction_head)
- ADD_PROTOCOL(ablation)

## required_protocols
- ablation

## avoid_combinations
- REMOVE_COMPONENT(prediction_head) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
