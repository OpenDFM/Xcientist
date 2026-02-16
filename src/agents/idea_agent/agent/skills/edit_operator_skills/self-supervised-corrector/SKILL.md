---
name: self-supervised-corrector
description: Attach a self-supervised correction component, inject residual corrections, and control over-correction with gates.
---

## defect_tags
- systematic_bias
- silent_failure
- drift

## guardrails
- Define correction signal and injection point.
- Include anti-overcorrection conditions in gating logic.
- Stress-test under known drift regimes.

## atomic_blueprint
- ADD_COMPONENT(self_supervised_corrector)
- REWIRE(self_supervised_corrector -> prediction_head)
- GATE_COMPONENT(self_supervised_corrector, when_confidence_gap_detected)
- ADD_PROTOCOL(regression,ablation,stress)

## required_protocols
- regression
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(prediction_head) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`