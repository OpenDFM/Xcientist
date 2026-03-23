---
name: feedback-closed-loop
description: Turn an open-loop process into a feedback-guided update mechanism that measures outcomes and adapts an existing rule online.
---

## defect_tags
- silent_failure
- drift
- open_loop_fragility

## guardrails
- Define the feedback signal, update cadence, and stability boundaries.
- Keep the adaptive update lightweight relative to the controlled path.
- Validate the loop under drift, delay, or stale-feedback regimes.

## atomic_blueprint
- ADD_COMPONENT(feedback_monitor)
- REWIRE(feedback_monitor -> adaptation_rule)
- ADD_PROTOCOL(ablation,stress)

## required_protocols
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(adaptation_rule) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
