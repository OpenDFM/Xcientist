---
name: feedback-closed-loop
description: Turn an open-loop process into a monitored feedback loop that measures outcomes and adapts future actions online.
---

## defect_tags
- silent_failure
- drift
- open_loop_fragility

## guardrails
- Define the feedback signal, update cadence, and stability boundaries.
- Keep the loop lightweight relative to the controlled path.
- Validate the loop under drift, delay, or stale-feedback regimes.

## atomic_blueprint
- ADD_COMPONENT(feedback_monitor)
- REWIRE(feedback_monitor -> control_policy)
- ADD_PROTOCOL(ablation,stress)

## required_protocols
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(control_policy) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
