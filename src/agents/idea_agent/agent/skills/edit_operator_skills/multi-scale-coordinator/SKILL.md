---
name: multi-scale-coordinator
description: Add a cross-scale coordinator and rewire routing without making gating the main idea.
---

## defect_tags
- scale_mismatch
- coordination_failure
- latency_bottleneck

## guardrails
- Specify routing conflict resolution.
- Quantify additional latency and compute.
- Keep validation centered on whether coordination improves the core mechanism.

## atomic_blueprint
- ADD_COMPONENT(multi_scale_coordinator)
- REWIRE(multi_scale_coordinator -> prediction_router)
- ADD_PROTOCOL(ablation)

## required_protocols
- ablation

## avoid_combinations
- REMOVE_COMPONENT(prediction_router) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
