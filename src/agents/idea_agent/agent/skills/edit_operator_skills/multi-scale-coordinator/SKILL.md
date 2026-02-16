---
name: multi-scale-coordinator
description: Add a cross-scale coordinator, rewire routing, and gate expensive coordination paths.
---

## defect_tags
- scale_mismatch
- coordination_failure
- latency_bottleneck

## guardrails
- Specify routing conflict resolution.
- Quantify additional latency and compute.
- Add stress tests for routing collapse scenarios.

## atomic_blueprint
- ADD_COMPONENT(multi_scale_coordinator)
- REWIRE(multi_scale_coordinator -> prediction_router)
- GATE_COMPONENT(multi_scale_coordinator, when_multi_scale_signal_strong)
- ADD_PROTOCOL(regression,ablation,stress)

## required_protocols
- regression
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(prediction_router) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`