---
name: resource-aware-adaptive-path
description: Add resource-aware path selection so execution adapts to workload, budget, or service-level pressure.
---

## defect_tags
- budget_instability
- load_sensitivity
- latency_bottleneck

## guardrails
- Define which resource signals are observed and how path decisions are made.
- Avoid hard-coded thresholds unless they are justified by system constraints.
- Report the trade-off surface across quality, latency, and cost.

## atomic_blueprint
- ADD_COMPONENT(resource_monitor)
- ADD_COMPONENT(adaptive_path_router)
- REWIRE(resource_monitor -> adaptive_path_router)
- ADD_PROTOCOL(ablation,stress)

## required_protocols
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(adaptive_path_router) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
