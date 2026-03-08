---
name: multi-scale-coordinator
description: Add a coordinator that aligns decisions across scales, layers, or service tiers without making coordination overhead the main contribution.
---

## defect_tags
- scale_mismatch
- coordination_failure
- latency_bottleneck

## guardrails
- Specify how routing conflicts are resolved across scales or layers.
- Quantify the coordination overhead in latency or compute terms.
- Keep validation centered on whether coordination improves the core task path.

## atomic_blueprint
- ADD_COMPONENT(scale_coordinator)
- REWIRE(scale_coordinator -> routing_layer)
- ADD_PROTOCOL(ablation)

## required_protocols
- ablation

## avoid_combinations
- REMOVE_COMPONENT(routing_layer) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
