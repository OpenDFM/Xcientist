---
name: multi-scale-coordinator
description: Add a cross-scale consistency mechanism that aligns representations or decisions across scales without making coordination overhead the main contribution.
---

## defect_tags
- scale_mismatch
- coordination_failure
- latency_bottleneck

## guardrails
- Specify the cross-scale consistency rule and what information each scale contributes.
- Quantify the coordination overhead in latency or compute terms.
- Keep validation centered on whether cross-scale consistency improves the core task path.

## structural_mode
- cross_scale_coordination

## scope_preference
- broad_architecture

## requires_control_centered_parent
- false

## atomic_blueprint
- ADD_COMPONENT(scale_consistency_module)
- REWIRE(scale_consistency_module -> scale_interface)
- ADD_PROTOCOL(ablation)

## required_protocols
- ablation

## avoid_combinations
- REMOVE_COMPONENT(scale_interface) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
