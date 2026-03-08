---
name: surgical-modularity
description: Perform a localized block-level intervention with replace/rewire and prove gain with focused ablation.
---

## defect_tags
- feature_dumping
- monolithic_design
- harder_to_ablate

## guardrails
- Touch only one weak block.
- Keep interfaces explicit after rewiring.
- Include mandatory block ablation.

## atomic_blueprint
- REPLACE_COMPONENT(weak_block -> modular_block)
- REWIRE(modular_block -> downstream_interface)
- ADD_PROTOCOL(ablation)

## required_protocols
- ablation

## avoid_combinations
- ADD_COMPONENT(extra_auxiliary_stack) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
