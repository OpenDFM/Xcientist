---
name: alternative-path-contrast
description: Introduce a structured alternative execution or recovery path so rare regimes and fallback behavior are explicitly contrasted and stress-tested.
---

## defect_tags
- brittle_single_path
- rare_regime_failure
- weak_fallback_behavior

## guardrails
- Limit the number of new paths; one alternative path should address one concrete failure regime.
- Make routing between primary and alternative paths explicit.
- Validate the regimes where the alternative path should dominate or recover failure.

## atomic_blueprint
- ADD_COMPONENT(alternative_path_module)
- REWIRE(alternative_path_module -> execution_router)
- ADD_PROTOCOL(ablation,stress)

## required_protocols
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(execution_router) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
