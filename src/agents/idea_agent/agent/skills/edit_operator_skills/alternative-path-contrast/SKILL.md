---
name: alternative-path-contrast
description: Introduce a structured alternative treatment for failure regimes so rare cases and recovery behavior are explicit without turning the method into generic control scaffolding.
---

## defect_tags
- brittle_single_path
- rare_regime_failure
- weak_fallback_behavior

## guardrails
- Limit the number of new paths; one alternative path should address one concrete failure regime.
- State the observable condition that separates the primary regime from the alternative treatment.
- Validate the regimes where the alternative treatment should dominate or recover failure.

## structural_mode
- path_branching

## scope_preference
- execution_path

## requires_control_centered_parent
- false

## atomic_blueprint
- ADD_COMPONENT(alternative_path_module)
- REWIRE(alternative_path_module -> failure_regime_interface)
- ADD_PROTOCOL(ablation,stress)

## required_protocols
- ablation
- stress

## avoid_combinations
- REMOVE_COMPONENT(failure_regime_interface) in the same plan

## execution_logic
1. Identify the defect from `defect_tags`.
2. Generate the solution by **instantiating** the `atomic_blueprint`.
3. Format: `FUNCTION_NAME(ARGUMENTS) -> REASONING`
