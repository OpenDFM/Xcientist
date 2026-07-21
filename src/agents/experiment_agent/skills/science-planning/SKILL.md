---
name: science-planning
description: Build unified science condition plans from validated prepare/code evidence
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Science Planning

## Mission
Translate validated handoff artifacts into one ordered `science` condition plan. A condition is fully defined by `enabled_components` and `disabled_components`; the all-components condition is the reference case where `disabled_components` is empty.

## Key Principles
- **No hardcoding**: Read dataset, model, and API bindings from validated prepare handoff. Do not invent or hardcode values.
- **Unified science**: Use one `science.plan` under `agent_reports/science/plan/latest.json`.
- **Real data only**: All experiments must use `dataset_candidate/` data.

## Protocol
- Plan all raw outputs under `results/science/<condition_id>/`.
- Plan exactly `1 + len(idea.json.components)` conditions.
- The first condition must be the only all-components reference: all canonical `idea.json.components` enabled, `disabled_components: []`, and no `reference_condition_id`.
- Every later condition must disable exactly one canonical component and reference the first all-components condition.
- Across component-disabled conditions, every canonical idea component must be disabled exactly once, with no omissions and no repeats.
- Every condition must declare dataset bindings, metrics, exact command, output directory, raw evidence paths, and component set.
- Only place conditions in parallel when they use disjoint output dirs and cannot overwrite each other.
- Keep plan, worker reports, reviewer reports, hook reports, evidence manifests, and summaries under `agent_reports/science/`.
- Preserve reviewer fields `result`, `metric`, `value`, `confidence`, `analysis`, and `method_context` for disabled-component conditions so final materialization can write `ablation_results.json`.

## Hard Rules
- `enabled_components` plus `disabled_components` must cover `idea.json.components` exactly for every condition.
- Component names must match `idea.json.components` exactly.
- Commands must run from the workspace root and should call project entrypoints as `python project/...`.
- Preflight, smoke, debug, subset, import-only, or dry-run evidence does not satisfy final science completion.
- Do not write final `ablation_results.json`; it is runtime-owned.
