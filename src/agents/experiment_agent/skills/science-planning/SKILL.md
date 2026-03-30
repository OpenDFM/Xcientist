---
name: science-planning
description: Build experiment task batches from validated prepare/code evidence
license: MIT
---

# Science Planning

## Mission
Translate validated handoff artifacts into explicit experiment batches that close science gaps with standardized baseline vs full method comparison.

## Key Principles
- **No hardcoding**: Read dataset, model, and API bindings from validated prepare handoff. Do NOT invent or hardcode values.
- **Standardized experiments**: Plan for `baseline` vs `full_method` comparison using the same dataset.
- **Real data only**: All experiments must use `dataset_candidate/` data.

## Protocol
- Write the lane-specific science plan with target, output path, and dependency metadata.
- Required conditions are `baseline`, `full_method`, and `ablation:<component>` for every required component.
- For ablation, derive components only from `idea.json.components`, preserve their order exactly, and carry their explanations into the step metadata.
- Mark each task with whether it is `smoke/debug` or `final`; only `final` + `dataset_scope=full` counts toward completion.
- The plan must bind each task to the validated prepare target model/API/data path instead of inventing a smaller substitute target.
- **CRITICAL**: Benchmark experiments MUST use the real data files from `dataset_candidate/` directory, NOT synthetic or randomly generated data.
- Only place tasks in the same parallel batch when they use disjoint output dirs and do not overwrite the same result files.
- Keep the lane plan, step contracts, executor reports, worker reports, validator reports, and human-readable lane summaries under `agent_reports/` using flat filenames.
- Update the lane summary under `agent_reports/standard_science_summary.md` or `agent_reports/ablation_science_summary.md` only as a human-readable summary of validator-backed evidence.
- For ablation, require step-level reports to preserve structured component verdicts and `method_context` so the later final-artifact materialization step can write `ablation_results.json`.

## Standard Science Requirements
- Plan for `baseline` (standard/original) and `full_method` (all components) conditions
- Both conditions must use the same dataset from `dataset_candidate/`
- Plan must produce comparable metrics between baseline and full method

## Hard Rules
- Every required benchmark must include full-system performance.
- Every required component must have an ablation result with `method_context`.
- Ablation component names and order must exactly match `idea.json.components`.
- Preflight tasks are allowed, but they do not satisfy any final benchmark condition.
- Do not hardcode dataset names, model IDs, or API keys in the plan.
