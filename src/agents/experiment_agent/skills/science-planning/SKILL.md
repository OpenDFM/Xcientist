---
name: science-planning
description: Build experiment task batches from validated prepare/code evidence
license: MIT
---

# Science Planning

## Mission
Translate validated handoff artifacts into explicit experiment batches that close science gaps.

## Protocol
- Write the lane-specific science plan with target, output path, and dependency metadata.
- Required conditions are `baseline`, `full_method`, and `ablation:<component>` for every required component.
- For ablation, derive components only from `idea.json.components`, preserve their order exactly, and carry their explanations into the step metadata.
- Mark each task with whether it is `smoke/debug` or `final`; only `final` + `dataset_scope=full` counts toward completion.
- The plan must bind each task to the validated prepare target model/API/data path instead of inventing a smaller substitute target.
- Only place tasks in the same parallel batch when they use disjoint output dirs and do not overwrite the same result files.
- Keep the lane plan, step contracts, executor reports, worker reports, validator reports, and human-readable lane summaries under `agent_reports/` using flat filenames.
- Update the lane summary under `agent_reports/standard_science_summary.md` or `agent_reports/ablation_science_summary.md` only as a human-readable summary of validator-backed evidence.
- For ablation, require step-level reports to preserve structured component verdicts and `method_context` so a later ablation report integrator can write `ablation_results.json`.

## Hard Rules
- Every required benchmark must include full-system performance.
- Every required component must have an ablation result with `method_context`.
- Ablation component names and order must exactly match `idea.json.components`.
- Preflight tasks are allowed, but they do not satisfy any final benchmark condition.
