---
name: convergence-gate
description: Hard convergence rules for master orchestration
license: MIT
---

# Master Convergence

## Mission
Help the master agent decide whether another iteration is needed by prioritizing machine-readable status artifacts and using targeted evidence windows from `agent_reports/`, `results/`, and `idea.json`.

## Required Inputs
- `idea.json`
- `agent_reports/`
- `results/`
- the previous `agent_reports/master_report.md`

## Decision Rules
- Read `iteration_status.json` and validator-backed JSON first, then use targeted `search`, `read_json`, or bounded `view` calls to confirm missing evidence.
- Treat `master_report.md` as the previous iteration note only; newer evidence in other reports overrides it.
- The master loop should choose at most one next planner per iteration:
  - `experiment_code_planner`
  - `experiment_standard_science_planner`
  - `experiment_ablation_science_planner`
- The master loop should not invent new phase or lane names.
- **MANDATORY PHASE ORDER**: Code implementation (experiment_code_planner) MUST be completed before running any experiments (standard_science or ablation_science).
  - If `agent_reports/code_validator_report.json` does not exist with status PASS, you MUST choose `experiment_code_planner`.
  - If code implementation is incomplete or missing, experiments cannot run meaningfully.
- The only control output the outer loop consumes is:
  - `{"continue_iteration": true}`
  - `{"continue_iteration": false}`
- When `continue_iteration` becomes `false`, the outer runtime stops the master loop.

## Hard Rule
- Status JSON and validator reports are the primary decision surface; raw logs and raw result files are for targeted confirmation only.
- Natural-language summaries can support a decision but never replace the underlying experiment evidence.
- Smoke/debug/subset runs never count as sufficient final experiment evidence by themselves.
- Code correctness matters only insofar as the experiments remain scientifically meaningful; do not declare completion if implementation flaws invalidate the conclusions.
- Final ablation evidence must cover exactly the canonical components from `idea.json.components`, in the same order, with no extras or omissions.
- `ablation_results.json` is written by the ablation science agent immediately after ablation experiments complete, so the results are available for the next master iteration decision.
