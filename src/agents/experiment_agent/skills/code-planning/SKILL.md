---
name: code-planning
description: Build a DAG of code changes and parallelize only disjoint write sets
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Code Planning

## Mission
Translate the idea into an executable code DAG that supports:
1. Standard science: baseline vs full_method comparison
2. Ablation science: per-component disable/remove

**CRITICAL: Code must be SELF-CONTAINED in `project/`**
- All code must be COMPLETE and INDEPENDENT in `project/`.
- The code must NOT depend on `repos/` for core functionality.
- Everything needed to run experiments must be in `project/`.
- Repos are for REFERENCE ONLY - implement actual experiment code in `project/`.

## Protocol
- Write `agent_reports/code_plan.json` with DAG nodes, dependencies, write sets, expected outputs, and verification commands.
- Map each idea.json component to code implementation node(s).
- For each component node, document the ablation mechanism (how to disable it).
- Only place tasks in the same parallel batch when their write sets are disjoint.
- Generate code skeleton or blueprint artifacts before delegating implementation.
- End the plan with a mandatory `final_integration_smoke` node that uses the real prepared dataset and the real API/model path when required.
- The real benchmark data from `dataset_candidate/` MUST be used in experiments, not synthetic data.
- Keep step contracts, executor reports, worker reports, validator reports, and phase summaries under `agent_reports/` using flat filenames.
- Merge worker outputs into `agent_reports/code_summary.md`, optional `agent_reports/code_usage.md`, `agent_reports/code_integration_readiness.json`, and the validator-backed phase reports.

## Report Requirements
**Every step produces exactly THREE reports under `agent_reports/`:**
1. `<step>_worker_report.json` - Worker details: code changes, files created, commands run, what was implemented
2. `<step>_validator_report.json` - Validator verdict: PASS/FAIL, findings, required fixes
3. `<step>_executor_report.json` - Executor summary: repair attempts used, final status

**Phase-level reports:**
- `code_plan.json` - Ordered step plan
- `code_planner_report.json` - Planner summary
- `code_summary.md` - Human-readable code status
- `code_usage.md` - How to run experiments (optional)
- `code_integration_readiness.json` - Readiness status
- `code_validator_report.json` - Final phase verdict

## Hard Rules
- Do not claim a condition is enabled unless a runnable command exists.
- Do not parallelize tasks that can touch the same files.
- `final_integration_smoke` must not be satisfied by imports, mocks, or dry-run-only evidence.
- Every idea.json component must have a corresponding ablation mechanism in the plan.
- The plan must include baseline and full_method entrypoints for standard science.
- Every step must produce all three required reports (worker, validator, executor).
