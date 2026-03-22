---
name: code-planning
description: Build a DAG of code changes and parallelize only disjoint write sets
license: MIT
---

# Code Planning

## Mission
Translate master requirements into an executable code DAG and parallel worker batches.

## Protocol
- Write `agent_reports/code_plan.json` with DAG nodes, dependencies, write sets, expected outputs, and verification commands.
- Only place tasks in the same parallel batch when their write sets are disjoint.
- Generate code skeleton or blueprint artifacts before delegating implementation.
- End the plan with a mandatory `final_integration_smoke` node that uses the real prepared dataset and the real API/model path when required.
- Keep step contracts, executor reports, worker reports, validator reports, and phase summaries under `agent_reports/` using flat filenames.
- Merge worker outputs into `agent_reports/code_summary.md`, optional `agent_reports/code_usage.md`, `agent_reports/code_integration_readiness.json`, and the validator-backed phase reports.

## Hard Rules
- Do not claim a condition is enabled unless a runnable command exists.
- Do not parallelize tasks that can touch the same files.
- `final_integration_smoke` must not be satisfied by imports, mocks, or dry-run-only evidence.
