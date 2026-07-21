---
name: code-planning
description: Build an ordered executable code plan for controlled experiment enablement
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Code Planning

## Mission
Translate the idea into an ordered executable code plan that supports:
1. All-components science conditions where every canonical idea component is enabled
2. Component-disabled science conditions using the same runner interface with one or more components disabled

**CRITICAL: Code must be SELF-CONTAINED in `project/`**
- All code must be COMPLETE and INDEPENDENT in `project/`.
- The code must NOT depend on `repos/` for core functionality.
- Everything needed to run experiments must be in `project/`.
- Repos are for REFERENCE ONLY - implement actual experiment code in `project/`.

## Protocol
- Write the managed artifact `code.plan` at `agent_reports/code/plan/latest.json` with ordered stages, project target paths, expected outputs, and verification commands.
- Map each idea.json component to code implementation node(s).
- For each component node, document the component-disable mechanism (how to turn that component off through the unified runner).
- Sequence stages so each worker can finish with a clean, runnable `project/`
  state before the next stage starts.
- Generate the managed `code.blueprint` artifact before delegating implementation.
- End the plan with a mandatory `final_integration_smoke` node that uses the real prepared dataset and the real API/model path when required.
- The real benchmark data from `dataset_candidate/` MUST be used in experiments, not synthetic data.
- Keep planner outputs, worker reports, reviewer reports, hook reports, phase summaries, and controlled artifacts under the structured `agent_reports/code/` layout.
- Merge worker outputs into `agent_reports/code/summary.md`, optional `agent_reports/code/usage.md`, `agent_reports/code/artifacts/integration_readiness.json`, and the reviewer-approved phase report `agent_reports/code/phase.json`.

## Report Requirements
The runtime owns reports. Do not write report files with tools. Each step keeps:
1. `agent_reports/code/worker/<step>/attempts/NNN.json` and `latest.json`
2. `agent_reports/code/review/<step>/attempts/NNN.json` and `latest.json`
3. `agent_reports/code/hook/<step>/attempts/NNN.json` and `latest.json`

**Phase-level reports:**
- `agent_reports/code/plan/latest.json` - Ordered step plan
- `agent_reports/code/plan/planner_report.json` - Planner summary
- `agent_reports/code/summary.md` - Human-readable code status
- `agent_reports/code/usage.md` - How to run experiments (optional)
- `agent_reports/code/artifacts/integration_readiness.json` - Readiness status
- `agent_reports/code/phase.json` - Final phase verdict

## Hard Rules
- Do not claim a condition is enabled unless a runnable command exists.
- Do not rely on parallel code execution; the runtime executes the ordered
  `stages` list and gates each stage before moving on.
- `final_integration_smoke` must not be satisfied by imports, mocks, or dry-run-only evidence.
- Every idea.json component must have a corresponding component-disable mechanism in the plan.
- The plan must include runnable entrypoints/configuration for all-components and component-disabled science conditions.
- Every step must finish through the worker/review/hook prefinish gate.
