---
name: code-enablement
description: Enable all required experiment conditions with runnable commands
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Code Enablement

## Mission
Implement the FULL IDEA in `project/` such that:
1. All-components science conditions can run with every canonical idea component enabled
2. Component-disabled science conditions can disable each idea.json component individually through the same runner interface
3. All experiments use real data from `dataset_candidate/` and real API credentials

**CRITICAL: Code must be SELF-CONTAINED in `project/`**
- All code must be COMPLETE and INDEPENDENT in `project/`.
- The code must NOT depend on `repos/` for core functionality.
- Everything needed to run experiments must be in `project/`.
- Later phases (science) will ONLY use code from `project/`, never from `repos/`.
- Repos are for REFERENCE ONLY - implement actual experiment code in `project/`.

## Protocol
- Read `agent_reports/prepare/artifacts/idea.md`, `agent_reports/prepare/artifacts/target_inventory.json`, and validated prepare worker/reviewer reports before editing code.
- Read `idea.json.components` through the validated prepare handoff and preserve canonical component names.
- Bind code changes to the declared real experiment targets discovered in those reports.
- **CRITICAL**: Benchmark code MUST use real data files from `dataset_candidate/`, NOT synthetic or random data.
- **CRITICAL**: All experiment code MUST be under `project/`, NOT `src/`.
- **CRITICAL**: Each idea.json component must have a component-disable mechanism that:
  - Allows the component to be disabled WITHOUT modifying other components
  - Provides a clear method_context describing what the component-disabled condition does
  - Is invokable by unified science conditions without changing source code between runs
- Ensure project supports: all-components runs and per-component disabled runs.
- The last step must be real `final_integration_smoke` run using `dataset_candidate/` data.
- Remove mock fallback from formal experiment entrypoints. Mock may exist only behind explicit debug-only paths.
- API calls must read credentials from `{workspace}/.env`, not hardcoded keys.
- Missing models must be downloaded from HuggingFace before declaring blocker.

## Report Requirements
The runtime owns reports. Do not write report files with tools. Each code step is recorded under:
- `agent_reports/code/worker/<step>/attempts/NNN.json` and `latest.json`
- `agent_reports/code/review/<step>/attempts/NNN.json` and `latest.json`
- `agent_reports/code/hook/<step>/attempts/NNN.json` and `latest.json`

## Required Output
- Final worker JSON response requested by the planner; the runtime persists the worker report.
- Managed smoke evidence at `agent_reports/code/artifacts/final_integration_smoke.json` for `final_integration_smoke`
- For each idea.json component: the component-disable mechanism and method_context

## Failure Conditions
- Required condition lacks runnable command
- `final_integration_smoke` does not use `dataset_candidate/` data or real API/model
- Benchmark code uses synthetic/random data or mock vectorstores
- Experiment code placed in `src/` instead of `project/`
- No component-disable mechanism for an idea.json component
- The all-components reference path or any component-disabled condition lacks a runnable entrypoint
- Component-disable mechanism requires modifying other components to work
- Code depends on `repos/` for core functionality
