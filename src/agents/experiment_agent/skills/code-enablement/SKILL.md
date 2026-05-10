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
1. Standard science experiments can run baseline vs full_method comparisons
2. Ablation science experiments can disable each idea.json component individually
3. All experiments use real data from `dataset_candidate/` and real API credentials

**CRITICAL: Code must be SELF-CONTAINED in `project/`**
- All code must be COMPLETE and INDEPENDENT in `project/`.
- The code must NOT depend on `repos/` for core functionality.
- Everything needed to run experiments must be in `project/`.
- Later phases (science) will ONLY use code from `project/`, never from `repos/`.
- Repos are for REFERENCE ONLY - implement actual experiment code in `project/`.

## Protocol
- Read `agent_reports/prepare_idea.md` and validated prepare worker/validator reports before editing code.
- Read `idea.json.components` through the validated prepare handoff and preserve canonical component names.
- Bind code changes to the declared real experiment targets discovered in those reports.
- **CRITICAL**: Benchmark code MUST use real data files from `dataset_candidate/`, NOT synthetic or random data.
- **CRITICAL**: All experiment code MUST be under `project/`, NOT `src/`.
- **CRITICAL**: Each idea.json component must have a disable/ablation mechanism that:
  - Allows the component to be disabled WITHOUT modifying other components
  - Provides a clear method_context describing what the ablated variant does
  - Is invokable by both standard science and ablation science
- Ensure project supports: baseline runs, full method runs, per-component ablation runs.
- The last step must be real `final_integration_smoke` run using `dataset_candidate/` data.
- Remove mock fallback from formal experiment entrypoints. Mock may exist only behind explicit debug-only paths.
- API calls must read credentials from `{workspace}/.env`, not hardcoded keys.
- Missing models must be downloaded from HuggingFace before declaring blocker.

## Report Requirements
**Every step produces exactly THREE reports under `agent_reports/`:**
1. `<step>_worker_report.json` - Worker details: code changes, files created, commands run, what was implemented
2. `<step>_validator_report.json` - Validator verdict: PASS/FAIL, findings, required fixes
3. `<step>_executor_report.json` - Executor summary: repair attempts used, final status

## Required Output
- Stage worker report requested by the planner
- Flat raw smoke artifacts under `agent_reports/` for `final_integration_smoke`
- For each idea.json component: the ablation mechanism (how to disable it) and method_context

## Failure Conditions
- Required condition lacks runnable command
- `final_integration_smoke` does not use `dataset_candidate/` data or real API/model
- Benchmark code uses synthetic/random data or mock vectorstores
- Experiment code placed in `src/` instead of `project/`
- No disable/ablation mechanism for an idea.json component
- Baseline and full_method entrypoints not both implemented
- Ablation mechanism requires modifying other components to work
- Code depends on `repos/` for core functionality
