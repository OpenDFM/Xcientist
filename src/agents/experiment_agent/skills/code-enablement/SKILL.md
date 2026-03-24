---
name: code-enablement
description: Enable all required experiment conditions with runnable commands
license: MIT
---

# Code Enablement

## Mission
Make every required experiment condition executable from the current project.

## Protocol
- Read `agent_reports/prepare_idea.md` and validated prepare worker/validator reports before editing code.
- Read `idea.json.components` through the validated prepare handoff and preserve canonical component names.
- Bind code changes to the declared real experiment targets discovered in those reports.
- **CRITICAL**: Benchmark code MUST use real data files from `dataset_candidate/` (e.g. LongMemEval JSON files), NOT synthetic or random data.
- **CRITICAL**: All experiment code MUST be under `project/`, NOT `src/`.
- Ensure project supports: baseline runs, full method runs, per-component ablation runs.
- The last step must be real `final_integration_smoke` run using `dataset_candidate/` data.
- Remove mock fallback from formal experiment entrypoints. Mock may exist only behind explicit debug-only paths.
- API calls must read credentials from `{workspace}/.env`, not hardcoded keys.
- Missing models must be downloaded from HuggingFace before declaring blocker.

## Required Output
- Stage worker report requested by the planner
- Flat raw smoke artifacts under `agent_reports/` for `final_integration_smoke`

## Failure Conditions
- Required condition lacks runnable command
- `final_integration_smoke` does not use `dataset_candidate/` data or real API/model
- Benchmark code uses synthetic/random data or mock vectorstores
- Experiment code placed in `src/` instead of `project/`
