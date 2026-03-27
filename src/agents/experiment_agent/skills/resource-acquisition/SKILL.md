---
name: resource-acquisition
description: Deterministic protocol for repository, dataset, and benchmark acquisition
license: MIT
---

# Resource Acquisition

## Mission
Prepare real, verifiable resources for the current experiment.

**CRITICAL: Code in `project/` must be SELF-CONTAINED**
- The code in `project/` must be COMPLETE and INDEPENDENT.
- It must NOT depend on `repos/` for core functionality.
- Repos are for REFERENCE ONLY - implement actual experiment code in `project/`.

## Protocol
- Read `idea.json` or `agent_reports/prepare_idea.md` first.
- Discover repositories, datasets, and benchmarks before writing documentation.
- Treat pre-existing directories as unverified until file checks pass.
- Record machine-readable status in the stage worker report requested by the planner.
- Record an explicit real experiment target inventory in the worker report:
  - exact models/checkpoints or API-backed models
  - exact dataset files/directories
  - exact benchmark entrypoints
  - exact API env vars and invocation surface
  - an explicit description of what later phases must run
- If the real experiment path requires local model checkpoints, acquire them during prepare rather than deferring them to science.
- The dataset acquisition step may also download model checkpoints when those checkpoints are part of the formal benchmark target.
- Create `agent_reports/prepare_idea.md` only after verification work is complete, not as an early summary artifact.
- For HuggingFace downloads, use the official HuggingFace endpoints and inherit the current shell network environment instead of rewriting proxy settings inside the agent.
- `prepare_idea.md` must use these EXACT headings:
  - `## Idea Summary`
  - `## Idea JSON Components`
  - `## Code Implementation Guidance`
  - `## Component Correspondence`
  - `## Dataset Usage Guidance`
  - `## Environment Variable Usage Guidance`
  - `## Resource Acquisition Log`
  - `## Repository-to-Dataset Mapping`
  - `## Real Experiment Targets`
  - `## Canonical Idea Components`
- Under `## Canonical Idea Components`, copy every `idea.json.components` entry exactly once, in the same order, preserving the `component` name and `explanation`.
- After writing `prepare_idea.md`, reopen it and confirm those headings exist before claiming completion.

## Report Requirements
**Every stage produces exactly THREE reports under `agent_reports/`:**
1. `<stage>_worker_report.json` - Worker details: commands run, files created, paths verified, blockers found
2. `<stage>_validator_report.json` - Validator verdict: PASS/FAIL, findings, required fixes
3. `<stage>_executor_report.json` - Executor summary: repair attempts used, final status

## Required Outputs
- Stage worker report requested by the planner
- Verified repository list with local paths
- Verified dataset list with `expected_paths`
- Verified local model/checkpoint list when the formal path depends on local models
- Verified benchmark list with runnable entrypoint or explicit blocker
- Real experiment target declarations with concrete model/API/data choices
- `agent_reports/prepare_idea.md` with the required headings and verified content
- Canonical component handoff copied from `idea.json.components`

## Failure Conditions
- Resource directories exist but no expected files are checked.
- A benchmark is claimed runnable without an entrypoint or smoke command.
- A dataset is claimed downloaded without explicit file evidence.
- Model/API/data targets are described vaguely instead of concretely.
- An API-backed dependency is needed but no usable invocation surface is documented.
- `prepare_idea.md` is only a generic project summary and does not contain the required protocol headings.
- `prepare_idea.md` omits, renames, duplicates, or reorders the canonical idea components.
- The agent prints completion before performing a final read-back validation of `prepare_idea.md`.
