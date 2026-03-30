---
name: prepare-planning
description: Plan ordered prepare subtasks and end with validator-backed handoff artifacts
license: MIT
---

# Prepare Planning

## Mission
Decompose workspace preparation into ordered worker tasks and end with validator-backed handoff artifacts for later phases.

## Protocol
- Create `agent_reports/prepare_plan.json` before dispatching workers.
- Dispatch workers in this fixed order:
  1. repository discovery/download
  2. environment setup
  3. dataset discovery/download
- Consume worker report files instead of relying on prose.
- After validator-backed completion, synthesize `agent_reports/prepare_idea.md` and any lightweight handoff notes needed by later phases.
- Treat `idea.json.components` as a first-class handoff artifact that must be copied into `prepare_idea.md` without renaming or reordering.
- **CRITICAL**: Dataset discovery MUST identify and record the exact file paths under `dataset_candidate/` that later phases (code, science) must use. For each dataset file, explicitly document:
  - The exact file path
  - The purpose/usage in experiments (e.g. "baseline evaluation", "ablation testing", "full method validation")
  - Which experiment conditions use it (baseline, full_method, ablation)
  - Any preprocessing or filtering required
- **CRITICAL**: Environment and API discovery MUST enumerate ALL environment variables and APIs that might be needed:
  - API keys (OPENAI_API_KEY, etc.)
  - Model IDs to use (e.g. `sentence-transformers/all-MiniLM-L6-v2` for embeddings, `gpt-4o` for LLM)
  - Any external services or tokens
  - For each, document the exact purpose and which component uses it
- **CRITICAL**: All implementation code MUST be written to `project/` directory in workspace, NOT `src/`. The `project/` directory is the designated location for experiment code.
- **CRITICAL**: Code in `project/` must be SELF-CONTAINED. It must NOT depend on `repos/` for core functionality. Repos are for REFERENCE ONLY.
- **CRITICAL**: `prepare_idea.md` must include `## Code Implementation Guidance` section with:
  - Required project structure and file organization
  - Key functions/methods to implement
  - Entry points for running experiments
  - Integration points between components
  - Expected API interfaces
- **CRITICAL**: `prepare_idea.md` must include `## Component Correspondence` section mapping:
  - Each idea.json component name → which files/functions implement it
  - Each component → which experiments validate it
  - Dependencies between components

## Report Requirements
**Every stage produces exactly THREE reports under `agent_reports/`:**
1. `<stage>_worker_report.json` - Worker details: commands run, files created, paths verified, blockers found
2. `<stage>_validator_report.json` - Validator verdict: PASS/FAIL, findings, required fixes
3. `<stage>_executor_report.json` - Executor summary: repair attempts used, final status

**Phase-level reports:**
- `prepare_plan.json` - Ordered stage plan
- `prepare_planner_report.json` - Planner summary
- `prepare_idea.md` - Self-complete handoff document
- `prepare_validator_report.json` - Final phase verdict

## Hard Rules
- Repositories must be validated before dataset-to-benchmark mappings are finalized.
- `prepare_idea.md` is human-facing only; validator reports remain the source of truth.
- `prepare_idea.md` must contain these EXACT sections:
  - `## Idea Summary` - Complete restatement of the idea
  - `## Idea JSON Components` - Full copy of idea.json.components
  - `## Code Implementation Guidance` - How to implement as code
  - `## Component Correspondence` - Which code implements which component
  - `## Dataset Usage Guidance` - Exact paths and usage for datasets
  - `## Environment Variable Usage Guidance` - Env vars needed by each component
  - `## Resource Acquisition Log` - What was acquired
  - `## Repository-to-Dataset Mapping` - Repo to dataset linkage
  - `## Real Experiment Targets` - Verified targets
  - `## Canonical Idea Components` - Canonical component list from idea.json
- Do not rename, reorder, or omit any of these sections.
- Every stage must produce all three required reports (worker, validator, executor).
