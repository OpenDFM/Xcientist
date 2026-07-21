---
name: prepare-planning
description: Plan ordered prepare subtasks and end with reviewer-approved handoff artifacts
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Prepare Planning

## Mission
Decompose workspace preparation into ordered worker tasks and end with reviewer-approved handoff artifacts for later phases.

## Protocol
- Create the managed `prepare.plan` artifact at `agent_reports/prepare/plan/latest.json` before dispatching workers.
- Dispatch workers in this fixed order:
  1. repository/resource discovery and repository validation
  2. dataset acquisition and loader/schema verification
  3. model/API target acquisition and dry-run/load verification
  4. final environment setup after resource choices are fixed
  5. synthesis of reviewer-approved handoff artifacts
- Consume worker report files instead of relying on prose.
- After reviewer-approved completion, synthesize `agent_reports/prepare/artifacts/idea.md` and `agent_reports/prepare/artifacts/target_inventory.json`.
- Treat `idea.json.components` as a first-class handoff artifact that must be copied into `agent_reports/prepare/artifacts/idea.md` without renaming or reordering.
- **CRITICAL**: Dataset discovery MUST identify and record the exact file paths under `dataset_candidate/` that later phases (code, science) must use. For each dataset file, explicitly document:
  - The exact file path
  - The purpose/usage in experiments (e.g. "all-components reference", "component-disabled condition", "full method validation")
  - Which science conditions use it
  - Any preprocessing or filtering required
- **CRITICAL**: Environment and API discovery MUST enumerate ALL environment variables and APIs that might be needed:
  - API keys (OPENAI_API_KEY, etc.)
  - Model IDs to use (e.g. `sentence-transformers/all-MiniLM-L6-v2` for embeddings, `gpt-4o` for LLM)
  - Any external services or tokens
  - For each, document the exact purpose and which component uses it
  - If `OPENAI_API_KEY` is required, always record the paired `OPENAI_BASE_URL` entry alongside it in reports and handoff artifacts
- **CRITICAL**: All implementation code MUST be written to `project/` directory in workspace, NOT `src/`. The `project/` directory is the designated location for experiment code.
- **CRITICAL**: Code in `project/` must be SELF-CONTAINED. It must NOT depend on `repos/` for core functionality. Repos are for REFERENCE ONLY.
- **CRITICAL**: `agent_reports/prepare/artifacts/idea.md` must include `## Code Implementation Guidance` section with:
  - Required project structure and file organization
  - Key functions/methods to implement
  - Entry points for running experiments
  - Integration points between components
  - Expected API interfaces
- **CRITICAL**: `agent_reports/prepare/artifacts/idea.md` must include `## Component Correspondence` section mapping:
  - Each idea.json component name → which files/functions implement it
  - Each component → which experiments validate it
  - Dependencies between components

## Report Requirements
The runtime owns reports. Do not write report files with tools. Each prepare stage keeps:
1. `agent_reports/prepare/worker/<stage>/attempts/NNN.json` and `latest.json`
2. `agent_reports/prepare/review/<stage>/attempts/NNN.json` and `latest.json`
3. `agent_reports/prepare/hook/<stage>/attempts/NNN.json` and `latest.json`

**Phase-level reports:**
- `agent_reports/prepare/plan/latest.json` - Ordered stage plan
- `agent_reports/prepare/plan/planner_report.json` - Planner summary
- `agent_reports/prepare/artifacts/discovery.json` - Search queries, candidates, selection rationale, rejection reasons
- `agent_reports/prepare/artifacts/repos.json` - Verified repository/source manifest
- `agent_reports/prepare/artifacts/dataset.json` - Verified dataset manifest
- `agent_reports/prepare/artifacts/model.json` - Verified model/API manifest
- `agent_reports/prepare/artifacts/env.json` - Verified final environment manifest
- `agent_reports/prepare/artifacts/idea.md` - Self-complete handoff document
- `agent_reports/prepare/artifacts/target_inventory.json` - Machine-readable resource/component/benchmark inventory
- `agent_reports/prepare/phase.json` - Final phase verdict

## Hard Rules
- Tavily or web search can discover candidates but cannot serve as completion evidence by itself.
- Repositories must be validated before dataset-to-benchmark mappings are finalized.
- The final environment must be validated after repo, dataset, and model/API targets are selected.
- If no suitable real resource can be acquired or verified, report a blocker with rejection reasons instead of substituting toy resources.
- `agent_reports/prepare/artifacts/idea.md` is human-facing only; reviewer reports remain the source of truth.
- `agent_reports/prepare/artifacts/idea.md` must contain these EXACT sections:
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
- Every stage must finish through the worker/review/hook prefinish gate.
