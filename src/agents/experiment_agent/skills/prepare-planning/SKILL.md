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
  - The exact file path (e.g. `dataset_candidate/longmemeval_oracle.json`)
  - The purpose/usage in experiments (e.g. "baseline evaluation", "ablation testing", "full method validation")
  - Which experiment conditions use it (baseline, full_method, ablation)
  - Any preprocessing or filtering required
- **CRITICAL**: Environment and API discovery MUST enumerate ALL environment variables and APIs that might be needed:
  - API keys (OPENAI_API_KEY, etc.)
  - Model IDs to use (e.g. `sentence-transformers/all-MiniLM-L6-v2` for embeddings, `gpt-4o` for LLM)
  - Any external services or tokens
  - For each, document the exact purpose and which component uses it
- **CRITICAL**: All implementation code MUST be written to `project/` directory in workspace, NOT `src/`. The `project/` directory is the designated location for experiment code.

## Hard Rules
- Repositories must be validated before dataset-to-benchmark mappings are finalized.
- `prepare_idea.md` is human-facing only; validator reports remain the source of truth.
- `prepare_idea.md` must contain an explicit `## Idea Components` section that preserves the exact `idea.json.components` list and order.
