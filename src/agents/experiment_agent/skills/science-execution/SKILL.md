---
name: science-execution
description: Execute unified science conditions through reviewer and hook gates
argument-hint: ""
allowed-tools: Bash(*), Read, Edit, Write, Glob, Grep, Agent
license: MIT
---

# Science Execution

## Mission
Execute exactly one science condition from the planner, preserving raw evidence and completing only through the prefinish reviewer and hook gate.

## Key Principles
- **No hardcoding**: Read dataset, model, API, and command bindings from validated prepare/code/science plan artifacts.
- **Real data only**: Use `dataset_candidate/` data, not synthetic or random data.
- **Unified conditions**: Treat all-components and component-disabled runs as the same kind of science condition; only their component sets differ.

## Internal Loop
1. Read the assigned condition contract, planner blueprint, validated prepare artifacts, code handoff, and existing science evidence.
2. Run from the workspace root. Do not `cd project`.
3. Execute the exact command or a faithful repaired command that preserves the declared component set.
4. Keep raw outputs under the declared `results/science/<condition_id>/` subtree.
5. Write controlled evidence manifests only through artifact tools under `agent_reports/science/evidence/`.
6. Stop with the required worker JSON; the prefinish hook persists worker/review/hook reports and returns feedback in the same session if anything fails.

## Condition Rules
- All-components conditions have `disabled_components: []` and can serve as references.
- Component-disabled conditions must use the declared disable/toggle mechanism and reference a passed all-components condition.
- The evidence must show the declared component set, command, dataset binding, metrics, output paths, and logs.
- Do not write final `ablation_results.json`; final materialization is runtime-owned.

## Hard Rules
- Do not produce final verdict while required coverage is missing.
- Reviewer report and hook receipt are the completion authority; summaries are human-readable only.
- Import tests, package checks, tiny snippets, smoke/debug/subset runs, and dry runs do not count as formal science evidence.
- Do not hardcode dataset names, model IDs, or API keys in experiment commands.

## Reviewer `result` Labels
The reviewer's `result` field is a vocabulary, not free text. Finalization accepts:
- `positive` — removing this component clearly hurts the metric (component helps).
- `negative` — removing this component clearly helps the metric (component hurts).
- `neutral` — the ablation ran end-to-end but the metric difference is within noise or practically small.
- `inconclusive` — the ablation ran end-to-end but the evidence does not support a directional conclusion; it is valid symbolic memory when `follow_up_required: false`.

Use `follow_up_required: true` to block completion when the run failed, coverage is missing, or evidence is insufficient to trust the condition. Do not use the `inconclusive` label itself as a blocker.
