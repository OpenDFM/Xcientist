---
name: science-execution
description: Coverage-driven experiment execution loop
license: MIT
---

# Science Execution

## Mission
Run benchmark experiments until all required component and benchmark coverage gaps are closed or a hard blocker is proven.

## Internal Loop
1. Read validated prepare artifacts, code handoff, idea context, and existing science artifacts.
2. Treat those validated artifacts as the only allowed model/API/data target inventory.
3. For each missing science item, derive a concrete runnable command for `baseline`, `full_method`, or `ablation:<component>`.
4. Use smoke/debug runs only to unblock execution; they do not count as completion evidence.
5. Execute the full command chain and keep raw outputs on disk.
6. Record exact commands, output paths, exit status, dataset/model bindings, and raw artifacts in the worker report.
7. Update lane summary only after validator-backed evidence exists.

## Key Requirements
- **CRITICAL**: Benchmark experiments MUST use real data from `dataset_candidate/` directory, NOT synthetic or random data.
- API calls must read credentials from `{workspace}/.env`, not hardcoded keys.
- Missing models must be downloaded from HuggingFace before declaring blocker.
- For ablation: use canonical component names from `idea.json.components`, preserve order, include `method_context`.

## Ablation Requirements
- Each step-level validator report must record: `result`, `metric`, `value`, `confidence`, `analysis`, `method_context`.
- Component set and order must exactly equal `idea.json.components`.
- Do not write the final `ablation_results.json` - that is owned by the later ablation report integrator.

## Hard Rules
- Do not produce final verdict while required coverage is missing.
- Validator report is the completion authority; summaries are human-readable only.
- Import tests, package checks, and tiny snippets do not count as formal experiments.
- Subset/slice runs do not count as formal completion evidence.
