---
name: science-execution
description: Coverage-driven experiment execution loop
license: MIT
---

# Science Execution

## Mission
Run standardized benchmark experiments (baseline vs full method) until all required coverage gaps are closed or a hard blocker is proven.

## Key Principles
- **No hardcoding**: Read dataset, model, and API bindings from validated prepare handoff (`prepare_idea.md`, worker reports). Do NOT use hardcoded values.
- **Real data only**: Must use `dataset_candidate/` data, NOT synthetic or random data.
- **Real APIs/Models**: Must use real API credentials from `{workspace}/.env` and real model checkpoints.
- **Standardized comparison**: Run baseline (standard/original) vs full method (all components enabled) for comparison.

## Internal Loop
1. Read validated prepare artifacts, code handoff, idea context, and existing science artifacts with bounded tool usage: `read_json` first, then `search`, then the smallest useful `view` window.
2. Treat those validated artifacts as the only allowed model/API/data target inventory.
3. For standard science: derive commands for `baseline` and `full_method` conditions.
4. Use smoke/debug runs only to unblock execution; they do not count as completion evidence.
5. Execute the full command chain and keep raw outputs on disk.
6. Record exact commands, output paths, exit status, dataset/model bindings, and raw artifacts in the worker report.
7. Update lane summary only after validator-backed evidence exists.

## Standard Science Requirements
- **Baseline condition**: Run with standard/original implementation
- **Full method condition**: Run with ALL idea.json components enabled
- Compare metrics between baseline and full method
- Both conditions must use the same dataset from `dataset_candidate/`
- Produce statistically meaningful comparison

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
- Do not hardcode dataset names, model IDs, or API keys in experiment commands.
