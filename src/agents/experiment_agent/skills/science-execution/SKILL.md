---
name: science-execution
description: Coverage-driven experiment execution loop
license: MIT
---

# Science Execution

## Mission
Run standard benchmark experiments until all required component and benchmark coverage gaps are closed or a hard blocker is proven.

## Internal Loop
1. Read the validated prepare artifacts, code handoff, idea context, and any existing science artifacts.
2. Treat those validated artifacts as the only allowed model/API/data target inventory.
3. For each missing science item, derive a concrete runnable command for `baseline`, `full_method`, or `ablation:<component>`.
4. Use smoke/debug runs only to unblock execution; they do not count as completion evidence.
5. Once preflight succeeds, immediately move to formal experiments on the declared real targets.
6. For completion evidence, execute the full command chain promised by the planner and keep raw outputs on disk.
7. Record exact commands, exact output paths, exit status, dataset/model bindings, and key raw artifacts in the worker report.
8. Update the lane summary under `agent_reports/standard_science_summary.md` or `agent_reports/ablation_science_summary.md` only after validator-backed evidence exists.
9. For ablation, ensure each step-level validator report records the component verdict fields needed by the later ablation report integrator: `result`, `metric`, `value`, `confidence`, `analysis`, and `method_context`.
10. For ablation, use only canonical component names from `idea.json.components`, preserve their order, and include a concrete `method_context` for each exact component.
11. If an external API is required and credentials are available, read the official API docs first and then use the API. Generated code that calls APIs must read credentials from `{workspace}/.env` file (e.g. using `python-dotenv` or manually parsing). Do NOT hardcode API keys or endpoints in generated code.
12. If a required model/checkpoint is missing locally, download it from HuggingFace or the official source before declaring a blocker.
13. For HuggingFace downloads, use the official HuggingFace endpoints and inherit the current shell network environment instead of rewriting proxy settings inside the agent.
14. **CRITICAL**: Benchmark experiments MUST use the real data files from `dataset_candidate/` directory, NOT synthetic or randomly generated data.
15. Repeat until validator-backed completion or a hard blocker is proven.

## Hard Rules
- Do not produce a final verdict while required coverage is missing.
- The final `ablation_results.json` is owned by a later ablation report integrator, not by the science worker.
- The validator report is the completion authority; lane summaries under `agent_reports/` are only human-readable summaries.
- Each ablation entry must include `result`, `metric`, `value`, `confidence`, `analysis`, and `method_context`.
- The final ablation component set and order must exactly equal `idea.json.components`.
- Import tests, package checks, and tiny inline snippets do not count as formal experiments.
- Subset runs and `--max-entries` style slices do not count as formal completion evidence.
- Lane summaries under `agent_reports/` are human-readable; completion is decided from validator-backed evidence.
