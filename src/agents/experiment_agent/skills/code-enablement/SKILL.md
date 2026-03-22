---
name: code-enablement
description: Enable all required experiment conditions with runnable commands
license: MIT
---

# Code Enablement

## Mission
Make every required experiment condition executable from the current project.

## Protocol
- Read `agent_reports/prepare_idea.md` plus the validated prepare worker and validator reports before editing code.
- Read `idea.json.components` through the validated prepare handoff and preserve those canonical component names in any ablation-related enablement notes.
- Bind code changes to the declared real experiment targets discovered in those reports.
- Ensure the project supports:
  - baseline runs
  - full method runs
  - per-component ablation or degraded fallback runs
- Persist runnable command inventory and blockers to the worker report requested by the planner.
- Treat syntax/import checks only as preflight. Do not treat them as experiment evidence.
- Do not run formal benchmark experiments here; only verify that commands and entrypoints are runnable.
- Do not write formal experiment result artifacts under `results/`. The Code phase may validate executability, but it must not claim science completion or materialize downstream result summaries.
- The last code step must be a real `final_integration_smoke` run, not a mock or import-only check.
- Remove or isolate mock fallback from formal experiment entrypoints. Mock behavior may exist only behind explicit debug-only paths.
- If an external API is required and credentials are available, read the official API docs first and then wire the API-backed execution path.
- If a required model/checkpoint is missing locally, download it from HuggingFace or the official source before declaring a blocker.
- For HuggingFace downloads, use the official HuggingFace endpoints and inherit the current shell network environment instead of rewriting proxy settings inside the agent.
- Include `target_bindings` in the worker report so later agents know which real targets were enabled.
- Include per-component enablement or blocker notes keyed by the exact canonical component names when ablation support is relevant.

## Required Output
- Stage worker report requested by the planner
- Flat raw smoke artifacts under `agent_reports/` when the assigned step is `final_integration_smoke`

## Failure Conditions
- Any required condition lacks a runnable command.
- A component is declared covered in prose but has no executable path.
- Formal benchmark result artifacts are produced by the Code agent.
- Formal entrypoints still silently fall back to mock backends.
- The worker report does not declare the real targets bound by the code changes.
- `final_integration_smoke` does not use the prepared dataset or the real API/model path when required.
