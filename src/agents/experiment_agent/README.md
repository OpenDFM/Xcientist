# Experiment Agent

Experiment Agent is the OpenHarness-based execution stack for the experiment phase of the X-Scientist pipeline. It prepares a workspace, runs code and science phases in a fixed order, gates every work unit through prefinish reviewers, and finishes through an OpenHarness finalization worker whose STOP/prefinish hook materializes ablation and symbolic-memory artifacts.

## Runtime Shape

The package now uses a single public CLI:

```bash
python -m src.agents.experiment_agent.main --experiment <experiment_id> [options]
```

Default behavior:
1. Check for a reviewer-approved prepare handoff
2. Run `prepare` only if the handoff is missing, invalid, or `--force` is set
3. Run `master`
4. Let `master` run `code`, then unified `science`
5. Run the finalization worker; its prefinish hook writes final ablation results and symbolic-memory receipt

Useful flags:

```bash
python -m src.agents.experiment_agent.main --experiment exp001 --prepare-only
python -m src.agents.experiment_agent.main --experiment exp001 --resume
```

## Directory Layout

```text
experiment_agent/
├── main.py
├── agents/
│   ├── base/           # Shared OpenHarness base agent and prompt builder
│   ├── prepare/        # Prepare planner, worker, and reviewer
│   ├── master/         # Master orchestrator
│   ├── code/           # Code planner, worker, and reviewer
│   ├── science/        # Science planner, worker, and reviewer
│   └── finalization/   # Finalization worker and STOP/prefinish gate
├── config.py           # Experiment-agent config and path helpers
├── runtime/            # Artifact registry, phase runner, contracts, OpenHarness bridge, finalization hooks
├── tools/              # Compatibility namespace; managed tools live in runtime/artifacts.py
├── telemetry/          # Console progress and lightweight telemetry helpers
├── skills/             # Curated experiment skills
```

## OpenHarness Conventions

- Agent entries are thin and live under `agents/<name>/entry.py`.
- Phase-local prompts live directly in the relevant planner/worker/reviewer Python modules.
- Planner agents own stage ordering; the shared runtime phase runner owns each worker/prefinish-review repair loop.
- OpenHarness conversation setup and tool access are shared through the base agent layer.
- The vendored OpenHarness source is loaded from `src/harness/src`; experiment-agent code should not import an external harness package by accident.

## Structured Outputs

The runtime keeps reviewer reports and step-level evidence as the authority for phase completion. Phase reports are organized as a readable story under `agent_reports/`:

```text
agent_reports/
├── _runtime/
│   ├── artifact_registry.json
│   ├── artifact_ledger.jsonl
│   ├── mcp_status.json
│   └── run_timeline.jsonl
├── prepare/
├── code/
├── science/
└── ablation/
```

Each phase contains `plan/latest.json`, `plan/executable.json`, `plan/planner_report.json`, phase summaries, and per-step `worker/`, `review/`, and `hook/` directories. Each step directory keeps immutable `attempts/NNN.json` plus `latest.json`.

The final structured experiment artifacts are:

- `agent_reports/ablation/final/ablation_results.json`
- `agent_reports/ablation/final/symbolic_memory_receipt.json`

`agent_reports/ablation/final/ablation_results.json` is written only after code and science reviewer gates have passed. The finalization worker enters an OpenHarness session, then its STOP/prefinish hook verifies the science lineage from `agent_reports/science/plan/executable.json` through each condition's hook/review/evidence reports and artifact-ledger hashes, writes the final artifact under `agent_reports/ablation/final/`, converts it into Idea Agent symbolic memory, and records the receipt. If finalization fails, the hook feedback is returned to the same finalization worker session.

## Workspace Path Contract

- All implementation code must live under `project/`.
- Selected implementation may be copied from `repos/` into `project/`, but runtime execution must not depend on `repos/` directly.
- Prepared datasets must live under `dataset_candidate/`.
- Experiment outputs must live under `results/`.
- All controlled artifacts, planner outputs, worker reports, reviewer reports, hook reports, and master coordination artifacts must live under the structured `agent_reports/` layout.
- Planner artifacts live at `agent_reports/<phase>/plan/latest.json` and `agent_reports/<phase>/plan/executable.json`, with the ordered step list in `stages`.
- Human-facing agent summaries such as `agent_reports/prepare/artifacts/idea.md`, `agent_reports/code/summary.md`, `agent_reports/science/summary.md`, `agent_reports/_runtime/master_report.md`, and `agent_reports/_runtime/master_summary.md` live under `agent_reports/`.
- Raw science outputs must live under `results/science/<condition_id>/`.
- The final `ablation_results.json` lives under `agent_reports/ablation/final/`.
- If repo code is copied into `project/`, record its source/target mapping with `record_sources`; provenance lives in `agent_reports/_runtime/artifact_ledger.jsonl`.
- Code/science step contracts must explicitly declare `repo_source_paths`, `repo_copy_intent`, and `project_target_paths`; repo usage must never remain implicit.
- `prepare` is a startup prerequisite, not a master-controlled phase. `master` runs only `code` and unified `science`, in that order.
