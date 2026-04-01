# Experiment Agent

Experiment Agent is the OpenHands-based execution stack for the experiment phase of the X-Scientist pipeline. It prepares a workspace, delegates code and science work through a master agent, runs an iteration integrator after each master-loop phase, and finishes with a final ablation report agent.

## Runtime Shape

The package now uses a single public CLI:

```bash
python -m src.agents.experiment_agent.main --experiment <experiment_id> [options]
```

Default behavior:
1. Check for a validator-backed prepare handoff
2. Run `prepare` only if the handoff is missing, invalid, or `--force` is set
3. Run `master`
4. Let `master` delegate to `code` and `science` sub-agents

Useful flags:

```bash
python -m src.agents.experiment_agent.main --experiment exp001 --prepare-only
python -m src.agents.experiment_agent.main --experiment exp001 --max-iterations 8 --resume
```

## Directory Layout

```text
experiment_agent/
├── main.py
├── agents/
│   ├── base/           # Shared OpenHands base classes and schemas
│   ├── prepare/        # Prepare planner, step executor, worker, validator
│   ├── master/         # Master orchestrator
│   ├── code/           # Code planner, step executor, worker, validator
│   ├── science/        # Science planners, step executor, worker, validator
│   └── reporting/      # Final ablation report agents
├── config.py           # Experiment-agent config and path helpers
├── runtime/            # Lightweight artifact helpers, shared contracts, cache, checkpoints, memory hooks
├── tools/              # Local helpers such as parsing and security helpers
├── telemetry/          # Console hooks and runtime logging helpers
├── skills/             # Curated OpenHands skills
```

## OpenHands Conventions

- Agent entries are thin and live under `agents/<name>/entry.py`.
- Phase-local prompts live directly in the relevant planner/worker/validator Python modules.
- Planner agents own stage ordering; phase-local step executors own the worker/validator repair loop for a single step.
- Skills are loaded through `skills/__init__.py` with curated subsets per agent.
- OpenHands conversation setup, MCP normalization, secret registration, and stats persistence are shared through the base agent layer.

## Structured Outputs

The runtime keeps validator reports and step-level evidence as the authority for phase completion. The only final structured experiment artifact written for downstream use is:

- `ablation_results.json`

That file is written only after the master iteration loop exits, by a dedicated final ablation report agent. The master loop itself uses the iteration integrator's `iteration_status.json` for next-step decisions. The final ablation artifact is written at workspace root from `idea.json` plus the ablation experiment records under `agent_reports/` and `results/ablation/`. A run is not considered complete until that final reporting step succeeds.

## Workspace Path Contract

- All implementation code must live under `project/`.
- Prepared datasets must live under `dataset_candidate/`.
- Experiment outputs must live under `results/`.
- All planner, step-executor, worker, validator, and master coordination artifacts must live under `agent_reports/` using flat filenames.

- Human-facing agent summaries such as `prepare_idea.md`, `code_summary.md`, `standard_science_summary.md`, `ablation_science_summary.md`, `master_report.md`, and `master_summary.md` live under `agent_reports/`.
- `results/` is reserved for raw science outputs.
- The final `ablation_results.json` lives at workspace root.
- `prepare` is a startup prerequisite, not a master-controlled phase. `master` governs only `code`, `standard_science`, and `ablation_science`.
