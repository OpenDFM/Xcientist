# Repository Guidance

## Purpose
- This repository hosts X-Scientist, a multi-agent research workflow covering survey, idea generation, experiment execution, and report writing.
- The `experiment_agent` under `src/agents/experiment_agent/` is the control plane for experiment workspaces, code enablement, standard science runs, ablations, and final reporting.

## Working Rules
- Prefer deterministic runtime state and validator-backed JSON artifacts over free-form summaries.
- Treat `.openhands/`, `.conversations/`, and `.checkpoints/` as runtime persistence, not business evidence.
- Keep final experiment artifacts compatible with the current workspace contract. In particular, `ablation_results.json` must be written at the workspace root and must keep its current schema.
- When reading files, prefer targeted search and bounded windows over full-file reads.

## Repository Layout
- `src/agents/experiment_agent/`: experiment orchestration, tools, prompts, runtime contracts, and final artifact materialization.
- `src/agents/idea_agent/`: idea-generation pipeline and its configs.
- `src/pipeline/`: integrated survey -> idea -> experiment loop.
- `tests/`: contract, manifest, and control-plane tests.
- `workspace/`: experiment and pipeline workspaces.

## Environment
- Prefer the Conda environment defined by `environment.yml`.
- `requirements.txt` is a lighter-weight fallback when Conda reproduction is not needed.
- For experiment-agent tests, ensure `PYTHONPATH` includes the repository root.
