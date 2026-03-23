# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/`. Agent implementations are split under `src/agents/`: `idea_agent/`, `experiment_agent/`, `paper_agent/`, and `survey_agent/`. Shared pipeline orchestration is in `src/pipeline/`, while repo-level config defaults live in `src/config/default.yaml`. Tests currently exist in top-level `tests/` and in some agent subtrees such as `src/agents/survey_agent/tests/`. Runtime outputs are written to `workspace/`, `logs/`, `database/`, and agent-specific `workspaces/`; treat these as generated artifacts unless a task explicitly targets them.

## Build, Test, and Development Commands
Prefer the reproducible Conda environment:
```bash
conda env create -f environment.yml
conda activate research-agent
```
Use `pip install -r requirements.txt` only for lighter setups. Main entrypoints:
```bash
./run_pipeline.sh                     # Integrated Survey -> Idea -> Experiment loop
./run_idea.sh                         # Run the idea agent
./run_survey.sh --topic "..."         # Run the survey agent
./run_experiment.sh --experiment demo --idea-json /path/to/idea_result.json --prepare
python -m pytest tests/test_compile.py
```

## Coding Style & Naming Conventions
This repository is Python-first. Follow PEP 8 with 4-space indentation, `snake_case` for modules/functions, `PascalCase` for classes, and descriptive YAML keys that mirror existing configs. Keep new modules inside the relevant agent package instead of adding cross-cutting scripts at the root. No repo-wide formatter or linter config is checked in, so match surrounding style and keep imports, logging, and CLI patterns consistent with adjacent files.

## Testing Guidelines
`pytest` is installed, but some tests use `unittest` and are run through `pytest`. Name new test files `test_*.py` and keep them close to the code they verify when a package already has a local `tests/` directory; otherwise use top-level `tests/`. Favor small, isolated tests with mocks for external LLM, API, and filesystem side effects. Run the narrowest relevant test first, then broaden as needed.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `Fix config priority and experiment_id naming` and `Add Pipeline integration with symbolic memory feedback loop`. Keep commit titles concise, capitalized, and focused on one change. Pull requests should include the affected agent or pipeline stage, configuration changes, manual test commands run, and sample output paths or screenshots when UI, diagrams, or generated documents change.

## Configuration & Secrets
Do not commit API keys or local absolute paths. Keep secrets in environment variables or untracked local config overrides, and verify paths in shell scripts before sharing examples.
