<p align="center">
 <img src="assets/logo.png" alt="Xcientist logo" width="70%">
</p>

<h2 align="center">Externalizing Research Synthesis and Decision-Making in AI Scientist through a Research Harness</h2>

<p align="center">
<img src="https://img.shields.io/badge/python-3.12-blue" alt="python">
<a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-tmp-red" alt="arXiv"></a>
<img src="https://img.shields.io/badge/website-page-blue" alt="website">
</p>

<div align="center">

**[English](README.md) | [简体中文](README_CN.md)**

</div>

Xcientist is a multi-agent research workflow for turning a topic into survey artifacts, structured ideas, executable experiments, and paper drafts. The repository currently centers on four agent stacks:

- `Survey Agent`: collects papers, builds topic clusters, and writes survey outputs.
- `Idea Agent (LigAgent)`: turns a topic or seed idea into a research proposal with survey-grounded retrieval, graph-backed references, and Memory-Guided MCTS.
- `Experiment Agent (SuperAgent)`: prepares a workspace, generates code, runs experiments, and integrates iteration reports.
- `Paper Agent`: reads an experiment workspace and writes a paper workspace with LaTeX drafts and compilation artifacts.

The repo also contains a prototype loop runner for `Survey -> Idea -> Experiment`, shared configuration, and a reusable memory subsystem.

## 🗂️ Repository Map

```text
Xcientist/
├── README.md
├── README_CN.md
├── environment.yml
├── run_survey.sh
├── run_idea.sh
├── run_experiment.sh
├── run_paper.sh
├── run_pipeline.sh
├── graph/                         # graph retrieval service and indexing scripts
├── scripts/                       # utility scripts such as MCP wrapper setup
├── skills/                        # local Codex skills used in this repo
└── src/
    ├── config/default.yaml        # unified project config
    ├── agents/
    │   ├── survey_agent/
    │   ├── idea_agent/
    │   ├── experiment_agent/
    │   └── paper_agent/
    ├── memory/                    # shared memory package
    └── pipeline/                  # end-to-end loop runner
```

## 🔄 How The Pieces Fit Together

```text
Topic
  -> Survey Agent
     output: survey.md + survey.json
  -> Idea Agent
     output: idea_result.json
  -> Experiment Agent
     output: workspace, results, ablation_results.json
  -> Paper Agent
     output: paper workspace, LaTeX draft, optional PDF
```

The pipeline runner in `src/pipeline/run_loop.py` automates the first three stages, but the individual agents remain the clearest way to operate and debug the system.

## ✅ Prerequisites

- `uv`
- Python `3.12`
- `node` and `npx` for Experiment Agent MCP servers
- API keys depending on which agent you run
- Local assets for graph-backed retrieval and memory-enabled workflows
  - Paper-Graph related resource [donwload link](https://drive.google.com/drive/folders/1lH1MI6gk7eh0HfvfOajcqAZg3n95v5BK?usp=drive_link), put them into `<repo_root>/data/processed`.
  - Embedding model download:
   ```
   mkdir -p models/bge-m3
   mkdir -p models/all-MiniLM-L6-v2
   modelscope download -- model baai/bge-m3 --local_dir <repo_root>/models/bge-m3
   modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --local_dir <repo_root>/models/all-MiniLM-L6-v2
   ```


## ⚙️ Installation

The default setup path is now `uv`.

```bash
uv sync
source .venv/bin/activate
cp .env.example .env
xcientist doctor
```

Common group combinations:

```bash
# Base CLI / config / API-only workflows
uv sync

# Memory-enabled and local-model workflows
uv sync --group memory --group ml

# Paper / PDF parsing stack
uv sync --group paper

# Full local environment
uv sync --all-groups
```

If you want local MCP wrapper scripts for Experiment Agent:

```bash
xcientist install-mcp-wrappers
```

`environment.yml` is still available as a legacy/full-environment fallback, but `uv sync` is the primary path for `Survey + Idea + Experiment + Pipeline`. The dependency layout is now split so the default install stays lightweight and heavy local-model / PDF stacks are opt-in.
After activation, the project exposes CLI entrypoints such as `xcientist`, `xcientist-survey`, and `xcientist-idea` directly in the shell.

## 🔐 Environment Variables

Different agents read slightly different variables. In practice, these are the most useful ones to define:

```bash
export OPENAI_API_KEY=...
export OPENAI_API_BASE=...
export OPENAI_BASE_URL=...
export SEMANTIC_SCHOLAR_API_KEY=...
export MINIMAX_API_KEY=...
export TAVILY_API_KEY=...
export GITHUB_AI_TOKEN=...
export JINA_API_KEY=...
```

Notes:

- Set both `OPENAI_API_BASE` and `OPENAI_BASE_URL` if you use a custom OpenAI-compatible endpoint.
- The CLI loads repo-root `.env` first and still falls back to `src/config/.env` for older setups.
- `src/config/default.yaml` is the main configuration file for the current unified workflow.
- Survey, Idea, Experiment, and Paper still have some agent-specific conventions on top of the unified config.

## 📦 Optional Local Assets

Some retrieval-heavy paths expect local assets that are not stored in the repository:

- `data/processed/graph.db`
- `data/processed/core_component_summary_vector_store/`
- `models/bge-m3/`
- `models/all-MiniLM-L6-v2/`

If you use graph-backed retrieval, start the graph service from the repository root:

```bash
uvicorn graph.server:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## 🚀 Quick Start

Recommended first-time flow:

```bash
uv sync --group memory --group ml
source .venv/bin/activate
cp .env.example .env
xcientist doctor
```

If doctor passes and your local assets are in place, use the commands below.

### 1. Run Survey Agent

Primary entrypoint:

```bash
xcientist survey
```

Override the topic directly:

```bash
xcientist survey --topic "LLM Agent Memory System"
```

Typical outputs:

- `src/agents/survey_agent/outputs/.../survey.md`
- `src/agents/survey_agent/outputs/.../survey.json`
- `src/agents/survey_agent/outputs/.../evaluation.txt`

### 2. Run Idea Agent

Primary entrypoint:

```bash
xcientist idea
```

Override the topic directly:

```bash
xcientist idea --topic "LLM Agent Memory System"
```

The default run uses `src/config/default.yaml`, materializes a run directory under `src/agents/idea_agent/runs/`, and writes `idea_result.json` plus logs.

### 3. Run Experiment Agent

Primary entrypoint:

```bash
xcientist experiment --experiment my_exp --idea-json /abs/path/to/idea_result.json
```

Prepare only:

```bash
xcientist experiment --experiment my_exp --idea-json /abs/path/to/idea_result.json --prepare-only
```

Direct entrypoint:

```bash
python -m src.agents.experiment_agent.main --experiment my_exp --resume --verbose
```

Key workspace outputs live under `workspace/<experiment_id>/` by default and usually include:

- `idea.json`
- `project/`
- `dataset_candidate/`
- `results/`
- `agent_reports/`
- `ablation_results.json`

### 4. Run Paper Agent

Paper Agent is intentionally not part of the default `uv` workflow yet. Keep using the direct/manual entrypoint if you need it.

Recommended entrypoint:

```bash
python -m src.agents.paper_agent.main \
  --experiment my_exp \
  --template-dir src/agents/paper_agent/latex/ICML2025_Template
```

Resume an existing paper workspace:

```bash
python -m src.agents.paper_agent.main \
  --experiment my_exp \
  --template-dir src/agents/paper_agent/latex/ICML2025_Template \
  --resume
```

`run_paper.sh` exists, but it is a local example wrapper with hard-coded values and is not the generic entrypoint.

### 5. Run The Prototype Pipeline

```bash
xcientist pipeline
```

This launches `src.pipeline.run_loop` with `src/config/default.yaml`. It is useful when you want a single command for the integrated loop, but it is still easier to inspect failures agent-by-agent.

## 🧭 Configuration Guide

The current configuration layout is mixed by design:

| Area | Primary source |
|------|----------------|
| Global project config | `src/config/default.yaml` |
| Survey Agent | `survey:` block in `src/config/default.yaml` plus `src/agents/survey_agent/config/*.yaml` |
| Idea Agent | `idea:` block in `src/config/default.yaml` |
| Experiment Agent | `experiment:` block in `src/config/default.yaml` and environment variables |
| Paper Agent | `paper:` block in `src/config/default.yaml` plus CLI flags |
| Pipeline | `pipeline:` block in `src/config/default.yaml` |

If you are starting fresh, edit `src/config/default.yaml` first. It is the most reliable single file to understand current defaults.

## 🤖 Agent Summaries

### Survey Agent

- Main entry: `src/agents/survey_agent/scripts/run_deep_survey.py`
- Purpose: paper collection, clustering, survey drafting, evaluation
- Outputs: survey markdown, survey JSON, evaluation artifacts

### Idea Agent (LigAgent)

- Main entry: `src/agents/idea_agent/run.py`
- Purpose: proposal generation from a topic or mature idea
- Distinctive behavior: survey-grounded retrieval, graph-backed references, domain-locked Memory-Guided MCTS
- Outputs: `idea_result.json`, logs, workflow artifacts

### Experiment Agent (SuperAgent)

- Main entry: `python -m src.agents.experiment_agent.main`
- Purpose: prepare workspace, generate implementation, run code/science iterations, integrate ablation results
- Outputs: experiment workspace and final `ablation_results.json`

### Paper Agent

- Main entry: `python -m src.agents.paper_agent.main`
- Purpose: transform an experiment workspace into a paper-writing workspace
- Outputs: LaTeX project, artifacts, run state, optional compiled PDF

## 📌 Current Project Status

What is stable enough to rely on:

- The four core agent stacks are present and wired together.
- The unified config loader in `src/config/__init__.py` is usable today.
- The root helper scripts exist and reflect the current code layout.

What to keep in mind:

- Some config examples still contain machine-specific absolute paths and should be overridden on a new machine.
- Several advanced workflows depend on local data or models that are not committed to the repository.
- There is no top-level `LICENSE` file in the current repository snapshot.
- There is no obvious root-level automated `tests/` suite in the current repository snapshot.
