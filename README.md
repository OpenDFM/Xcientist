<p align="center">
 <img src="assets/logo.png" alt="Xcientist logo" width="70%">
</p>

<h2 align="center">Externalizing Research Synthesis and Validation in AI Scientists through a Research Harness</h2>

<p align="center">
  <a href="https://kotohanon.github.io/Xcientist/"><img src="https://img.shields.io/badge/Project-Website-4C1?logo=githubpages&logoColor=white" alt="Project website"></a>
  <a href="https://arxiv.org/pdf/2606.18874"><img src="https://img.shields.io/badge/Paper_2606.18874-B31B1B?logo=arxiv&logoColor=white" alt="2606.18874"></a>
    <a href="https://huggingface.co/datasets/KotoHanon/paper-graph-infrastructure"><img src="https://img.shields.io/badge/Dataset-paper--graph--infrastructure-FFD21E?logo=huggingface&logoColor=black" alt="paper-graph-infrastructure dataset"></a>
  <a href="https://www.python.org/downloads/release/python-3120/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12"></a>
</p>

<div align="center">

**[English](README.md) | [简体中文](README_CN.md)**

</div>

<p align="center">
  <img src="assets/overview.jpg" alt="Xcientist system overview" width="100%">
</p>

## Table of Contents

- [Repository Map](#repository-map)
- [How The Pieces Fit Together](#how-the-pieces-fit-together)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Optional Local Assets](#optional-local-assets)
- [Quick Start](#quick-start)
  - [Fastest Path](#fastest-path)
  - [Run Survey Agent](#run-survey-agent)
  - [Run Idea Agent](#run-idea-agent)
  - [Run Experiment Agent](#run-experiment-agent)
  - [Run Blog Agent](#run-blog-agent)
  - [Run The Prototype Pipeline](#run-the-prototype-pipeline)
- [Configuration Guide](#configuration-guide)
- [Citation](#citation)

Xcientist is a multi-agent research workflow for turning a topic into survey artifacts, structured ideas, executable experiments, and technical blog articles. The repository currently centers on four agent stacks:

- `Survey Agent`: collects papers, builds topic clusters, and writes survey outputs.
- `Idea Agent (LigAgent)`: turns a topic or seed idea into a research proposal with survey-grounded retrieval, graph-backed references, and Memory-Guided MCTS.
- `Experiment Agent (SuperAgent)`: prepares a workspace, generates code, runs experiments, and integrates iteration reports.
- `Blog Agent`: reads an experiment workspace and writes a technical blog article with generated figures and quality checks.

The repo also contains a prototype loop runner for `Survey -> Idea -> Experiment -> Blog`, shared configuration, and a reusable memory subsystem.

<a id="repository-map"></a>
## 🗂️ Repository Map

```text
Xcientist/
├── README.md / README_CN.md       # documentation
├── pyproject.toml / uv.lock        # Python package metadata and locked uv environment
├── requirements.txt                # compatibility dependency list
├── run_survey.sh                   # wrapper for `xcientist survey`
├── run_idea.sh                     # wrapper for `xcientist idea`
├── run_experiment.sh               # wrapper for `xcientist experiment`
├── run_blog.sh                     # wrapper for `xcientist blog`
├── run_pipeline.sh                 # wrapper for `xcientist pipeline`
├── scripts/                        # setup and environment helper scripts
│   ├── install_base.sh
│   ├── install_heavy.sh
│   ├── install_mcp_wrappers.sh
│   └── sync_claude_anthropic_env.py
├── src/
│   ├── __main__.py                 # `python -m src` entrypoint
│   ├── cli.py                      # unified `xcientist` CLI
│   ├── config/
│   │   ├── __init__.py             # unified config loader
│   │   └── default.yaml            # main project config
│   ├── pipeline/                   # Survey -> Idea -> Experiment -> Blog loop
│   ├── agents/
│   │   ├── survey_agent/           # paper retrieval, clustering, survey generation
│   │   ├── idea_agent/             # LigAgent proposal generation
│   │   ├── experiment_agent/       # SuperAgent experiment orchestration
│   │   └── blog_agent/             # technical blog generation
│   └── memory/                     # shared vector/symbolic memory APIs
├── graph/                          # graph retrieval service and indexing scripts
├── database/                       # local caches used by retrieval workflows
├── assets/                         # project images and static assets
└── workspace/                      # default runtime workspace, created/used locally
```

<a id="how-the-pieces-fit-together"></a>
## 🔄 How The Pieces Fit Together

```text
Topic
  -> Survey Agent
     output: survey.md + survey.json
  -> Idea Agent
     output: idea_result.json
  -> Experiment Agent
     output: workspace, results, ablation_results.json
  -> Blog Agent
     output: blog workspace, article draft, generated figures
```

The pipeline runner in `src/pipeline/run_loop.py` automates the full `Survey -> Idea -> Experiment -> Blog` flow, but the individual agents remain the clearest way to operate and debug the system.

<a id="prerequisites"></a>
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

<a id="installation"></a>
## ⚙️ Installation

The default setup path is now `uv`.

```bash
git clone --depth 1 https://github.com/OpenDFM/Xcientist.git
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

# PDF parsing stack
uv sync --group pdf

# Blog Agent full workflow: PDF parsing + image generation / OCR / text removal
uv sync --group pdf --group blog

# Full local environment
uv sync --all-groups
```

If you want local MCP wrapper scripts for Experiment Agent:

```bash
xcientist install-mcp-wrappers
```

`environment.yml` is still available as a legacy/full-environment fallback, but `uv sync` is the primary path for `Survey + Idea + Experiment + Blog + Pipeline`. The dependency layout is now split so the default install stays lightweight and heavy local-model / PDF stacks are opt-in.
After activation, the project exposes CLI entrypoints such as `xcientist`, `xcientist-survey`, and `xcientist-idea` directly in the shell.

<a id="environment-variables"></a>
## 🔐 Environment Variables

Different agents read slightly different variables. In practice, these are the most useful ones to define:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export SEMANTIC_SCHOLAR_API_KEY=...
export ANTHROPIC_API_KEY=...
export ANTHROPIC_BASE_URL=...
export SERPER_API_KEY=...
export GITHUB_AI_TOKEN=...
export JINA_API_KEY=...
export TAVILY_API_KEY=...
export HF_TOKEN=...
```

Notes:

- Set both `OPENAI_API_BASE` and `OPENAI_BASE_URL` if you use a custom OpenAI-compatible endpoint.
- The CLI loads repo-root `.env` first and still falls back to `src/config/.env` for older setups.
- `src/config/default.yaml` is the main configuration file for the current unified workflow.
- Survey, Idea, Experiment, and Blog still have some agent-specific conventions on top of the unified config.

<a id="optional-local-assets"></a>
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

<a id="quick-start"></a>
## 🚀 Quick Start

Recommended first-time flow:

```bash
uv sync --group memory --group ml
source .venv/bin/activate
cp .env.example .env
xcientist doctor
```

If doctor passes and your local assets are in place, use the commands below.

<a id="fastest-path"></a>
### Fastest Path

Using the provided `Training-Free Memory System for LLM Agents` example:

Generate survey only:

```bash
xcientist survey --topic "Training-Free Memory System for LLM Agents"
```

Run ideation from the provided sample survey:

```bash
xcientist idea --topic "Training-Free Memory System for LLM Agents"
```

Run experiment from the provided sample idea:

```bash
xcientist experiment --experiment agent_memory --idea-json <repo_root>/src/agents/idea_agent/example/idea_result.json
```

Start blog generation from the sample experiment workspace:

```bash
xcientist blog --experiment agent_memory --source-workspace <repo_root>/workspace/training-free-memory-example
```

For further configuration changes, edit `src/config/default.yaml`.

<a id="run-survey-agent"></a>
### 1. Run Survey Agent

Primary entrypoint:

```bash
xcientist survey
```

Override the topic directly:

```bash
xcientist survey --topic <your_topic_name>
```

Typical outputs:

- `src/agents/survey_agent/outputs/.../survey.md`
- `src/agents/survey_agent/outputs/.../survey.json`
- `src/agents/survey_agent/outputs/.../evaluation.txt`

<a id="run-idea-agent"></a>
### 2. Run Idea Agent

Primary entrypoint:

```bash
xcientist idea
```

Override the topic directly:

```bash
xcientist idea --topic <your_topic_name>
```

The default run uses `src/config/default.yaml`, materializes a run directory under `src/agents/idea_agent/runs/`, and writes `idea_result.json` plus logs.

<a id="run-experiment-agent"></a>
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

<a id="run-blog-agent"></a>
### 4. Run Blog Agent

Blog Agent generates a technical blog article from an existing experiment workspace.

Recommended entrypoint:

```bash
xcientist blog --experiment my_exp
```

With the default workspace root at `<repo_root>/workspace`, this reads the source experiment from:

```bash
<repo_root>/workspace/my_exp
```

You can also pass that experiment workspace explicitly:

```bash
xcientist blog --experiment my_exp --source-workspace <repo_root>/workspace/my_exp
```

If the experiment workspace is not under the blog agent's default source path, pass it explicitly:

```bash
xcientist blog --experiment my_exp --source-workspace /abs/path/to/experiment_workspace
```

Resume an existing blog workspace:

```bash
xcientist blog --experiment my_exp --resume
```

`./run_blog.sh` remains available as a compatibility wrapper and delegates to the same `xcientist blog` command.

<a id="run-the-prototype-pipeline"></a>
### 5. Run The Prototype Pipeline

Run the integrated loop with the topic from `src/config/default.yaml`:

```bash
xcientist pipeline
```

Override the research topic at launch time:

```bash
xcientist pipeline --topic "Training-Free Memory System for LLM Agents"
```

Use a custom config file:

```bash
xcientist pipeline --config /abs/path/to/config.yaml --topic "Your Research Topic"
```

<a id="configuration-guide"></a>
## 🧭 Configuration Guide

The current configuration layout is mixed by design:

| Area | Primary source |
|------|----------------|
| Global project config | `src/config/default.yaml` |
| Survey Agent | `survey:` block in `src/config/default.yaml` plus `src/agents/survey_agent/config/*.yaml` |
| Idea Agent | `idea:` block in `src/config/default.yaml` |
| Experiment Agent | `experiment:` block in `src/config/default.yaml` and environment variables |
| Blog Agent | `blog:` block in `src/config/default.yaml` plus `BLOG_AGENT_SOURCE_WORKSPACE` when needed |
| Pipeline | `pipeline:` block in `src/config/default.yaml` |

If you are starting fresh, edit `src/config/default.yaml` first. It is the most reliable single file to understand current defaults.

<a id="citation"></a>
## Citation

If you use Xcientist in your research, please cite:

```bibtex
@misc{
   to be released
}
```
