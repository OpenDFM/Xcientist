# X-Scientist

[English](README.md) | [中文](README_CN.md)

## Overview

X-Scientist is the current research-agent system in this repository, the codebase now centers around four core agents :

1. **Survey Agent**: collects papers, builds literature clusters, writes survey drafts, and produces `survey.md` / `survey.json` artifacts for downstream retrieval.
2. **Idea Agent (LigAgent)**: turns a topic or a mature seed idea into a structured research proposal through survey-grounded retrieval, graph-backed Core references, analysis, and Memory-Guided MCTS.
3. **Experiment Agent (SuperAgent)**: converts an idea into runnable code, executes experiments, and iterates on failures.
4. **Paper Agent**: reads an experiment workspace and writes a paper workspace with LaTeX drafts and compilation artifacts.

Shared subsystems that matter in the current repo:

- `src/config/default.yaml`: the current unified config file used directly by Idea Agent and referenced by the pipeline code.
- `src/memory/`: shared memory subsystem used by parts of Idea Agent / Experiment Agent.
- `src/pipeline/`: prototype loop code for Survey -> Idea -> Experiment, but not the primary documented entrypoint right now.

## Environment Rebuild

It is still recommended to use **conda + environment.yml** first, because that best matches the current repository state.

### Method 1: conda + environment.yml (Recommended)

```bash
conda env create -f environment.yml
conda activate research-agent
```

### Method 2: pip + requirements.txt

```bash
conda create -n research-agent python=3.10 -y
conda activate research-agent
pip install -r requirements.txt
```

Notes:

- `requirements.txt` includes editable installs for `src/memory` and `src/agents`.
- Paper compilation currently expects a local TeX toolchain such as `tectonic`, `latexmk`, or `pdflatex`.
- If you use a custom OpenAI-compatible endpoint, set both `OPENAI_BASE_URL` and `OPENAI_API_BASE`, because different agents currently read different variable names.

## Local Asset Setup

Current LigAgent usage depends on several local assets in addition to the Python environment.

Prepare the local environment in this order:

1. Download the processed graph package from the shared Google Drive folder and copy it into `<repo_root>/data/processed/`.

Google Drive:
`https://drive.google.com/drive/folders/1lH1MI6gk7eh0HfvfOajcqAZg3n95v5BK?usp=drive_link`

Keep the directory structure intact. The current graph-backed retrieval path expects at least:

- `data/processed/graph.db`
- `data/processed/core_component_summary_vector_store/faiss.index`
- `data/processed/core_component_summary_vector_store/meta.json`

2. Download the embedding model `BAAI/bge-m3` into `<repo_root>/models/bge-m3/`.

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
```

This matches the current defaults in `graph/index_core_component_summaries.py` and the graph-vector-store loader used by LigAgent.

3. Start the local graph engine with FastAPI + Uvicorn from the `<repo_root>/graph/`.

```bash
uvicorn graph.server:app --host 127.0.0.1 --port 8000
```

The server entrypoint is `graph/server.py`. A quick health check is:

```bash
curl http://127.0.0.1:8000/health
```

Important implementation detail:

- `graph/server.py` reads the graph database from `<repo_root>/data/processed/graph.db`.
- The prebuilt Core-component vector store is expected under `<repo_root>/data/processed/core_component_summary_vector_store/`.
- The default local embedding-model path used by the graph indexing code is `<repo_root>/models/bge-m3/`.

## Configuration Layout

The repository uses a mixed configuration layout.

| Area | Current source of truth | Notes |
|------|-------------------------|-------|
| Idea Agent | `src/config/default.yaml` -> `idea:` | This is the default config loaded by `src/agents/idea_agent/utils/core/config_loader.py`. You can override the file via `IDEA_AGENT_CONFIG=/path/to/config.yaml`. |
| Survey Agent | `src/agents/survey_agent/config/*.yaml` | Standalone Survey runs still use Hydra configs with `BasicInfo`, `APIInfo`, and `ModuleInfo`. |
| Experiment Agent | environment variables + `src/agents/experiment_agent/shared/utils/config.py` | Runtime behavior is mostly controlled by env vars and code constants. |
| Paper Agent | CLI args + environment variables + `src/agents/paper_agent/utils/config.py` | `--template-dir` and workspace env vars are the main knobs. |
| Pipeline prototype | `src/config/default.yaml` -> `pipeline:` | The code exists, but there is no `run_pipeline.sh` in the repository root. |

Important practical detail:

- Several shipped Survey / Paper / unified config files still contain machine-specific absolute example paths. Treat them as examples and override them before running on a new machine.

## Four Agents

### 1. Survey Agent

**Purpose**: build a literature survey around a topic and persist structured survey outputs that can later feed OutcomeRAG and Idea Agent.

**Actual runtime workflow**:

```
Seed paper collection
-> reference / citation expansion
-> paper reading and keynotes
-> clustering
-> intra-cluster analysis
-> inter-cluster analysis
-> outline generation
-> survey drafting
-> review / revise
-> survey + references export
-> evaluation
```

**Standalone config locations**:

- `src/agents/survey_agent/config/deep_survey.yaml`
- `src/agents/survey_agent/config/deep_survey_xiaomi.yaml`
- `src/agents/survey_agent/config/outcomeRAG.yaml`

**Key inputs**:

- `BasicInfo.topic`
- `BasicInfo.save_path`
- `BasicInfo.save_json_path`
- `BasicInfo.evaluation_save_path`
- `APIInfo.llm_api_key`
- `APIInfo.llm_api_base_url`
- `ModuleInfo.WorkCollector.*`
- `ModuleInfo.WorkAnalyzer.*`
- `ModuleInfo.SurveyGenerator.*`

**Outputs**:

- `BasicInfo.save_path`: generated survey markdown
- `BasicInfo.save_json_path`: structured survey JSON
- `BasicInfo.evaluation_save_path`: evaluation text output
- cache/database artifacts under `BasicInfo.cache_path`

**Recommended usage**:

The bundled YAMLs contain example absolute output paths, so the safest pattern is to override the output paths on the command line:

```bash
python -m src.agents.survey_agent.scripts.run_deep_survey \
  --config-path src/agents/survey_agent/config \
  --config-name deep_survey \
  BasicInfo.topic="LLM Agent Memory System" \
  BasicInfo.save_path="src/agents/survey_agent/outputs/llm_agent_memory_system.md" \
  BasicInfo.save_json_path="src/agents/survey_agent/outputs/llm_agent_memory_system.json" \
  BasicInfo.evaluation_save_path="src/agents/survey_agent/outputs/llm_agent_memory_system_eval.txt"
```

**How it connects to Idea Agent**:

- Idea Agent's `idea.run.rag_config` usually points to `src/agents/survey_agent/config/outcomeRAG.yaml`.
- That OutcomeRAG config must in turn point to survey outputs that actually exist on your machine.

---

### 2. Idea Agent (LigAgent)

**Purpose**: turn a topic or mature idea into a structured research proposal through survey-grounded retrieval, graph-backed Core references, structured analysis, and a Memory-Guided MCTS search.

**Actual runtime workflow**:

The main workflow is a conditional stage graph:

| Stage | Role |
|------|------|
| `knowledge_aquisition` | Cold-start retrieval: query generation -> OutcomeRAG over survey sections -> graph.db Core retrieval -> reference selection |
| `advanced_analysis` | Summarize survey excerpts and selected Core references into mechanisms, gaps, and search seeds |
| `re_analysis_replan` | Revise topic focus, mature idea, and retrieval direction when RAG context already exists |
| `idea_generation` | Run Memory-Guided MCTS; if `LigAgent-Pro` is enabled, search all presets from one shared root and fuse them before persisting `idea_result.json` |

Current control flow:

- Cold start: `knowledge_aquisition -> advanced_analysis -> idea_generation`
- If `artifact["rag_hits"]` already exists: `advanced_analysis -> re_analysis_replan -> idea_generation`

**What is distinctive about the current Idea Agent**:

- **Survey-grounded retrieval with graph-backed references**: OutcomeRAG supplies survey sections and citation titles, while downstream references come from `graph.db` Core nodes rather than parsed papers.
- **Contract mode**: `idea.run.mature_idea` turns the provided idea into the MCTS root.
- **Root-domain locking**: the MCTS root is classified into one or two fixed research domains and descendants must stay there.
- **Preset-driven search**: `idea.mcts.idea_taste_mode` affects evaluation weights, skill selection bias, and component-generation guidance.
- **LigAgent-Pro**: when `idea.run.LigAgent-Pro` is enabled, LigAgent runs all five idea taste presets from one prepared root and fuses them.
- **Cross-domain theory transfer with domain lock**: external mechanisms can be borrowed, but the final instantiated idea must stay in the home domain.
- **Dual memory guidance**: vector memory provides text snippets, while symbolic memory provides operator priors and evaluation hints.

**Config source**:

- `src/config/default.yaml` under the `idea:` section
- optional override: `IDEA_AGENT_CONFIG=/path/to/another_config.yaml`

Idea Agent loads its default settings from the unified config rather than from a separate `src/agents/idea_agent/config/...` directory.

**Key inputs**:

- `idea.run.topics`
- `idea.run.LigAgent-Pro`
- `idea.run.mature_idea` (optional)
- `idea.run.rag_config`
- `idea.agent.model`
- `idea.mcts.*`
- `idea.fusion.*`

**Outputs**:

- by default, `src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json`
- by default, `src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/logs/ligagent.log`

**Minimal config example**:

```yaml
idea:
  run:
    topics:
      - "LLM Agent Memory System"
    LigAgent-Pro: true
    output_root: "runs"
    rag_config: "src/agents/survey_agent/config/outcomeRAG.yaml"
    # mature_idea: "Optional contract root"

  agent:
    model: "gpt-5-mini"

  mcts:
    max_iterations: 64
    max_depth: 3
    branching_factor: 3
    idea_taste_mode: "moonshot_inventor"
    generation_model: "gpt-5-mini"
    evaluation_model: "gpt-5.2"
    symbolic_memory_path: "output/symbolic_memory.json"

  fusion:
    enabled: true
    model: "gpt-5.2"
    min_candidates: 2
```

**Usage**:

```bash
./run_idea.sh
# or
python src/agents/idea_agent/run.py
```

This runner is config-driven and does not take `--topic` directly in the standalone entrypoint.

> For current LigAgent internals, including artifact fields, MCTS expansion, preset handling, root-domain control, theory-transfer retrieval, and persistence, see `src/agents/idea_agent/README.md`.

---

### 3. Experiment Agent (SuperAgent)

**Purpose**: materialize an idea into a runnable project, execute experiments, and iterate between code generation and scientific validation.

**Actual runtime phases**:

1. **Prepare phase**: creates the workspace, writes `idea.md`, clones reference repos, and downloads candidate datasets.
2. **Engineering layer**: builds the code project.
3. **Science layer**: executes experiments and records results / feedback.

**Primary entrypoints**:

- `python -m src.agents.experiment_agent.prepare`
- `python -m src.agents.experiment_agent.main`
- convenience wrapper: `./run_experiment.sh`

**Inputs**:

- `--experiment`: experiment ID / workspace name
- `--idea-json`: required by `run_experiment.sh`
- `--prepare`: optional wrapper flag to run prepare first
- `--resume`: supported by the main module
- `--fresh`: force a fresh main run

**Workspace outputs**:

Under `src/agents/experiment_agent/workspaces/<experiment_id>/` the runtime may create:

- `idea.json`: copied source idea
- `idea.md`: markdown proposal materialized by PrepareAgent
- `project/`: generated project code
- `repos/`: cloned reference repositories
- `dataset_candidate/`: downloaded candidate datasets
- `specs/`: generated specs / plans / reports mirrored for compatibility
- `cached/`: checkpoint and state data
- `logs/`: workspace-level logs

Important downstream result locations typically live under:

- `project/result/code/iter_v*/`
- `project/result/science/iter_v*/`

**Environment variables commonly used**:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `MINIMAX_API_KEY`
- `SERPER_API_KEY`
- `XIAOMI_API_KEY`
- `CODEAGENT_WORKSPACES_DIR`
- `EXPERIMENT_AGENT_MEMORY_ENABLED`
- `EXPERIMENT_AGENT_MEMORY_WRITEBACK`
- `EXPERIMENT_AGENT_MEMORY_TOOL_LOGS`
- `EXPERIMENT_AGENT_MEMORY_PROMPT_INJECTION`
- `AGENT_BASH_TIMEOUT_SECONDS`

**Current model / runtime constants**:

- defined in `src/agents/experiment_agent/shared/utils/config.py`
- note that the shell wrapper and the Python module may behave differently because the wrapper exports its own env defaults

**Recommended usage**:

Using the wrapper:

```bash
./run_experiment.sh --experiment my_exp --idea-json /path/to/idea_result.json --prepare
```

Using the underlying modules directly:

```bash
mkdir -p src/agents/experiment_agent/workspaces/my_exp
cp /path/to/idea_result.json src/agents/experiment_agent/workspaces/my_exp/idea.json

python -m src.agents.experiment_agent.prepare --experiment my_exp --force --clone-depth 1 --verbose
python -m src.agents.experiment_agent.main --experiment my_exp --resume --verbose
```

Current wrapper behavior to be aware of:

- `run_experiment.sh` disables Experiment Agent memory by default via environment variables.
- the wrapper copies `idea.json` only when `--prepare` is used.
- the wrapper always invokes the main phase with `--resume`.

---

### 4. Paper Agent

**Purpose**: read an Experiment Agent workspace and generate a separate paper workspace containing specs, LaTeX files, compilation logs, and paper-writing artifacts.

**Current inputs**:

- `--experiment`: experiment workspace name from Experiment Agent
- `--template-dir`: LaTeX template directory
- `--resume`: resume an existing paper workspace

**Current workspace outputs**:

Under `src/agents/paper_agent/workspaces/<experiment_id>/` the runtime creates:

- `paper/`: copied or initialized LaTeX project
- `artifacts/`: compile logs, PDF pages, extracted assets, sub-agent artifacts
- `specs/`: paper-side spec / plan / constitution files
- `state/paper_state.json`: resumable run state

**How Paper Agent reads experiment results**:

It resolves input from `src/agents/experiment_agent/workspaces/<experiment_id>/` and looks for:

- `idea.md`
- `specs/`
- `project/`
- result files such as `project/result/science/iter_v*/result_summary.json`

**Environment variables commonly used**:

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `MINIMAX_API_KEY`
- `PAPER_AGENT_WORKSPACES_DIR`
- `EXPERIMENT_AGENT_WORKSPACES_DIR`
- `PAPER_AGENT_BASH_TIMEOUT_SECONDS`
- `PAPER_AGENT_ENABLE_TRACING`

**Compilation detail**:

- PDF compilation currently looks for local `tectonic`, `latexmk`, or `pdflatex`.
- If none of them are available, LaTeX compilation will fail even though the paper workspace is created.

**Recommended usage**:

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

Important note:

- `run_paper.sh` is currently a local example wrapper with a hard-coded experiment name and template path. It is not the recommended generic entrypoint.

## Pipeline Status

The repository still contains a Survey -> Idea -> Experiment loop under `src/pipeline/run_loop.py`, and `src/config/default.yaml` still has a `pipeline:` block.

The repository root does not include:

```bash
./run_pipeline.sh
```

This command is not available in the current repository because:

- `run_pipeline.sh` does not exist in the current repo root
- `src/pipeline/run_loop.py` imports `load_config` from `src.config`, but `src/config/__init__.py` does not currently export that function

So the reliable workflow today is to run the four agents individually:

```
Survey -> Idea -> Experiment -> Paper
```

## Directory Structure

```text
ResearchAgent/
├── README.md
├── README_CN.md
├── environment.yml
├── requirements.txt
├── run_idea.sh
├── run_experiment.sh
├── run_paper.sh                  # local example wrapper, not the recommended generic launcher
├── src/
│   ├── config/
│   │   └── default.yaml          # current unified config used by Idea Agent / pipeline prototype
│   ├── agents/
│   │   ├── survey_agent/
│   │   │   ├── config/           # Hydra configs for standalone survey runs
│   │   │   ├── scripts/
│   │   │   ├── modules/
│   │   │   └── outputs/          # example survey outputs
│   │   ├── idea_agent/
│   │   │   ├── run.py
│   │   │   ├── agent/
│   │   │   ├── utils/
│   │   │   └── runs/             # runtime output root
│   │   ├── experiment_agent/
│   │   │   ├── prepare.py
│   │   │   ├── main.py
│   │   │   ├── layers/
│   │   │   ├── shared/
│   │   │   └── workspaces/       # runtime-generated; may not exist before the first run
│   │   └── paper_agent/
│   │       ├── main.py
│   │       ├── entry.py
│   │       ├── latex/ICML2025_Template/
│   │       └── workspaces/       # runtime-generated; may not exist before the first run
│   ├── memory/
│   └── pipeline/
└── tests/
```

## End-to-End Workflow

```text
1. Survey Agent
   Input: research topic
   Output: survey.md + survey.json

   ↓ (Survey output is referenced by OutcomeRAG config)

2. Idea Agent
   Input: idea.run.topics / idea.run.mature_idea
   Output: src/agents/idea_agent/runs/<slug-timestamp-uuid>/idea_result.json

   ↓ (idea_result.json -> experiment workspace idea.json)

3. Experiment Agent
   Input: experiment workspace + idea.json
   Output: generated project + experiment results + specs / reports

   ↺ Experiment feedback / result summary can be fed back into Idea Agent
     for the next proposal iteration

   Loop: Idea Agent <-> Experiment Agent
   Repeat until the idea / implementation is ready for paper writing

   ↓ (experiment workspace becomes Paper Agent input)

4. Paper Agent
   Input: experiment workspace + LaTeX template
   Output: paper workspace with LaTeX, artifacts, and optional compiled PDF
```

## Full Example

```bash
# Step 1: generate survey artifacts
python -m src.agents.survey_agent.scripts.run_deep_survey \
  --config-path src/agents/survey_agent/config \
  --config-name deep_survey \
  BasicInfo.topic="LLM Agent Memory System" \
  BasicInfo.save_path="src/agents/survey_agent/outputs/llm_agent_memory_system.md" \
  BasicInfo.save_json_path="src/agents/survey_agent/outputs/llm_agent_memory_system.json" \
  BasicInfo.evaluation_save_path="src/agents/survey_agent/outputs/llm_agent_memory_system_eval.txt"

# Step 2: generate research ideas from unified config
./run_idea.sh

# Step 3: run experiment
./run_experiment.sh --experiment my_exp --idea-json src/agents/idea_agent/runs/<topic-run>/idea_result.json --prepare

# Step 4: write paper
python -m src.agents.paper_agent.main \
  --experiment my_exp \
  --template-dir src/agents/paper_agent/latex/ICML2025_Template
```

## Notes

1. **Update machine-specific paths first**: the repository still contains example absolute paths in several YAML / Python config files.
2. **Prefer environment variables for secrets**: API keys and base URLs are safer in env vars than in committed config files.
3. **Survey output paths matter**: OutcomeRAG and Idea Agent only work if the survey files referenced by the RAG config actually exist.
4. **Runtime workspaces are created on demand**: missing `workspaces/` directories in `experiment_agent` or `paper_agent` before the first run are normal.
5. **Use direct module commands when in doubt**: they reflect the current code paths most directly.
