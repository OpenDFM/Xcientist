# X-Scientist User Guide

[English](README.md) | [中文](README_CN.md)

## Overview

X-Scientist is a research assistant system with three sub-agents, supporting a full workflow from idea generation to experiment execution and paper writing.

## Environment Rebuild

It is recommended to use **conda + environment.yml** first, which best reproduces the current environment (including both conda and pip packages).

### Method 1: conda + environment.yml (Recommended)

Use this when you want a near-identical runtime environment.

```bash
conda env create -f environment.yml
conda activate research-agent
```

### Method 2: pip + requirements.txt

Use this when you only want pip-based dependency installation (weaker reproducibility for conda binary dependencies).

```bash
conda create -n research-agent python=3.10 -y
conda activate research-agent

# Install dependencies in the project root
pip install -r requirements.txt
```

## Three Agents

### 1. Idea Agent

**Purpose**: turn a topic or mature idea into a structured research proposal through retrieval, literature analysis, and a Memory-Guided MCTS search.

**Actual runtime workflow**:

The main workflow is now a conditional stage graph rather than the old "five-action loop":

| Stage | Role |
|------|------|
| `knowledge_aquisition` | Cold-start retrieval: Semantic Scholar seed → OutcomeRAG query → citation expansion → paper enrichment/filtering |
| `advanced_analysis` | Summarize curated literature into mechanisms, gaps, and search seeds |
| `re_analysis_replan` | Revise topic focus, mature idea, and retrieval direction when RAG context already exists |
| `idea_generation` | Run Memory-Guided MCTS, materialize the best idea, and persist `idea_result.json` |

Current control flow:
- Cold start: `knowledge_aquisition -> advanced_analysis -> idea_generation`
- If `artifact["rag_hits"]` already exists: `advanced_analysis -> re_analysis_replan -> idea_generation`

**What is distinctive about the current Idea Agent**:
- **Contract mode**: `run.mature_idea` turns the provided idea into the MCTS root; descendants refine rather than drift.
- **Root-domain locking**: the MCTS root is classified into one or two fixed research domains; all child ideas must stay in those domains.
- **Preset-driven search**: `mcts.idea_taste_mode` now affects three things at once: evaluation weights, skill selection bias, and component-generation guidance.
- **Cross-domain theory transfer with domain lock**: `theory-transfer-injection` can retrieve external paper-graph mechanisms from other domains, but instantiation is still forced to remain in the idea's home domain.
- **Dual memory guidance**: vector memory provides text snippets, while symbolic memory provides prospective operator priors and retrospective evaluation hints.

**Inputs**:
- `run.topics`: topic list in `src/agents/idea_agent/config/run/default.yaml`
- `run.mature_idea` (optional): enables contract-rooted search
- `run.rag_config`: OutcomeRAG config path
- `mcts.idea_taste_mode`: preset controlling search posture (`moonshot_inventor`, `bridge_builder`, `steady_engineer`, `ambitious_realist`, `evidence_first`)

**Outputs**:
- by default, `src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json`
  - contains `title`, `abstract`, `introduction`, `components`, `algorithm`, `reference_papers`, and `mcts_evolution`
- by default, `src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/logs/ligagent.log`
- in-memory `artifact["idea_pool"]` entries also keep `evaluation`, `search_score`, `search_path`, `pareto_candidates`, and `search_trace`

**Config location**: `src/agents/idea_agent/config/`

Key config files:
- `src/agents/idea_agent/config/run/default.yaml`
  - `topics`, `parallelism`, `output_root`, `rag_config`, `mature_idea`
- `src/agents/idea_agent/config/mcts/default.yaml`
  - `max_iterations`, `max_depth`, `branching_factor`, `idea_taste_mode`
  - `generation_model`, `evaluation_model`
  - `symbolic_memory_path`, `skill_prior_success_threshold`
  - `theory_transfer_retrieval_top_k`, `theory_transfer_similarity_threshold`

**Minimal example**:
```yaml
run:
  topics:
    - "Diffusion Models for Reinforcement Learning in Games"
  parallelism: 1
  output_root: "runs"
  rag_config: "src/agents/survey_agent/config/outcomeRAG.yaml"
  # mature_idea: "Optional contract root"

mcts:
  max_iterations: 64
  max_depth: 3
  branching_factor: 3
  idea_taste_mode: "evidence_first"
  generation_model: "gpt-5-mini"
  evaluation_model: "gpt-5.4"
  symbolic_memory_path: "output/symbolic_memory.json"
```

`run.rag_config` must point to a valid OutcomeRAG config whose saved survey outputs already exist.

> For the current internals of LigAgent, including artifact fields, MCTS expansion, preset handling, root-domain control, theory-transfer retrieval, and persistence, see `src/agents/idea_agent/README.md`.

**Usage**:
```bash
./run_idea.sh
# or
python src/agents/idea_agent/run.py
```

---

### 2. Experiment Agent

**Purpose**: Executes experiment tasks, runs code, and produces experiment outputs.

**Inputs**:
- `--experiment`: experiment ID (directory name under workspace)
- `--idea-json`: path to the idea JSON generated by Idea Agent
- `--prepare` (optional): preparation phase, including repository cloning, etc.

**Outputs**:
- `src/agents/experiment_agent/workspaces/{experiment_id}/idea.json`: copied idea input
- `src/agents/experiment_agent/workspaces/{experiment_id}/project/`: generated project code
- `src/agents/experiment_agent/workspaces/{experiment_id}/logs/`: runtime logs
- file specified by `--idea-json`: experiment result (JSON, generated by scripts)

**Processing flow**:
1. Create `{experiment_id}` under `workspaces/`
2. Copy `idea-json` into that directory as `idea.json`
3. `prepare` and `main` only require `experiment`; they automatically read `idea.json`

**Environment variables**:

| Env Var | Description | Default |
|---------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `OPENAI_API_BASE` | OpenAI API Base URL | - |
| `SHOW_LLM_REASONING` | Show LLM reasoning | 1 |
| `EXPERIMENT_AGENT_MEMORY_ENABLED` | Enable memory | 1 |
| `EXPERIMENT_AGENT_MEMORY_WRITEBACK` | Memory writeback | 1 |
| `EXPERIMENT_AGENT_MEMORY_TOOL_LOGS` | Tool logging | 0 |
| `AGENT_BASH_TIMEOUT_SECONDS` | Bash timeout (ms) | 600000 |
| `CODEAGENT_WORKSPACES_DIR` | Workspace root dir | `workspaces/` |

**Model configuration** (in `src/agents/experiment_agent/shared/utils/config.py`):

```python
# Code Layer
CODE_ARCHITECT_MODEL = "gpt-5.1"
CODE_MANAGER_MODEL = "MiniMax-M2.1"
CODE_WORKER_MODEL = "MiniMax-M2.1"
CODE_INTEGRATOR_MODEL = "MiniMax-M2.1"

# Science Layer
SCIENCE_ARCHITECT_MODEL = "MiniMax-M2.1"
SCIENCE_MANAGER_MODEL = "MiniMax-M2.1"
SCIENCE_WORKER_MODEL = "MiniMax-M2.1"
SCIENCE_INTEGRATOR_MODEL = "MiniMax-M2.1"

# Prepare Layer
PREPARE_AGENT_MODEL = "MiniMax-M2.1"
```

**Usage**:
```bash
# Prepare and run (idea-json will be copied to workspaces/{experiment}/idea.json)
./run_experiment.sh --experiment my_exp --idea-json /path/to/idea_result.json --prepare

# Run only (without prepare)
./run_experiment.sh --experiment my_exp --idea-json /path/to/idea_result.json

# Resume experiment
./run_experiment.sh --experiment my_exp --idea-json /path/to/idea_result.json
```

---

### 3. Paper Agent

**Purpose**: Writes academic papers based on experiment outputs.

**Inputs**:
- `--experiment`: experiment ID
- `--resume`: resume previous work
- `--template-dir`: LaTeX template directory

**Outputs**:
- `paper_agent/workspaces/{run_name}/`: paper LaTeX files
- compiled PDF

**Environment variables**:

| Env Var | Description | Default |
|---------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `OPENAI_API_BASE` | OpenAI API Base URL | - |
| `PAPER_AGENT_ENABLE_TRACING` | Enable tracing | 0 |
| `PAPER_AGENT_BASH_TIMEOUT_SECONDS` | Bash timeout (s) | 600 |
| `PAPER_COMPILE_DOCKER_IMAGE` | LaTeX compile image | texlive/texlive:latest |
| `PAPER_AGENT_WORKSPACES_DIR` | Paper workspace | `paper_agent/workspaces/` |
| `EXPERIMENT_AGENT_WORKSPACES_DIR` | Experiment workspace | `experiment_agent/workspaces/` |

**Model configuration** (in `src/agents/paper_agent/utils/config.py`):

```python
PAPER_ARCHITECT_MODEL = "MiniMax-M2.1"
PAPER_WRITER_MODEL = "MiniMax-M2.1"
PAPER_REVIEWER_MODEL = "MiniMax-M2.1"
PAPER_ANALYSIS_MODEL = "MiniMax-M2.1"
PAPER_LITERATURE_MODEL = "MiniMax-M2.1"
PAPER_VIZ_MODEL = "MiniMax-M2.1"
PAPER_VLM_MODEL = "gpt-4o"
```

**Usage**:
```bash
./run_paper.sh
```

---

## Directory Structure

```
ResearchAgent/
├── run_idea.sh           # Idea Agent entry
├── run_experiment.sh     # Experiment Agent entry
├── run_paper.sh          # Paper Agent entry
├── src/agents/
│   ├── idea_agent/
│   │   ├── config/            # Idea Agent configs (run/mcts/search/agent, etc.)
│   │   ├── run.py             # Idea Agent entry
│   │   └── runs/              # Output: runs/<slug-timestamp-uuid>/{idea_result.json,logs/}
│   ├── experiment_agent/
│   │   ├── shared/utils/config.py  # Global config
│   │   ├── workspaces/        # Experiment workspace
│   │   └── layers/
│   └── paper_agent/
│       ├── utils/config.py    # Paper config
│       ├── latex/ICML2025_Template/  # Paper template
│       └── workspaces/        # Paper workspace
```

---

## End-to-End Workflow

```
1. Idea Agent
	 Input: research topics + background knowledge
	 Output: runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json
   
	 ↓ (idea_result.json → workspaces/{exp}/idea.json)

2. Experiment Agent
	 Input: --experiment + --idea-json
	 Output: experiment results (JSON) + project code
   
	 ↓ (Experiment output as Paper input)

3. Paper Agent
	 Input: experiment results + paper template
	 Output: complete paper (LaTeX + PDF)
```

---

## Full Example

```bash
# Step 1: Generate research ideas
./run_idea.sh
# Output: src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json

# Step 2: Run experiment (automatically copies idea_result.json to workspace/my_exp/idea.json)
./run_experiment.sh --experiment my_exp --idea-json src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json --prepare

# Step 3: Write paper
./run_paper.sh
```

## Notes

1. **API key setup**: make sure API keys in `run/default.yaml` are correctly configured
2. **RAG output dependency**: for OutcomeRAG, `save_path` / `save_json_path` must point to existing survey outputs
3. **Memory feature**: Experiment Agent memory is enabled by default and can be disabled via env vars
4. **Timeout setup**: adjust `AGENT_BASH_TIMEOUT_SECONDS` according to experiment complexity
5. **Template path**: Paper Agent uses an ICML2025 template and the path can be customized
