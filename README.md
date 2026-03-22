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

### 4. Pipeline (Integrated Runner)

**Purpose**: Runs Survey → Idea → Experiment in a continuous loop with feedback from experiment results back to idea generation.

**Workflow**:

```
Survey → Idea → Experiment → [Convert Results to Symbolic Memory] → Next Iteration
                                                                          ↓
                                                                    (loop up to max_iterations)
```

**Key Features**:
- **Resume Support**: Automatically resumes from the last completed phase
- **State Management**: Tracks progress in `pipeline.yaml`
- **Symbolic Memory Integration**: Converts ablation results to symbolic memory for future idea generation
- **Unified Workspace**: All outputs organized under `workspace/pipeline_runs/{pipeline_name}/`

**Inputs** (via `src/config/default.yaml`):
- `idea.topic`: Research topic
- `idea.mature_idea` (optional): Seed idea for contraction mode
- `pipeline.iterate.max_iterations`: Number of Idea→Experiment iterations
- `pipeline.resume_enabled`: Enable/disable resume

**Outputs**:
- `workspace/pipeline_runs/{pipeline_name}/survey/`: Survey outputs
- `workspace/pipeline_runs/{pipeline_name}/experiments/{experiment_id}/`: Experiment workspaces
- `workspace/idea_skill_priors/symbolic_memory.json`: Symbolic memory from ablation results

**Usage**:
```bash
python -m src.pipeline.run_loop
# or
./run_pipeline.sh
```

**Configuration** (`src/config/default.yaml`):
```yaml
pipeline:
  name: "diffusion_rl"          # Pipeline name (auto-generated if empty)
  state_file: "pipeline.yaml"   # State file for resume
  resume_enabled: true           # Enable resume
  skip_survey: true              # Skip survey if already done

  iterate:
    max_iterations: 3           # Number of iterations

  output:
    root: "pipeline_runs"       # Output root under workspace
```

---

## Three Agents

### 1. Idea Agent

**Purpose**: Generates innovative research ideas from given topics via a Memory-Guided MCTS pipeline.

**Workflow**:

The agent runs a fixed multi-turn loop. Turn 1 is always `knowledge_aquisition`; subsequent turns are selected by the LLM from the action space:

| Action | Role |
|--------|------|
| `knowledge_aquisition` | Semantic Scholar seed → RAG query → OutcomeRAG retrieval → citation expansion → enrich & filter papers |
| `advanced_analysis` | LLM extracts key methods, pain points, and open questions from curated papers |
| `idea_generation` | Memory-Guided MCTS search produces the best idea; triggers `persist_final_idea` |
| `idea_evaluation` | Standalone LLM scoring of the latest idea |
| `re_analysis_replan` | Expands topic and retrieval keywords when the current direction is exhausted |

**Inputs**:
- `run.topics`: list of research topics (`src/agents/idea_agent/config/run/default.yaml`)
- `run.mature_idea` (optional): enables **Contract mode** — MCTS root is derived from the mature idea and all expansions stay within its mechanism

**Outputs**:
- `runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json`: final idea (title, abstract, introduction, algorithm, reference_papers, mcts_evolution)
- `runs/<topic-slug>-<timestamp>-<uuid>/logs/ligagent.log`: full run log

**Config location**: `src/agents/idea_agent/config/` (edit YAML directly)

Key config files:
- `src/agents/idea_agent/config/run/default.yaml` — runtime params: **topics / max_turns / parallelism / rag_config / mature_idea**
- `src/agents/idea_agent/config/mcts/default.yaml` — MCTS: **max_iterations / max_depth / branching_factor / generation_model / evaluation_model**

**Runtime config (`run/default.yaml`)**:
```yaml
run:
  topics:
    - "Diffusion Models for Reinforcement Learning in Games"
  max_turns: 4          # max agent turns per topic
  parallelism: 1        # concurrent workers (1 = serial)
  output_root: "runs"   # relative to idea_agent root
  console_logs: true
  rag_config: "src/agents/survey_agent/config/outcomeRAG.yaml"
  # Optional: enable Contract mode
  # mature_idea: "..."

  # API credentials
  openai_api_key: "your-api-key"
  openai_base_url: "https://api.example.com/v1"
  s2_api_key: "your-s2-key"
  s2_api_timeout: "60"
  serper_api_key: "your-serper-key"
  mineru_model_source: "modelscope"
```

**MCTS config (`mcts/default.yaml`)**:
```yaml
mcts:
  max_iterations: 128
  max_depth: 3
  branching_factor: 3
  exploration_constant: 1.15
  generation_model: "gpt-5-mini"
  evaluation_model: "gpt-5.2"
  generation_temperature: 0.7
  evaluation_temperature: 0.001
  min_confidence_for_memory: 0.6   # LTM write-back threshold
```

**RAG config**: `run.rag_config` must point to a valid OutcomeRAG config whose `save_path` / `save_json_path` reference existing survey outputs.

> For a detailed description of LigAgent internals (Artifact structure, MCTS mechanisms, persist_final_idea flow, LTM, etc.), see `src/agents/idea_agent/README.md`.

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
| `EXPERIMENT_AGENT_WORKSPACE_DIR` | Experiment workspace dir | `workspace/<experiment>` |

**Model configuration** (in `src/config/default.yaml` under `experiment.models`):

```python
prepare = "MiniMax-M2.7"
code = "MiniMax-M2.7"
master = "MiniMax-M2.7"
science = "MiniMax-M2.7"
default = "MiniMax-M2.7"

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
