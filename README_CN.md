# X-Scientist 使用指南

[English](README.md) | [中文](README_CN.md)

## 概述

X-Scientist 是一个由三个子代理组成的科研辅助系统，支持从想法生成、实验执行到论文撰写的完整研究流程。

## 环境重建

建议优先使用 **conda + environment.yml**，可最大程度复现你当前环境（包含 conda 包与 pip 包）。

### 方法一：conda + environment.yml（推荐）

适用于希望尽量一致复现本项目运行环境的场景。

```bash
conda env create -f environment.yml
conda activate research-agent
```

### 方法二：pip + requirements.txt

适用于仅使用 pip 方式安装依赖的场景（对 conda 二进制依赖的复现能力较弱）。

```bash
conda create -n research-agent python=3.10 -y
conda activate research-agent

# 在项目根目录安装依赖
pip install -r requirements.txt
```

### 4. Pipeline（集成运行器）

**作用**：将 Survey → Idea → Experiment 串联为连续循环，实验结果可反馈给想法生成环节。

**工作流程**：

```
Survey → Idea → Experiment → [将结果转换为符号记忆] → 下一轮
                                                              ↓
                                                       (循环至多 max_iterations 次)
```

**核心功能**：
- **断点续跑**：自动从上次完成的阶段恢复
- **状态管理**：在 `pipeline.yaml` 中跟踪进度
- **符号记忆集成**：将消融结果转换为符号记忆，供未来想法生成使用
- **统一工作空间**：所有输出组织在 `workspace/pipeline_runs/{pipeline_name}/` 下

**输入**（通过 `src/config/default.yaml` 配置）：
- `idea.topic`：研究主题
- `idea.mature_idea`（可选）：契约模式的种子想法
- `pipeline.iterate.max_iterations`：Idea→Experiment 迭代次数
- `pipeline.resume_enabled`：启用/禁用断点续跑

**输出**：
- `workspace/pipeline_runs/{pipeline_name}/survey/`：Survey 输出
- `workspace/pipeline_runs/{pipeline_name}/experiments/{experiment_id}/`：实验工作空间
- `workspace/idea_skill_priors/symbolic_memory.json`：消融结果转换的符号记忆

**用法**：
```bash
python -m src.pipeline.run_loop
# 或
./run_pipeline.sh
```

**配置**（`src/config/default.yaml`）：
```yaml
pipeline:
  name: "diffusion_rl"          # Pipeline 名称（为空则自动生成）
  state_file: "pipeline.yaml"    # 断点续跑状态文件
  resume_enabled: true           # 启用断点续跑
  skip_survey: true             # 已完成 survey 时跳过

  iterate:
    max_iterations: 3           # 迭代次数

  output:
    root: "pipeline_runs"      # 工作空间下的输出根目录
```

---

## 三个 Agent

### 1. Idea Agent（想法生成）

**作用**：通过 Memory-Guided MCTS 流水线，从给定研究主题中生成创新性研究想法。

**工作流程**：

Agent 运行固定的多轮循环。第 1 轮固定执行 `knowledge_aquisition`，后续轮次由 LLM 从动作空间中自主选择：

| 动作 | 作用 |
|------|------|
| `knowledge_aquisition` | Semantic Scholar 种子检索 → 生成 RAG 查询 → OutcomeRAG 检索 → 引用扩展 → 丰富并筛选论文 |
| `advanced_analysis` | LLM 从精选论文中提取关键方法、痛点与开放问题 |
| `idea_generation` | Memory-Guided MCTS 搜索产出最优想法；触发 `persist_final_idea` |
| `idea_evaluation` | 对最新想法进行独立 LLM 评分 |
| `re_analysis_replan` | 当前方向耗尽时，扩展 topic 与检索关键词 |

**输入**：
- `run.topics`：研究主题列表（在 `src/agents/idea_agent/config/run/default.yaml` 中配置）
- `run.mature_idea`（可选）：开启 **Contract 模式** — MCTS 根节点由成熟想法派生，所有扩展在其机制范围内进行

**输出**：
- `runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json`：最终想法（含 title、abstract、introduction、algorithm、reference_papers、mcts_evolution）
- `runs/<topic-slug>-<timestamp>-<uuid>/logs/ligagent.log`：完整运行日志

**配置位置**：`src/agents/idea_agent/config/`（推荐直接改 YAML）

主要配置文件：
- `src/agents/idea_agent/config/run/default.yaml` — 运行时参数：**topics / max_turns / parallelism / rag_config / mature_idea**
- `src/agents/idea_agent/config/mcts/default.yaml` — MCTS：**max_iterations / max_depth / branching_factor / generation_model / evaluation_model**

**运行时配置（`run/default.yaml`）**：
```yaml
run:
  topics:
    - "Diffusion Models for Reinforcement Learning in Games"
  max_turns: 4          # 每个 topic 最多运行轮次
  parallelism: 1        # 并发 worker 数（1 = 串行）
  output_root: "runs"   # 相对于 idea_agent 根目录
  console_logs: true
  rag_config: "src/agents/survey_agent/config/outcomeRAG.yaml"
  # 可选：开启 Contract 模式
  # mature_idea: "..."

  # API 凭据
  openai_api_key: "your-api-key"
  openai_base_url: "https://api.example.com/v1"
  s2_api_key: "your-s2-key"
  s2_api_timeout: "60"
  serper_api_key: "your-serper-key"
  mineru_model_source: "modelscope"
```

**MCTS 配置（`mcts/default.yaml`）**：
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
  min_confidence_for_memory: 0.6   # LTM 写回置信度门控
```

**RAG 配置**：`run.rag_config` 必须指向有效的 OutcomeRAG 配置，且其中的 `save_path` / `save_json_path` 须指向已生成的 survey 输出。

> LigAgent 内部机制（Artifact 结构、MCTS 详解、persist_final_idea 流程、LTM 等）详见 `src/agents/idea_agent/README.md`。

**用法**：
```bash
./run_idea.sh
# 或
python src/agents/idea_agent/run.py
```

---

### 2. Experiment Agent（实验执行）

**作用**：执行实验任务，运行代码并生成实验结果。

**输入**：
- `--experiment`：实验ID（workspace下的目录名）
- `--idea-json`：Idea输出的JSON文件路径
- `--prepare`（可选）：准备阶段，包括克隆仓库等

**输出**：
- `src/agents/experiment_agent/workspaces/{experiment_id}/idea.json`：复制后的想法输入
- `src/agents/experiment_agent/workspaces/{experiment_id}/project/`：生成的项目代码
- `src/agents/experiment_agent/workspaces/{experiment_id}/logs/`：运行日志
- `--idea-json` 指定的文件：实验结果（JSON格式，由脚本生成）

**处理流程**：
1. 在 `workspaces/` 下创建 `{experiment_id}` 目录
2. 复制 `idea-json` 到目录中，命名为 `idea.json`
3. `prepare` 和 `main` 只需 `experiment` 参数，自动读取 `idea.json`

**环境变量配置**：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `OPENAI_API_BASE` | OpenAI API Base URL | - |
| `SHOW_LLM_REASONING` | 显示LLM推理过程 | 1 |
| `EXPERIMENT_AGENT_MEMORY_ENABLED` | 启用记忆功能 | 1 |
| `EXPERIMENT_AGENT_MEMORY_WRITEBACK` | 记忆写回 | 1 |
| `EXPERIMENT_AGENT_MEMORY_TOOL_LOGS` | 记录工具日志 | 0 |
| `AGENT_BASH_TIMEOUT_SECONDS` | Bash超时(毫秒) | 600000 |
| `CODEAGENT_WORKSPACES_DIR` | 工作空间根目录 | `workspaces/` |

**模型配置**（在 `src/agents/experiment_agent/shared/utils/config.py` 中）：

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

**用法**：
```bash
# 准备并运行实验（idea-json会被复制到workspaces/{experiment}/idea.json）
./run_experiment.sh --experiment my_exp --idea-json /path/to/idea_result.json --prepare

# 仅运行实验（不准备）
./run_experiment.sh --experiment my_exp --idea-json /path/to/idea_result.json

# 恢复实验
./run_experiment.sh --experiment my_exp --idea-json /path/to/idea_result.json
```

---

### 3. Paper Agent（论文撰写）

**作用**：根据实验结果撰写学术论文。

**输入**：
- `--experiment`：实验ID
- `--resume`：恢复之前的工作
- `--template-dir`：LaTeX模板目录

**输出**：
- `paper_agent/workspaces/{run_name}/`：论文LaTeX文件
- 编译后的PDF

**环境变量配置**：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `OPENAI_API_KEY` | OpenAI API Key | - |
| `OPENAI_API_BASE` | OpenAI API Base URL | - |
| `PAPER_AGENT_ENABLE_TRACING` | 启用追踪 | 0 |
| `PAPER_AGENT_BASH_TIMEOUT_SECONDS` | Bash超时(秒) | 600 |
| `PAPER_COMPILE_DOCKER_IMAGE` | LaTeX编译镜像 | texlive/texlive:latest |
| `PAPER_AGENT_WORKSPACES_DIR` | 论文工作空间 | `paper_agent/workspaces/` |
| `EXPERIMENT_AGENT_WORKSPACES_DIR` | 实验工作空间 | `experiment_agent/workspaces/` |

**模型配置**（在 `src/agents/paper_agent/utils/config.py` 中）：

```python
PAPER_ARCHITECT_MODEL = "MiniMax-M2.1"
PAPER_WRITER_MODEL = "MiniMax-M2.1"
PAPER_REVIEWER_MODEL = "MiniMax-M2.1"
PAPER_ANALYSIS_MODEL = "MiniMax-M2.1"
PAPER_LITERATURE_MODEL = "MiniMax-M2.1"
PAPER_VIZ_MODEL = "MiniMax-M2.1"
PAPER_VLM_MODEL = "gpt-4o"
```

**用法**：
```bash
./run_paper.sh
```

---

## 目录结构

```
ResearchAgent/
├── run_idea.sh           # Idea Agent入口
├── run_experiment.sh     # Experiment Agent入口
├── run_paper.sh          # Paper Agent入口
├── src/agents/
│   ├── idea_agent/
│   │   ├── config/            # Idea Agent配置（run/mcts/search/agent等）
│   │   ├── run.py             # Idea Agent入口
│   │   └── runs/              # 输出：runs/<slug-timestamp-uuid>/{idea_result.json,logs/}
│   ├── experiment_agent/
│   │   ├── shared/utils/config.py  # 全局配置
│   │   ├── workspaces/        # 实验工作空间
│   │   └── layers/
│   └── paper_agent/
│       ├── utils/config.py    # 论文配置
│       ├── latex/ICML2025_Template/  # 论文模板
│       └── workspaces/        # 论文工作空间
```

---

## 工作流程串联

```
1. Idea Agent
   输入: 研究主题 + 背景知识
   输出: runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json
   
   ↓ (idea_result.json → workspaces/{exp}/idea.json)

2. Experiment Agent
   输入: --experiment + --idea-json
   输出: 实验结果 (JSON) + 项目代码
   
   ↓ (Experiment输出作为Paper输入)

3. Paper Agent
   输入: 实验结果 + 论文模板
   输出: 完整论文 (LaTeX + PDF)
```

---

## 完整示例

```bash
# Step 1: 生成研究想法
./run_idea.sh
# 输出: src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json

# Step 2: 执行实验（自动将idea_result.json复制到workspace/my_exp/idea.json）
./run_experiment.sh --experiment my_exp --idea-json src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json --prepare

# Step 3: 撰写论文
./run_paper.sh
```

## 注意事项

1. **API Key配置**：确保 `run/default.yaml` 中的 API Key 已正确设置
2. **RAG 输出依赖**：使用 OutcomeRAG 时需保证 `save_path` / `save_json_path` 指向已有 survey 输出
3. **记忆功能**：Experiment Agent的记忆功能默认开启，可通过环境变量关闭
4. **超时设置**：根据实验复杂度调整 `AGENT_BASH_TIMEOUT_SECONDS`
5. **模板路径**：Paper Agent使用ICML2025模板，路径可自定义
