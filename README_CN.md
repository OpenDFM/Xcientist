# Research Agent 使用指南

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

如果环境名冲突，可指定新名称：

```bash
conda env create -f environment.yml -n research-agent-copy
conda activate research-agent-copy
```

### 方法二：pip + requirements.txt

适用于仅使用 pip 方式安装依赖的场景（对 conda 二进制依赖的复现能力较弱）。

```bash
conda create -n research-agent python=3.10 -y
conda activate research-agent

# 在项目根目录安装依赖
pip install -r requirements.txt
```

## 三个 Agent

### 1. Idea Agent（想法生成）

**作用**：基于给定主题和背景知识，生成创新性研究想法。

**输入**：
- `run.topics`：研究主题列表（在 `src/agents/idea_agent/config/run/default.yaml` 中配置）
- `run.mature_idea`（可选）：若提供，将触发 **Contract 模式**，MCTS 根节点由 mature idea 派生并受合约约束

**输出**：
- `runs/<topic-时间戳>/idea_result.json`：研究想法JSON结果
- `runs/<topic-时间戳>/logs/ligagent.log`：运行日志

**检索流程（含 RAG）**：
- Semantic Scholar 获取 seed papers → 生成 RAG query → Survey Agent OutcomeRAG 检索 survey 子章节 → 抽取引用标题并反查 paperId → 解析/摘要后写入 memory。

**配置位置**：`src/agents/idea_agent/config/`（推荐直接改 YAML）

最常用的配置在：
- `src/agents/idea_agent/config/run/default.yaml`（运行参数：**topics / parallelism / rag_config** 等）
- `src/agents/idea_agent/config/dataset/default.yaml`（数据集检索/评分参数）
- `src/agents/idea_agent/config/baseline/default.yaml`（baseline 检索/评分参数）
- `src/agents/idea_agent/config/mcts/default.yaml`（搜索策略与模型参数）

**示例：运行参数（最常改）**
```yaml
# src/agents/idea_agent/config/run/default.yaml
run:
  topics:
    - "Diffusion Models for RL in Games"
  parallelism: 1
  rag_config: "src/agents/survey_agent/config/outcomeRAG.yaml"
  max_turns: 4
  output_root: "runs"
  console_logs: false
  # 可选：开启 Contract 模式（成熟想法约束）
  # mature_idea: "..."
```

**示例：API 环境配置**
```yaml
# src/agents/idea_agent/config/run/default.yaml
run:
  openai_api_key: "your-api-key"
  openai_base_url: "https://api.example.com/v1"
  s2_api_key: "your-s2-key"
  s2_api_timeout: 60
  serper_api_key: "your-serper-key"
  mineru_model_source: null
```

**进阶示例：MCTS 搜索参数**
```yaml
# src/agents/idea_agent/config/mcts/default.yaml
mcts:
  max_iterations: 2
  max_depth: 5
  branching_factor: 4
  exploration_constant: 1.2
  generation_model: "gpt-5-mini"
  evaluation_model: "gpt-5.2"
  generation_temperature: 0.7
```

**RAG 配置（Survey Agent OutcomeRAG）**：
- `run.rag_config`：OutcomeRAG 配置路径（必须包含有效的 `save_path` / `save_json_path`，指向已生成的 survey 输出）。
- 示例：`src/agents/survey_agent/config/outcomeRAG.yaml`（请根据实际 survey 输出修改路径）。

**用法**：
```bash
./run_idea.sh
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
│   │   ├── config/            # Idea Agent配置（run/mcts/dataset/baseline等）
│   │   ├── run.py             # Idea Agent入口
│   │   └── runs/              # 输出目录
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
   输出: runs/<topic-时间戳>/idea_result.json
   
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
# 输出: src/agents/idea_agent/runs/<topic-时间戳>/idea_result.json

# Step 2: 执行实验（自动将idea_result.json复制到workspace/my_exp/idea.json）
./run_experiment.sh --experiment my_exp --idea-json src/agents/idea_agent/runs/<topic-时间戳>/idea_result.json --prepare

# Step 3: 撰写论文
./run_paper.sh
```

## 注意事项

1. **API Key配置**：确保 `run/default.yaml` 中的 API Key 已正确设置
2. **RAG 输出依赖**：使用 OutcomeRAG 时需保证 `save_path` / `save_json_path` 指向已有 survey 输出
3. **记忆功能**：Experiment Agent的记忆功能默认开启，可通过环境变量关闭
4. **超时设置**：根据实验复杂度调整 `AGENT_BASH_TIMEOUT_SECONDS`
5. **模板路径**：Paper Agent使用ICML2025模板，路径可自定义
