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

## 三个 Agent

### 1. Idea Agent（想法生成）

**作用**：把研究主题或已有成熟想法转成结构化研究 proposal，核心路径是“检索 → 分析 → Memory-Guided MCTS 搜索 → 持久化”。

**当前实际工作流**：

现在主流程不是旧版的“五动作循环”，而是一个条件分支 workflow：

| 阶段 | 作用 |
|------|------|
| `knowledge_aquisition` | 冷启动检索：Semantic Scholar 种子检索 → OutcomeRAG 查询 → 引用扩展 → 论文丰富与筛选 |
| `advanced_analysis` | 从精选论文中提炼机制、痛点、开放问题与搜索种子 |
| `re_analysis_replan` | 当已有 RAG 上下文时，重写 topic 焦点、成熟想法和检索方向 |
| `idea_generation` | 运行 Memory-Guided MCTS；若开启 `LigAgent-Pro`，则从同一 root 并行搜索所有 preset，并在写出 `idea_result.json` 之前执行 fusion |

当前控制流：
- 冷启动：`knowledge_aquisition -> advanced_analysis -> idea_generation`
- 若 `artifact["rag_hits"]` 已存在：`advanced_analysis -> re_analysis_replan -> idea_generation`

**当前版本 Idea Agent 的几个关键特点**：
- **Contract 模式**：`run.mature_idea` 会直接作为 MCTS 根节点，后续搜索只做机制内 refinement，不做随意漂移。
- **根领域锁定**：MCTS 根节点会先被分类到 1 到 2 个固定研究领域，所有子 idea 都必须留在这些领域内。
- **Preset 驱动搜索**：`mcts.idea_taste_mode` 现在同时影响评估权重、`select_skills()` 的 bias，以及 component 生成时的 taste guidance。
- **LigAgent-Pro**：当 `run."LigAgent-Pro"` 打开时，LigAgent 会从同一份 prepared root context 并行跑五种 idea taste preset，再用 GPT-5.4 fusion agent 融合各 mode 的 best candidate。
- **跨领域理论迁移但不换领域**：`theory-transfer-injection` 可以检索别的领域里的 paper-graph 机制，但实例化时仍必须留在当前 idea 的 home domain。
- **双重记忆系统**：向量记忆负责文本提示，符号记忆负责前瞻式 operator prior 和回顾式 evaluator hints。

**输入**：
- `run.topics`：研究主题列表，定义在 `src/agents/idea_agent/config/run/default.yaml`
- `run."LigAgent-Pro"`：打开多 preset 并行搜索与 fusion
- `run.mature_idea`（可选）：开启 contract-rooted search
- `run.rag_config`：OutcomeRAG 配置路径
- `mcts.idea_taste_mode`：搜索口味 preset，目前支持 `moonshot_inventor`、`bridge_builder`、`steady_engineer`、`ambitious_realist`、`evidence_first`

**输出**：
- 默认写到 `src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json`
  - 包含 `title`、`abstract`、`introduction`、`components`、`algorithm`、`reference_papers`、`mcts_evolution`
  - 若最终 idea 来自 fusion，还会额外包含 `fusion_evolution`
- 默认写到 `src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/logs/ligagent.log`
- 运行中的 `artifact["idea_pool"]` 条目还会保留 `evaluation`、`search_score`、`search_path`、`pareto_candidates`、`search_trace`，以及可选 fusion provenance

**配置位置**：`src/agents/idea_agent/config/`

主要配置文件：
- `src/agents/idea_agent/config/run/default.yaml`
  - `topics`、`LigAgent-Pro`、`output_root`、`rag_config`、`mature_idea`
- `src/agents/idea_agent/config/mcts/default.yaml`
  - `max_iterations`、`max_depth`、`branching_factor`、`idea_taste_mode`
  - `generation_model`、`evaluation_model`
  - `symbolic_memory_path`、`skill_prior_success_threshold`
  - `theory_transfer_retrieval_top_k`、`theory_transfer_similarity_threshold`
- `src/agents/idea_agent/config/fusion/default.yaml`
  - `enabled`、`only_when_ligagent_pro`、`model`、`temperature`、`max_tokens`、`min_candidates`

**最小示例**：
```yaml
run:
  topics:
    - "Diffusion Models for Reinforcement Learning in Games"
  "LigAgent-Pro": false
  output_root: "runs"
  rag_config: "src/agents/survey_agent/config/outcomeRAG.yaml"
  # mature_idea: "可选的 contract root"

mcts:
  max_iterations: 64
  max_depth: 3
  branching_factor: 3
  idea_taste_mode: "evidence_first"
  generation_model: "gpt-5-mini"
  evaluation_model: "gpt-5.4"
  symbolic_memory_path: "output/symbolic_memory.json"

fusion:
  enabled: true
  model: "gpt-5.4"
  min_candidates: 2
```

`run.rag_config` 必须指向有效的 OutcomeRAG 配置，且对应 survey 输出已经存在。

> LigAgent 的当前内部实现，包括 artifact 字段、MCTS 扩展逻辑、preset 机制、root-domain 控制、theory-transfer 检索和持久化细节，详见 `src/agents/idea_agent/README_CN.md`。

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
