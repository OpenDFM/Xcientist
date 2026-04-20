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

Xcientist 是一个面向科研流程的多 Agent 系统，目标是把一个研究主题逐步推进为综述材料、结构化 idea、可执行实验，以及论文草稿。当前仓库主要由四个核心 Agent 组成：

- `Survey Agent`：检索论文、构建主题聚类并生成 survey 结果。
- `Idea Agent（LigAgent）`：基于 survey 检索、图谱 reference 和 Memory-Guided MCTS，把主题或成熟想法转成研究 proposal。
- `Experiment Agent（SuperAgent）`：准备实验工作空间、生成代码、执行实验，并整合迭代结果。
- `Paper Agent`：读取实验工作空间，生成论文工作空间、LaTeX 草稿和编译产物。

仓库中还包含一个 `Survey -> Idea -> Experiment` 的原型 pipeline、统一配置，以及可复用的 memory 子系统。

## 🗂️ 仓库结构

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
├── graph/                         # 图检索服务与索引脚本
├── scripts/                       # 工具脚本，例如 MCP wrapper 安装
├── skills/                        # 仓库内使用的本地 Codex skill
└── src/
    ├── config/default.yaml        # 统一项目配置
    ├── agents/
    │   ├── survey_agent/
    │   ├── idea_agent/
    │   ├── experiment_agent/
    │   └── paper_agent/
    ├── memory/                    # 共享 memory 包
    └── pipeline/                  # 端到端 loop runner
```

## 🔄 整体流程

```text
研究主题
  -> Survey Agent
     输出：survey.md + survey.json
  -> Idea Agent
     输出：idea_result.json
  -> Experiment Agent
     输出：workspace、results、ablation_results.json
  -> Paper Agent
     输出：paper workspace、LaTeX 草稿、可选 PDF
```

`src/pipeline/run_loop.py` 可以自动串起前三步，但如果你需要定位问题，逐个 Agent 运行通常更直观。

## ✅ 环境要求

- `uv`
- Python `3.12`
- `Experiment Agent` 需要 `node` 和 `npx` 来启动 MCP server
- 运行不同 Agent 所需的 API key
- 如果要手工使用 `Paper Agent`，还需要本地 TeX 工具链，例如 `tectonic`、`latexmk` 或 `pdflatex`
- 图检索和 memory 能力依赖仓库外的本地数据或模型
   - 论文图相关资源的[下载链接](https://drive.google.com/drive/folders/1lH1MI6gk7eh0HfvfOajcqAZg3n95v5BK?usp=drive_link)，将它们放入 `<repo_root>/data/processed`
   - 向量模型下载：
   ```
   mkdir -p models/bge-m3
   mkdir -p models/all-MiniLM-L6-v2
   modelscope download -- model baai/bge-m3 --local_dir <repo_root>/models/bge-m3
   modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --local_dir <repo_root>/models/all-MiniLM-L6-v2
   ```

## ⚙️ 安装

推荐使用 `uv`：

```bash
uv sync
source .venv/bin/activate
cp .env.example .env
xcientist doctor
```

常见分组安装方式：

```bash
# 仅安装基础 CLI / 配置 / API 工作流
uv sync

# 安装 memory 与本地模型相关能力
uv sync --group memory --group ml

# 安装 Paper / PDF 解析相关依赖
uv sync --group paper

# 安装完整本地环境
uv sync --all-groups
```

如果你希望给 `Experiment Agent` 预先安装本地 MCP wrapper：

```bash
xcientist install-mcp-wrappers
```

`environment.yml` 仍然保留，作为兼容旧环境或全量环境的备选方案；但 `Survey + Idea + Experiment + Pipeline` 的主路径现在是 `uv sync`。依赖现在已经拆组，默认安装保持轻量，本地模型与 PDF 解析这类重依赖按需安装。
激活环境后，`xcientist`、`xcientist-survey`、`xcientist-idea` 这类 CLI 命令会直接出现在当前 shell 中。

## 🔐 环境变量

不同 Agent 读取的变量名并不完全一致，实际使用中最建议先配置这些：

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

说明：

- 如果你使用自定义 OpenAI-compatible 接口，最好同时设置 `OPENAI_API_BASE` 和 `OPENAI_BASE_URL`。
- CLI 会优先读取仓库根目录 `.env`，同时兼容旧的 `src/config/.env`。
- 当前主配置文件是 `src/config/default.yaml`。
- Survey、Idea、Experiment、Paper 在统一配置之外，仍有各自的运行约定。

## 📦 可选本地资源

部分检索路径依赖不在仓库中的本地资源：

- `data/processed/graph.db`
- `data/processed/core_component_summary_vector_store/`
- `models/bge-m3/`
- `models/all-MiniLM-L6-v2/`

如果要启用 graph-backed retrieval，可在仓库根目录启动图服务：

```bash
uvicorn graph.server:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## 🚀 快速开始

第一次建议先这样做：

```bash
uv sync --group memory --group ml
source .venv/bin/activate
cp .env.example .env
xcientist doctor
```

当 doctor 通过、graph/db 和模型到位后，再按下面的命令运行。

### 1. 运行 Survey Agent

推荐入口：

```bash
xcientist survey
```

直接覆盖 topic：

```bash
xcientist survey --topic "LLM Agent Memory System"
```

典型输出：

- `src/agents/survey_agent/outputs/.../survey.md`
- `src/agents/survey_agent/outputs/.../survey.json`
- `src/agents/survey_agent/outputs/.../evaluation.txt`

### 2. 运行 Idea Agent

推荐入口：

```bash
xcientist idea
```

直接覆盖 topic：

```bash
xcientist idea --topic "LLM Agent Memory System"
```

默认会使用 `src/config/default.yaml`，并在 `src/agents/idea_agent/runs/` 下创建运行目录，写出 `idea_result.json` 和日志。

### 3. 运行 Experiment Agent

推荐入口：

```bash
xcientist experiment --experiment my_exp --idea-json /abs/path/to/idea_result.json
```

仅准备工作空间：

```bash
xcientist experiment --experiment my_exp --idea-json /abs/path/to/idea_result.json --prepare-only
```

直接入口：

```bash
python -m src.agents.experiment_agent.main --experiment my_exp --resume --verbose
```

默认工作空间在 `workspace/<experiment_id>/`，常见产物包括：

- `idea.json`
- `project/`
- `dataset_candidate/`
- `results/`
- `agent_reports/`
- `ablation_results.json`

### 4. 运行 Paper Agent

`Paper Agent` 目前没有纳入默认的 `uv` 主路径，仍按手工入口使用。

推荐入口：

```bash
python -m src.agents.paper_agent.main \
  --experiment my_exp \
  --template-dir src/agents/paper_agent/latex/ICML2025_Template
```

恢复已有 paper workspace：

```bash
python -m src.agents.paper_agent.main \
  --experiment my_exp \
  --template-dir src/agents/paper_agent/latex/ICML2025_Template \
  --resume
```

`run_paper.sh` 目前存在，但它只是一个带硬编码参数的本地示例脚本，不适合作为通用入口。

### 5. 运行原型 Pipeline

```bash
xcientist pipeline
```

它会以 `src/config/default.yaml` 启动 `src.pipeline.run_loop`。如果你想一条命令跑通集成链路，这个入口可用；如果你要排查问题，还是建议逐个 Agent 运行。

## 🧭 配置说明

当前配置布局是混合式的：

| 区域 | 主要来源 |
|------|----------|
| 全局项目配置 | `src/config/default.yaml` |
| Survey Agent | `src/config/default.yaml` 的 `survey:` 段，以及 `src/agents/survey_agent/config/*.yaml` |
| Idea Agent | `src/config/default.yaml` 的 `idea:` 段 |
| Experiment Agent | `src/config/default.yaml` 的 `experiment:` 段和环境变量 |
| Paper Agent | `src/config/default.yaml` 的 `paper:` 段和 CLI 参数 |
| Pipeline | `src/config/default.yaml` 的 `pipeline:` 段 |

如果你是第一次配置这个项目，优先阅读和修改 `src/config/default.yaml`。

## 🤖 各 Agent 简述

### Survey Agent

- 主入口：`src/agents/survey_agent/scripts/run_deep_survey.py`
- 作用：检索论文、聚类、撰写 survey、执行评估
- 输出：survey markdown、survey JSON、评估结果

### Idea Agent（LigAgent）

- 主入口：`src/agents/idea_agent/run.py`
- 作用：从主题或成熟 idea 生成 proposal
- 核心特点：survey 驱动检索、graph-backed references、带领域锁定的 Memory-Guided MCTS
- 输出：`idea_result.json`、日志、workflow 产物

### Experiment Agent（SuperAgent）

- 主入口：`python -m src.agents.experiment_agent.main`
- 作用：准备工作空间、生成实现、执行 code/science 迭代、汇总 ablation 结果
- 输出：实验工作空间与最终 `ablation_results.json`

### Paper Agent

- 主入口：`python -m src.agents.paper_agent.main`
- 作用：把实验工作空间转换成论文写作工作空间
- 输出：LaTeX 项目、产物目录、运行状态、可选 PDF

## 📌 当前仓库状态

目前可以放心依赖的部分：

- 四个核心 Agent 都已经落在仓库中，并且存在连接路径。
- `src/config/__init__.py` 提供的统一配置加载器当前可用。
- 根目录的运行脚本和当前代码结构是一致的。

使用时需要额外注意：

- 部分配置示例仍然带有历史机器上的绝对路径，迁移到新机器时要先覆盖。
- 某些高级工作流依赖本地数据或模型，这些内容没有提交到仓库。
- 当前仓库快照中没有顶层 `LICENSE` 文件。
- 当前仓库快照中也没有明显的根级 `tests/` 自动化测试目录。
