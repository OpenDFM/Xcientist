# X-Scientist

[English](README.md) | [中文](README_CN.md)

## 概述

X-Scientist 是当前这个仓库里的科研 Agent 系统，当前代码实际上围绕四个核心 Agent 展开：

1. **Survey Agent**：检索论文、构建文献聚类、生成 survey 草稿，并产出可供下游检索使用的 `survey.md` / `survey.json`。
2. **Idea Agent（LigAgent）**：把研究主题或成熟想法转成结构化研究 proposal，核心路径是检索、分析和 Memory-Guided MCTS。
3. **Experiment Agent（SuperAgent）**：把 idea 落成可运行代码，执行实验，并围绕失败与反馈继续迭代。
4. **Paper Agent**：读取实验工作空间，生成独立的论文工作空间、LaTeX 草稿和编译产物。

当前仓库里几个重要的共享子系统：

- `src/config/default.yaml`：当前统一配置文件，Idea Agent 直接使用，pipeline 原型代码也引用它。
- `src/memory/`：共享 memory 子系统，Idea Agent / Experiment Agent 的部分能力会用到。
- `src/pipeline/`：Survey -> Idea -> Experiment 的循环原型代码，但目前不是主推荐入口。

## 环境重建

当前仍然建议优先使用 **conda + environment.yml**，这样最接近仓库现在的依赖状态。

### 方法一：conda + environment.yml（推荐）

```bash
conda env create -f environment.yml
conda activate research-agent
```

### 方法二：pip + requirements.txt

```bash
conda create -n research-agent python=3.10 -y
conda activate research-agent
pip install -r requirements.txt
```

说明：

- `requirements.txt` 里包含了 `src/memory` 和 `src/agents` 的 editable install。
- Paper Agent 当前的 PDF 编译依赖本地 TeX 工具链，例如 `tectonic`、`latexmk` 或 `pdflatex`。
- 如果你使用自定义的 OpenAI-compatible base URL，最好同时设置 `OPENAI_BASE_URL` 和 `OPENAI_API_BASE`，因为不同 Agent 目前读取的环境变量名并不完全一致。

## 配置布局

当前仓库采用混合配置布局。

| 模块 | 当前配置来源 | 说明 |
|------|--------------|------|
| Idea Agent | `src/config/default.yaml` 的 `idea:` 段 | `src/agents/idea_agent/utils/core/config_loader.py` 默认从这里加载；也可以用 `IDEA_AGENT_CONFIG=/path/to/config.yaml` 覆盖。 |
| Survey Agent | `src/agents/survey_agent/config/*.yaml` | Standalone Survey 运行仍使用 Hydra 配置，键名是 `BasicInfo`、`APIInfo`、`ModuleInfo`。 |
| Experiment Agent | 环境变量 + `src/agents/experiment_agent/shared/utils/config.py` | 运行期行为主要由 env 和代码常量控制。 |
| Paper Agent | CLI 参数 + 环境变量 + `src/agents/paper_agent/utils/config.py` | 主要控制项是 `--template-dir` 和 workspace 相关环境变量。 |
| Pipeline 原型 | `src/config/default.yaml` 的 `pipeline:` 段 | 代码还在，但仓库根目录里没有 `run_pipeline.sh`。 |

一个需要特别注意的事实：

- 仓库中部分 Survey / Paper / 统一配置文件仍带有历史机器上的绝对路径示例，迁移到新机器时要先覆盖这些路径。

## 四个 Agent

### 1. Survey Agent

**作用**：围绕一个主题生成文献综述，并把结构化的 survey 结果保存下来，供 OutcomeRAG 和 Idea Agent 继续使用。

**当前实际工作流**：

```
种子论文检索
-> 引用 / 被引扩展
-> 论文阅读与 keynotes
-> 聚类
-> 簇内分析
-> 簇间分析
-> 大纲生成
-> survey 草稿生成
-> review / revise
-> 导出 survey 与参考文献
-> 评价
```

**Standalone 配置位置**：

- `src/agents/survey_agent/config/deep_survey.yaml`
- `src/agents/survey_agent/config/deep_survey_xiaomi.yaml`
- `src/agents/survey_agent/config/outcomeRAG.yaml`

**关键输入**：

- `BasicInfo.topic`
- `BasicInfo.save_path`
- `BasicInfo.save_json_path`
- `BasicInfo.evaluation_save_path`
- `APIInfo.llm_api_key`
- `APIInfo.llm_api_base_url`
- `ModuleInfo.WorkCollector.*`
- `ModuleInfo.WorkAnalyzer.*`
- `ModuleInfo.SurveyGenerator.*`

**输出**：

- `BasicInfo.save_path`：生成的 survey markdown
- `BasicInfo.save_json_path`：结构化 survey JSON
- `BasicInfo.evaluation_save_path`：评价结果文本
- `BasicInfo.cache_path` 下的缓存 / database 产物

**推荐用法**：

由于仓库自带的 YAML 里包含示例绝对输出路径，最稳妥的方式是运行时直接覆盖输出路径：

```bash
python -m src.agents.survey_agent.scripts.run_deep_survey \
  --config-path src/agents/survey_agent/config \
  --config-name deep_survey \
  BasicInfo.topic="LLM Agent Memory System" \
  BasicInfo.save_path="src/agents/survey_agent/outputs/llm_agent_memory_system.md" \
  BasicInfo.save_json_path="src/agents/survey_agent/outputs/llm_agent_memory_system.json" \
  BasicInfo.evaluation_save_path="src/agents/survey_agent/outputs/llm_agent_memory_system_eval.txt"
```

**它和 Idea Agent 的连接方式**：

- Idea Agent 的 `idea.run.rag_config` 通常会指向 `src/agents/survey_agent/config/outcomeRAG.yaml`。
- 而这个 OutcomeRAG 配置里引用的 survey 文件，必须是真实存在于你本机上的输出文件。

---

### 2. Idea Agent（LigAgent）

**作用**：把研究主题或已有成熟想法转成结构化研究 proposal，核心路径是“检索 -> 分析 -> Memory-Guided MCTS 搜索 -> 持久化”。

**当前实际工作流**：

主流程是一个条件分支 workflow：

| 阶段 | 作用 |
|------|------|
| `knowledge_aquisition` | 冷启动检索：Semantic Scholar 种子检索 -> OutcomeRAG 查询 -> 引用扩展 -> 论文丰富与筛选 |
| `advanced_analysis` | 从精选论文中提炼机制、痛点、开放问题与搜索种子 |
| `re_analysis_replan` | 当已有 RAG 上下文时，重写 topic 焦点、成熟想法和检索方向 |
| `idea_generation` | 运行 Memory-Guided MCTS；若开启 `LigAgent-Pro`，则从同一 root 并行搜索所有 preset，并在写出 `idea_result.json` 之前执行 fusion |

当前控制流：

- 冷启动：`knowledge_aquisition -> advanced_analysis -> idea_generation`
- 若 `artifact["rag_hits"]` 已存在：`advanced_analysis -> re_analysis_replan -> idea_generation`

**当前版本 Idea Agent 的关键特点**：

- **Contract 模式**：`idea.run.mature_idea` 会把给定想法直接作为 MCTS 根节点。
- **根领域锁定**：MCTS root 会先被分类到 1 到 2 个固定领域，后续子节点不能偏离。
- **Preset 驱动搜索**：`idea.mcts.idea_taste_mode` 同时影响评估权重、skill 选择偏置和 component 生成引导。
- **LigAgent-Pro**：当 `idea.run.LigAgent-Pro` 打开时，会从同一 prepared root 并行跑五种 preset，再做融合。
- **跨领域理论迁移但不换领域**：可以借机制，但最终 idea 必须留在原本领域。
- **双重记忆系统**：向量记忆提供文本提示，符号记忆提供 operator prior 和 evaluator hints。

**当前配置来源**：

- `src/config/default.yaml` 中的 `idea:` 段
- 可选覆盖方式：`IDEA_AGENT_CONFIG=/path/to/another_config.yaml`

Idea Agent 的默认设置来自统一配置文件，而不是单独的 `src/agents/idea_agent/config/...` 目录。

**关键输入**：

- `idea.run.topics`
- `idea.run.LigAgent-Pro`
- `idea.run.mature_idea`（可选）
- `idea.run.rag_config`
- `idea.agent.model`
- `idea.mcts.*`
- `idea.fusion.*`

**输出**：

- 默认输出到 `src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/idea_result.json`
- 默认日志在 `src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/logs/ligagent.log`

**最小配置示例**：

```yaml
idea:
  run:
    topics:
      - "LLM Agent Memory System"
    LigAgent-Pro: true
    output_root: "runs"
    rag_config: "src/agents/survey_agent/config/outcomeRAG.yaml"
    # mature_idea: "可选的 contract root"

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

**用法**：

```bash
./run_idea.sh
# 或
python src/agents/idea_agent/run.py
```

当前 standalone 入口是**配置驱动**的，不直接通过 `--topic` 传参。

> LigAgent 的当前内部实现，包括 artifact 字段、MCTS 扩展逻辑、preset 机制、root-domain 控制、theory-transfer 检索和持久化细节，详见 `src/agents/idea_agent/README_CN.md`。

---

### 3. Experiment Agent（SuperAgent）

**作用**：把 idea 落成可运行项目，执行实验，并在代码层和科学层之间迭代。

**当前实际阶段**：

1. **Prepare 阶段**：创建 workspace、写入 `idea.md`、克隆参考仓库、下载候选数据集。
2. **Engineering layer**：生成和整合项目代码。
3. **Science layer**：执行实验，记录结果与反馈。

**主要入口**：

- `python -m src.agents.experiment_agent.prepare`
- `python -m src.agents.experiment_agent.main`
- 便捷脚本：`./run_experiment.sh`

**输入**：

- `--experiment`：实验 ID / workspace 名称
- `--idea-json`：`run_experiment.sh` 需要
- `--prepare`：wrapper 中可选，先执行 prepare
- `--resume`：主模块支持断点续传
- `--fresh`：强制 fresh run

**工作空间输出**：

在 `src/agents/experiment_agent/workspaces/<experiment_id>/` 下，运行期可能生成：

- `idea.json`：复制进来的原始 idea
- `idea.md`：PrepareAgent 物化后的 markdown proposal
- `project/`：生成的项目代码
- `repos/`：克隆下来的参考仓库
- `dataset_candidate/`：候选数据集
- `specs/`：为兼容性同步出来的 specs / plans / reports
- `cached/`：checkpoint 与状态缓存
- `logs/`：workspace 级日志

关键实验结果一般落在：

- `project/result/code/iter_v*/`
- `project/result/science/iter_v*/`

**常用环境变量**：

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

**当前模型 / 运行常量位置**：

- 定义在 `src/agents/experiment_agent/shared/utils/config.py`
- 需要注意 shell wrapper 和 Python 模块本体的默认 env 行为并不完全相同

**推荐用法**：

使用 wrapper：

```bash
./run_experiment.sh --experiment my_exp --idea-json /path/to/idea_result.json --prepare
```

直接使用底层模块：

```bash
mkdir -p src/agents/experiment_agent/workspaces/my_exp
cp /path/to/idea_result.json src/agents/experiment_agent/workspaces/my_exp/idea.json

python -m src.agents.experiment_agent.prepare --experiment my_exp --force --clone-depth 1 --verbose
python -m src.agents.experiment_agent.main --experiment my_exp --resume --verbose
```

当前 wrapper 的两个现实行为需要注意：

- `run_experiment.sh` 默认通过环境变量把 Experiment Agent memory 关闭了。
- wrapper 只有在带 `--prepare` 时才会复制 `idea.json`。
- wrapper 在主流程阶段固定使用 `--resume`。

---

### 4. Paper Agent（论文撰写）

**作用**：读取 Experiment Agent 的 workspace，并生成独立的论文工作空间，里面包含 specs、LaTeX 文件、编译日志和论文写作产物。

**当前输入**：

- `--experiment`：Experiment Agent 的 workspace 名称
- `--template-dir`：LaTeX 模板目录
- `--resume`：恢复已有 paper workspace

**当前输出**：

在 `src/agents/paper_agent/workspaces/<experiment_id>/` 下，运行时会生成：

- `paper/`：复制或初始化后的 LaTeX 项目
- `artifacts/`：编译日志、PDF 页面、提取资产、子代理中间产物
- `specs/`：paper 侧的 spec / plan / constitution 文件
- `state/paper_state.json`：可续跑状态文件

**Paper Agent 如何读取实验结果**：

它会从 `src/agents/experiment_agent/workspaces/<experiment_id>/` 中解析输入，重点查找：

- `idea.md`
- `specs/`
- `project/`
- 以及诸如 `project/result/science/iter_v*/result_summary.json` 这样的结果文件

**常用环境变量**：

- `OPENAI_API_KEY`
- `OPENAI_API_BASE`
- `MINIMAX_API_KEY`
- `PAPER_AGENT_WORKSPACES_DIR`
- `EXPERIMENT_AGENT_WORKSPACES_DIR`
- `PAPER_AGENT_BASH_TIMEOUT_SECONDS`
- `PAPER_AGENT_ENABLE_TRACING`

**编译细节**：

- 当前 PDF 编译会优先寻找本机上的 `tectonic`、`latexmk` 或 `pdflatex`。
- 如果这些工具都不存在，paper workspace 仍会创建，但 PDF 编译会失败。

**推荐用法**：

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

重要说明：

- `run_paper.sh` 当前是一个带硬编码实验名和模板路径的本地示例脚本，不适合作为通用入口来写进正式文档。

## Pipeline 当前状态

仓库里仍然保留了 `src/pipeline/run_loop.py` 里的 Survey -> Idea -> Experiment 循环，也保留了 `src/config/default.yaml` 里的 `pipeline:` 配置段。

仓库根目录里没有：

```bash
./run_pipeline.sh
```

这个命令在当前仓库里不可用，原因是：

- 当前仓库根目录里并没有 `run_pipeline.sh`
- `src/pipeline/run_loop.py` 会从 `src.config` 导入 `load_config`，但 `src/config/__init__.py` 目前并没有导出这个函数

所以当前最稳妥的工作流是分别运行四个 Agent：

```
Survey -> Idea -> Experiment -> Paper
```

## 目录结构

```text
ResearchAgent/
├── README.md
├── README_CN.md
├── environment.yml
├── requirements.txt
├── run_idea.sh
├── run_experiment.sh
├── run_paper.sh                  # 本地示例 wrapper，不是推荐的通用入口
├── src/
│   ├── config/
│   │   └── default.yaml          # Idea Agent / pipeline 原型当前使用的统一配置
│   ├── agents/
│   │   ├── survey_agent/
│   │   │   ├── config/           # Standalone Survey 的 Hydra 配置
│   │   │   ├── scripts/
│   │   │   ├── modules/
│   │   │   └── outputs/          # 示例 survey 输出
│   │   ├── idea_agent/
│   │   │   ├── run.py
│   │   │   ├── agent/
│   │   │   ├── utils/
│   │   │   └── runs/             # 运行时输出根目录
│   │   ├── experiment_agent/
│   │   │   ├── prepare.py
│   │   │   ├── main.py
│   │   │   ├── layers/
│   │   │   ├── shared/
│   │   │   └── workspaces/       # 运行时动态创建，初始可能不存在
│   │   └── paper_agent/
│   │       ├── main.py
│   │       ├── entry.py
│   │       ├── latex/ICML2025_Template/
│   │       └── workspaces/       # 运行时动态创建，初始可能不存在
│   ├── memory/
│   └── pipeline/
└── tests/
```

## 端到端流程

```text
1. Survey Agent
   输入：研究主题
   输出：survey.md + survey.json

   ↓（Survey 输出被 OutcomeRAG 配置引用）

2. Idea Agent
   输入：idea.run.topics / idea.run.mature_idea
   输出：src/agents/idea_agent/runs/<slug-timestamp-uuid>/idea_result.json

   ↓（idea_result.json -> experiment workspace 的 idea.json）

3. Experiment Agent
   输入：experiment workspace + idea.json
   输出：生成项目 + 实验结果 + specs / reports

   ↺ 实验反馈 / 结果摘要可以回流到 Idea Agent
     用于下一轮 proposal refinement

   Loop：Idea Agent <-> Experiment Agent
   持续迭代，直到 idea 和实现达到可以写论文的状态

   ↓（experiment workspace 成为 Paper Agent 输入）

4. Paper Agent
   输入：experiment workspace + LaTeX template
   输出：包含 LaTeX、artifacts 和可选 PDF 的 paper workspace
```

## 完整示例

```bash
# Step 1: 生成 survey 产物
python -m src.agents.survey_agent.scripts.run_deep_survey \
  --config-path src/agents/survey_agent/config \
  --config-name deep_survey \
  BasicInfo.topic="LLM Agent Memory System" \
  BasicInfo.save_path="src/agents/survey_agent/outputs/llm_agent_memory_system.md" \
  BasicInfo.save_json_path="src/agents/survey_agent/outputs/llm_agent_memory_system.json" \
  BasicInfo.evaluation_save_path="src/agents/survey_agent/outputs/llm_agent_memory_system_eval.txt"

# Step 2: 基于统一配置生成 idea
./run_idea.sh

# Step 3: 跑实验
./run_experiment.sh --experiment my_exp --idea-json src/agents/idea_agent/runs/<topic-run>/idea_result.json --prepare

# Step 4: 写论文
python -m src.agents.paper_agent.main \
  --experiment my_exp \
  --template-dir src/agents/paper_agent/latex/ICML2025_Template
```

## 注意事项

1. **先改机器相关路径**：仓库里仍有不少 YAML / Python 配置保留了历史机器的绝对路径示例。
2. **密钥优先放环境变量**：API key 和 base URL 更适合放在 env 里，而不是直接提交到配置文件中。
3. **Survey 输出路径必须对得上**：OutcomeRAG 和 Idea Agent 只有在 RAG 配置里引用的 survey 文件真实存在时才能正常工作。
4. **`workspaces/` 目录按需生成**：`experiment_agent` 和 `paper_agent` 下在首次运行前没有 `workspaces/` 是正常现象。
5. **不确定时优先用 Python 模块入口**：它们比旧辅助脚本更接近当前代码的真实行为。
