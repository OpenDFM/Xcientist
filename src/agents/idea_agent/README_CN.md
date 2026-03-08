# LigAgent — Idea Agent 系统说明

LigAgent 是 ResearchAgent 中负责想法生成的子系统。它把一个研究主题，或者用户提供的成熟想法，转成结构化研究 proposal，核心手段是检索、文献分析、Memory-Guided MCTS 搜索和最终 idea materialization。

## 概览

当前实现围绕五个原则组织：

- **条件分支 workflow**，而不是自由轮询的动作循环
- **Contract 模式**，用于 `run.mature_idea`
- **根领域锁定**，避免 theory transfer 悄悄把 idea 带到别的领域
- **Preset 驱动的搜索姿态**，由 `idea_taste_mode` 控制
- **双重记忆引导**：向量记忆负责文本提示，符号记忆负责算子先验和评估校准

高层流程如下：

```mermaid
graph TD
    T[Topic 或 mature idea] --> KA[knowledge_aquisition]
    KA --> AA[advanced_analysis]
    AA --> IG[idea_generation]

    RH[已有 rag_hits] --> AA2[advanced_analysis]
    AA2 --> RR[re_analysis_replan]
    RR --> IG

    IG --> MCTS[Memory-Guided MCTS]
    MCTS --> PFI[persist_final_idea]
    PFI --> OUT[idea_result.json]
```

主流程由 `utils/workflow/ligagent_flow.py` 决定：

- 若 `artifact["rag_hits"]` 为空：`knowledge_aquisition -> advanced_analysis -> idea_generation`
- 若 `artifact["rag_hits"]` 已存在：`advanced_analysis -> re_analysis_replan -> idea_generation`

现在的主流程里已经没有旧版那种“五动作控制器”了，文档应以这条条件 workflow 为准。

## 运行时工作流

### `knowledge_aquisition`

冷启动检索阶段。

- 用 `artifact["retrieval_keywords"]` 从 Semantic Scholar 做种子检索
- 基于论文 keynote 或 `mature_idea` 构造 OutcomeRAG 查询
- 扩展引用并把 citation title 映射回论文
- 做全文补齐、筛选和压缩，得到下游使用的 curated paper batch

写入：

- `artifact["references"]`
- `artifact["rag_query"]`
- `artifact["rag_hits"]`
- `artifact["rag_contents"]`
- `artifact["paper_contents"]`

### `advanced_analysis`

把筛选后的论文批次转成结构化分析。

- 提炼关键机制、痛点、开放问题和后续搜索种子
- 把可复用背景知识追加到 `artifact["background_knowledge"]`

写入：

- `artifact["analysis"]`
- `artifact["background_knowledge"]`

### `re_analysis_replan`

只在已有 RAG 上下文时进入。

- 重写当前 topic framing
- 可能更新 `artifact["mature_idea"]`
- 更新检索关键词，为下一轮 retrieval 做准备

### `idea_generation`

它会从以下上下文构造 MCTS 输入：

- `artifact["analysis"]`
- `artifact["idea_pool"]`
- `artifact["background_knowledge"]`
- curated paper context
- 可选的 `artifact["mature_idea"]`

随后执行：

1. 注入 symbolic priors
2. 调用 `MemoryGuidedMCTS.search(...)`
3. 把最佳 idea 写入 `artifact["idea_pool"]`
4. 通过 `persist_final_idea(...)` 生成 `idea_result.json`

## Artifact 与输出

`artifact` 是整次 LigAgent 运行的唯一可变状态容器。`agent/artifacts.py` 里最关键的字段包括：

| 字段 | 含义 |
|------|------|
| `topic` | 当前 topic 历史 |
| `run_topic` | launcher 传入的原始 topic |
| `mature_idea` | contract root 或 replanning 后的成熟想法 |
| `background_knowledge` | 分析阶段生成的背景知识 |
| `analysis` | 结构化分析结果 |
| `references` | 筛选后的论文批次 |
| `rag_query`、`rag_hits`、`rag_contents` | OutcomeRAG 上下文 |
| `paper_contents` | 论文解析元数据 |
| `idea_pool` | `idea_generation` 输出的 canonical idea payload |
| `evaluations` | winning candidate 的评估结果 |
| `retrieval_keywords` | 当前检索关键词 |
| `workflow_trace`、`workflow_state`、`operation_trace` | 执行元数据 |

如果 `idea_taste_mode` 成功解析，`LigAgent.__init__` 还会补充：

- `artifact["idea_taste_mode"]`
- `artifact["idea_taste_label"]`

每个 `idea_pool` 的 winning entry 会保留比最终 JSON 更多的信息：

- canonical idea 字段：`title`、`abstract`、`method`、`components` 等
- `evaluation`
- `search_score`
- `search_path`
- `pareto_candidates`
- `search_trace`

最终持久化出来的 `idea_result.json` 则更偏论文写作接口，主要包含：

- `title`
- `abstract`
- `introduction`
- `components`
- `algorithm`
- `reference_papers`
- `mcts_evolution`
- 可选的 `idea_contract`

## Memory-Guided MCTS

### 根状态与根领域锁定

`build_root_state(...)` 会从三个来源之一初始化根节点：

1. 如果存在 `mature_idea`，直接用它作为 contract root
2. 否则，如果已有 `idea_pool`，用最新一条 idea 作为起点
3. 否则，用分析结果和背景知识合成一个 baseline seed

真正搜索开始前，`MemoryGuidedMCTS.search(...)` 会先给根节点分类出 1 到 2 个固定领域，并把它们写入每个 `IdeaState` 的 `root_domains`。

这个设计会约束后续几件事：

- 每个 child idea 都继承同一组 `root_domains`
- skill instantiation prompt 明确禁止领域漂移
- `theory-transfer-injection` 可以参考别的领域，但不能把 idea 的 home domain 改掉

### 核心搜索对象

主要运行时结构有：

- `IdeaState`：当前 idea 快照，含 components、defects、budget、`root_domains`、`edit_plan`、`skill_metrics`
- `IdeaNode`：MCTS 节点，持有 parent / children / visits / value / evaluation
- `IdeaEvaluation`：多指标评估结果
- `EditPlan`：skill 编译后的原子 component 编辑计划和验证协议

### 缺陷标签与技能库

搜索空间建立在 canonical defect registry 和一组 edit-operator skills 上。

当前内置 skill 包括：

| Skill | 主要用途 |
|------|----------|
| `mechanism-commit-innovation` | 锁定一个明确的机制级创新 |
| `alternative-path-contrast` | 引入 fallback / rare-regime 路径 |
| `surgical-modularity` | 做局部可消融的模块化改动 |
| `multi-scale-coordinator` | 协调多尺度 / 多层级决策 |
| `hierarchical-decomposition` | 把平面流程改成显式层次结构 |
| `feedback-closed-loop` | 从 open loop 变成可监控的反馈闭环 |
| `theory-transfer-injection` | 从别的领域注入可迁移原则 |
| `speculative-execution-with-repair` | 乐观路径加 repair / rollback |
| `resource-aware-adaptive-path` | 让执行路径适应 budget / load |

Evaluator 只允许返回 `utils/mcts/defect_registry.py` 里的 canonical defect tags。根节点会在第一次 expand 前先跑一次 evaluator，用真实 `detected_defects` 替换掉占位符 `unexplored_gap`。

### Idea Taste Presets

现在的 `idea_taste_mode` 已经不只是“调评估权重”。

它同时影响三层：

1. **evaluation weights**：通过 `apply_idea_taste_preset(...)`
2. **skill selection bias**：通过 `SkillCatalog.select_skills(...)`
3. **component generation guidance**：通过 skill-instantiation prompt

当前可用 preset：

| Preset | 搜索姿态 |
|------|----------|
| `moonshot_inventor` | 追求单个大胆机制和超额上限 |
| `bridge_builder` | 偏向跨领域迁移和适配 |
| `steady_engineer` | 偏向小而稳、容易落地的改动 |
| `ambitious_realist` | 高上限但保持基本可实现性 |
| `evidence_first` | 优先最容易被严格验证的机制 |

每个 preset 现在都定义：

- `weights`
- `skill_bias`
- `instantiation_guidance`

### Expand 阶段

`expand_node_with_skills(...)` 是当前最关键的搜索入口。

对每个待扩展节点，会按顺序执行：

1. **检索向量记忆**
   - `VectorMemoryAccessor.retrieve_bundle(...)` 返回 field knowledge、anti-patterns 和 fix recipes

2. **计算 symbolic action priors**
   - `symbolic_memory.compute_action_priors(...)` 从符号记忆里取出 operator-level 先验

3. **选择候选 skill**
   - `SkillCatalog.select_skills(...)` 对每个 skill 计算：
     - `defect_score`
     - `skill_prior`
     - `preset_bias`
     - `gate_score`
   - 当前权重公式是：

   ```text
   selection_total =
       0.55 * defect_score +
       0.20 * skill_prior +
       0.20 * preset_bias +
       0.05 * gate_score
   ```

4. **融合 preset 排序与 symbolic rerank**
   - 先从 symbolic-memory priors 中为每个已选 skill 得到一个 `symbolic_score`
   - 再在候选集内部归一化为 `symbolic_norm`
   - 最终排序分数是：

   ```text
   final_order_score = 0.80 * selection_total + 0.20 * symbolic_norm
   ```

   这样 symbolic memory 仍然重要，但不会完全压掉 preset-aware 的初始选择。

5. **编译 plan**
   - `compile_plan(...)` 把 skill blueprint 编译成 `ComponentEdit` 和验证协议
   - `SkillUsagePrior` 中学到的 `rule_constraints` 会追加到 plan guardrails

6. **实例化 plan**
   - prompt 现在会显式注入：
     - `idea_taste_mode`
     - `idea_taste_label`
     - `taste_guidance`
     - 固定的 `root_domains`
   - taste guidance 只是软约束，不能覆盖 plan、target defects 或 guardrails

7. **`theory-transfer-injection` 的特殊处理**
   - 先根据当前 idea 和 edit plan 构造 transfer query
   - 再从 root domains 之外检索 paper-graph nodes
   - 若没有满足阈值的跨领域 reference，则直接跳过该 child
   - 若成功检索，则把 transfer query 和 cross-domain references 一起注入 instantiation prompt

8. **物化 child state**
   - 用 LLM 返回的 `component_mapping` 和 `edit_reasons` 把 generic plan 改成 concrete idea
   - child 的 `skill_metrics` 现在会额外记录：
     - `idea_taste_mode`
     - `skill_selection_breakdown`
     - `symbolic_rerank_breakdown`

### Simulate / Evaluate 阶段

`simulate_node_value(...)` 会构造 evaluator prompt，输入包括：

- topic
- mature idea
- 最新 analysis
- 当前 run 的 idea pool snapshot
- paper context
- compiled edit plan
- candidate idea
- rewrite path
- canonical defect registry
- retrospective symbolic-memory hints

评估阶段的几个关键点：

- evaluator 输出会被解析成 `IdeaEvaluation`
- novelty 可能会被 `ComponentNoveltyScorer` 覆盖重算
- 如果 protocol score 缺失，会根据 edit plan 兜底估算
- 评估结果按 `(IdeaState.signature, path_key)` 缓存

### 回传与经验学习

在 simulation 之后：

- rollout 分数会沿路径回传
- `update_skill_prior_from_evaluation(...)` 会更新 `SkillUsagePrior`
- trace 会记录 operator、defects、rationale、score、path、evaluation、signature、edit plan 和 `skill_metrics`

### Experience 写回

当 `evaluation.confidence > min_confidence_for_memory` 时，当前节点会被写成一条经验。

每条经验包含：

- defect 摘要
- 选中的 skill / operator
- lift estimate
- title / context
- feedback
- edit plan

这些经验最终通过 `SlotProcess` 写回到长期记忆系统。

## 配置

主要配置文件有两份：

- `config/run/default.yaml`
- `config/mcts/default.yaml`

### `run/default.yaml`

最重要的键包括：

- `topics`
- `parallelism`
- `output_root`
- `console_logs`
- `rag_config`
- `mature_idea`
- API 凭据和检索端点

### `mcts/default.yaml`

最重要的搜索相关键包括：

- `max_iterations`
- `max_depth`
- `branching_factor`
- `exploration_constant`
- `idea_taste_mode`
- `generation_model`
- `evaluation_model`
- `generation_temperature`
- `evaluation_temperature`
- `component_novelty_*`
- `theory_transfer_retrieval_top_k`
- `theory_transfer_similarity_threshold`
- `symbolic_memory_path`
- `skill_prior_success_threshold`

截至当前代码版本，仓库中的默认 preset 是 `evidence_first`。

## Quickstart

```bash
./run_idea.sh
# 或
PYTHONPATH=. python src/agents/idea_agent/run.py
```

输出目录：

```text
src/agents/idea_agent/runs/<topic-slug>-<timestamp>-<uuid>/
```

最关键的两个产物是：

- `idea_result.json`
- `logs/ligagent.log`

项目级使用方式请参考仓库根目录的 `README_CN.md`。
