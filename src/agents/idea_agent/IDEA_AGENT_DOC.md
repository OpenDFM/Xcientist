# Idea Agent System Design: Memory-Guided Idea Evolution
> Version: 1.0.0
> Date: 2024-12-24
> Context: Automated Scientific Idea Discovery (LigAgent)

## 目录 (Table of Contents)

1. 架构总览 (High-Level Architecture)
2. 记忆与知识层 (Memory & Knowledge Layer)
3. 树搜索引擎 (Memory-Guided MCTS)
4. 行为协议 (Action Protocols)
5. 工具定义 (Tools)
6. 输出工件 (Artifacts)
7. 关键设计原则 (Design Principles)
8. 快速上手指南 (Quickstart Guide)

## 1. 架构总览 (High-Level Architecture)

Idea Agent 将研究灵感生成建模为**“知识获取 → 分析归纳 → MCTS 搜索 → 评估落地”**的循环流程。
核心驱动是一个**Memory-Guided MCTS**，并将外部文献检索与长期记忆检索统一为“上下文燃料”。

```mermaid
graph TD
    subgraph "LigAgent Core"
        A[Topic Bootstrap] --> B{LLM Action Selector}
        B -->|knowledge_aquisition| C[Paper Search Tool]
        B -->|advanced_analysis| D[Analysis Synthesizer]
        B -->|idea_generation| E[Memory-Guided MCTS]
        B -->|idea_evaluation| F[Idea Evaluator]
        B -->|re_analysis_replan| G[Topic Replanner]
    end

    subgraph "Knowledge & Memory"
        C --> H[PaperRepository]
        H --> I[(Parsed Papers Cache)]
        E --> J[Long-Term Memory (FAISS)]
        D --> K[(Short-Term Memory)]
        E --> K
        F --> K
        G --> K
    end

    E --> L[idea_result.json]
    K --> B
```

---

## 2. 记忆与知识层 (Memory & Knowledge Layer)

### 2.1 短期记忆 (Working Memory)
**定位**：每次运行的“工作记忆”。用于追踪主题、检索关键词、分析、候选想法与步骤日志。

**Memory Layout (`src/agents/idea_agent/agent/memory.py`)**:
```python
memory = {
    "topic": [],
    "survey": "",
    "background_knowledge": [],
    "analysis": [],
    "references": [],
    "idea_pool": [],
    "evaluations": [],
    "retrieval_keywords": [],
    "paper_contents": {},
    "dialogue": {},
    "steps": [],
    "memory_structure": {}
}
```

### 2.2 文献仓库 (Paper Repository)
**定位**：轻量化文献解析层，复用 Survey Agent 的下载与解析能力，同时避免全量 survey 图的开销。

**Pipeline**:
1. `semantic_search` 从 Semantic Scholar 获取文献信息。
2. `PaperRepository.prepare_papers()` 下载并生成 Markdown 解析内容。
3. `IdeaPaperAnalyzer.ensure_keynotes()` 生成 keynotes 摘要。
4. 结果落地到 `memory["paper_contents"]` 和缓存目录。

### 2.3 长期记忆 (Long-Term Memory)
**定位**：跨 run 的“经验记忆”，用于在 MCTS 扩展与评估阶段提供缺陷修复、反例与成功范式。

**Memory Bundle**:
- Semantic Store: Field knowledge
- Episodic Store: Anti-patterns
- Procedural Store: Fix recipes

**Persist 条件**:
- 仅当评估 `confidence > min_confidence_for_memory` 时，才写入 LTM。

---

## 3. 树搜索引擎 (Memory-Guided MCTS)

MCTS 的根节点来自 `idea_pool` 或最新分析摘要，随后通过“编辑算子”迭代扩展，最终选出最优方案并回写经验。

### 3.1 节点结构 (Idea Node)
```python
class IdeaNode:
    state: IdeaState
    parent: Optional[IdeaNode]
    children: List[IdeaNode]
    visits: int
    value_sum: float
    evaluation: Optional[IdeaEvaluation]

    def uct_value(self, parent_visits, exploration_constant):
        return (value_sum / visits) + c * sqrt(log(parent_visits) / visits)
```

### 3.2 状态载荷 (IdeaState)
```python
class IdeaState:
    title: str
    abstract: str
    core_contribution: str
    method: str
    experiments: str
    risks: str
    tags: List[str]
    operator: str
    target_defects: List[str]
    rationale: str
```

### 3.3 评估信号 (IdeaEvaluation)
```python
class IdeaEvaluation:
    novelty: float
    feasibility: float
    clarity: float
    impact: float
    risk: float
    conciseness: float
    confidence: float

    @property
    def composite(self):
        return 0.30*novelty + 0.25*impact + 0.20*feasibility + 0.15*clarity + 0.10*conciseness - 0.2*risk
```

### 3.4 关键机制
- **扩展阶段**：基于 `EDIT_OPERATORS` 提议新的 idea 子节点。
- **模拟评估**：LLM 评分 + 结构化 JSON 输出。
- **回传更新**：平均值 + UCT 探索平衡。
- **经验沉淀**：符合置信度阈值时写入 Long-Term Memory。

---

## 4. 行为协议 (Action Protocols)

LigAgent 在每轮根据 `memory["steps"]` 决定下一步行动，形成可复现的多阶段 pipeline。

### 4.1 knowledge_aquisition
**场景**：检索论文、构建初始文献上下文。

**输入**：`retrieval_keywords[-1]`
**输出**：`memory["references"]`, `memory["paper_contents"]`

### 4.2 advanced_analysis
**场景**：对文献进行结构化分析，产出关键方法/痛点/未来方向。

**输入**：`memory["references"]`
**输出**：`memory["analysis"]`, `memory["background_knowledge"]`

### 4.3 idea_generation (MCTS)
**场景**：基于当前 topic + memory 上下文进行 MCTS 搜索，产出最优 idea。

**输入**：`analysis`, `idea_pool`, `paper_context`, `background_knowledge`
**输出**：`memory["idea_pool"]`, `memory["evaluations"]`, `idea_result.json`

### 4.4 idea_evaluation
**场景**：对已生成的 idea 进行进一步评估。

**输入**：`memory["idea_pool"][-1]`
**输出**：更新 `idea_pool[-1]["evaluation"]`

### 4.5 re_analysis_replan
**场景**：当当前路径不足时，重新设定 topic 与检索关键词。

**输入**：`memory["idea_pool"][-1]`, `retrieval_keywords`, `topic`
**输出**：新增 `topic`, `retrieval_keywords` 条目

---

## 5. 工具定义 (Tools)

Idea Agent 使用轻量 tool 接口与外部检索交互。

#### Tool 1: `semantic_search(query, limit=10)`
*   **场景**: 拉取论文元数据与摘要。
*   **来源**: Semantic Scholar API + 本地缓存。
*   **输出**: 论文列表 (title, abstract, authors, year, url)。

#### Tool 2: `semantic_recommend(positive_ids, negative_ids, ...)`
*   **场景**: 根据已知论文推荐更多文献。
*   **输出**: 论文列表 (排序可按引用/最新/相关性)。

---

## 6. 输出工件 (Artifacts)

### 6.1 idea_result.json
Idea Agent 的最终产物，包含完整 idea、方法设计、参考文献、数据集与 MCTS 轨迹。

```json
{
  "title": "...",
  "abstract": "...",
  "introduction": "...",
  "algorithm": ["..."],
  "reference_papers": ["..."],
  "datasets": ["..."],
  "mcts_evolution": {
    "best_path": "...",
    "iterations": ["..."]
  }
}
```

### 6.2 Logs & Runs
`scripts/run.py` 会为每个 topic 生成独立 run 目录，包含：
- `logs/ligagent.log`
- `idea_result.json`

---

## 7. 关键设计原则 (Design Principles)

1. **MCTS 驱动创新**：不是一次生成，而是系统化探索与演化。
2. **记忆即策略**：短期记忆提供上下文，长期记忆提供“范式纠偏”。
3. **文献先行**：所有想法必须通过文献 grounding 进行约束与补强。
4. **可追溯与可复现**：search_trace + idea_result 提供完整溯源。

---

## 8. 快速上手指南 (Quickstart Guide)

本节面向第一次使用 LigAgent 的读者，目标是在 5-10 分钟内跑通一次完整流程。

### 8.1 准备环境
- **API Key**: `OPENAI_API_KEY` (用于 LLM 推理)
- **可选**: `S2_API_KEY` (用于 Semantic Scholar 检索，缺失时会降级)
- **配置路径**: `IDEA_AGENT_SURVEY_CONFIG` (可选，默认使用 Survey Agent 的 deep_survey.yaml)

### 8.2 最快运行一次
```bash
python src/agents/idea_agent/scripts/run.py --topics "Graph Reasoning for LLMs" --max-turns 3
```

### 8.3 运行会发生什么
1. **knowledge_aquisition**: 拉取文献并填充 `memory["references"]` 与 `memory["paper_contents"]`。
2. **advanced_analysis**: 生成领域分析与关键问题。
3. **idea_generation**: 运行 Memory-Guided MCTS 并产出候选 idea。
4. **idea_evaluation**: 对最佳 idea 进行结构化打分。

> 如果 agent 判断当前方向不足，会触发 `re_analysis_replan` 以扩展 topic 与检索关键词。

### 8.4 产物在哪里
- `runs/<topic-slug-时间戳>/idea_result.json`：最终 idea 与 MCTS 搜索轨迹。
- `runs/<topic-slug-时间戳>/logs/ligagent.log`：完整运行日志。

### 8.5 如何调整效果
- **扩展搜索深度**: `src/agents/idea_agent/agent/mcts.py` 中的 `MCTSConfig.max_depth`。
- **增加迭代次数**: `MCTSConfig.max_iterations`。
- **更改模型**: `src/agents/idea_agent/agent/ligagent.py` 内的 `self.model` 或 MCTS 的 generation/evaluation model。
- **限制成本**: 降低 `--max-turns` 或减少 `branching_factor`。
