# Experiment Agent Skills 扩展计划

> **For Claude:** Use superpowers:writing-plans to create implementation plan.

**Goal:** 为 experiment_agent 添加 6 个核心科学 Skills，增强 AI/ML 研究能力

**Architecture:** 基于现有 Skills 框架 (SKILL.md 格式)，遵循 OpenHands AgentContext 加载模式

**Tech Stack:** OpenHands Skills System, Markdown (SKILL.md)

---

## 新增 Skills 列表

### 1. scientific-visualization
- **功能**: 科学数据可视化
- **库**: matplotlib, plotly, seaborn
- **触发词**: plot, visualize, chart, graph, figure, draw

### 2. statistical-analysis
- **功能**: 统计假设检验与显著性分析
- **库**: scipy, statsmodels
- **触发词**: statistic, hypothesis, p-value, significance, test

### 3. data-processing
- **功能**: 数据清洗、转换、特征工程
- **库**: pandas, polars, numpy
- **触发词**: preprocess, clean, transform, feature, encode

### 4. machine-learning
- **功能**: ML 模型训练与评估流程
- **库**: scikit-learn, pytorch
- **触发词**: train, model, predict, classifier, regressor

### 5. scientific-writing
- **功能**: 论文与报告撰写规范
- **触发词**: write, paper, report, section, abstract

### 6. literature-review
- **功能**: 学术文献搜索与综述
- **触发词**: review, literature, search, paper, arxiv

---

## Skills 目录结构

```
skills/
├── constitution/SKILL.md          (existing)
├── code-blueprint/SKILL.md         (existing)
├── code-implementation/SKILL.md    (existing)
├── experiment-planning/SKILL.md    (existing)
├── experiment-execution/SKILL.md   (existing)
├── analysis-framework/SKILL.md     (existing)
│
├── scientific-visualization/       (NEW)
│   └── SKILL.md
├── statistical-analysis/           (NEW)
│   └── SKILL.md
├── data-processing/                (NEW)
│   └── SKILL.md
├── machine-learning/               (NEW)
│   └── SKILL.md
├── scientific-writing/             (NEW)
│   └── SKILL.md
└── literature-review/              (NEW)
    └── SKILL.md
```

---

## SKILL.md 格式规范

每个 Skill 包含：
- YAML frontmatter (name, description, triggers, license)
- # 标题
- ## Mission (使命)
- ## Key Principles (关键原则)
- ## Usage Guidelines (使用指南)
- ## Examples (示例)

---

## 验证方法

1. 运行 `python -c "from skills import print_loaded_skills; print_loaded_skills()"`
2. 确认所有 12 个 Skills 被加载
3. 测试触发词是否正确激活对应 Skill
