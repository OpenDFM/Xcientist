# Experiment Agent (SuperAgent)

## 概述

**SuperAgent** 是一个双层AI自动化系统，专为科学研究自动化而设计。它能够从研究提案（Research Proposal）自动生成可执行的科研代码，运行实验，分析结果，并基于反馈迭代优化代码，形成一个完整的科研自动化闭环。

## 核心架构

SuperAgent 采用双层架构设计，通过标准化协议进行层间通信：

### 1. Code Layer (工程层/代码生成层)
**职责**：从研究提案生成并维护科研代码库

**核心组件**：
- **Architect** (`architect.py`): 将研究提案转换为代码蓝图（Blueprint），定义项目结构、文件清单和技术栈
- **Manager** (`manager.py`): 管理代码生成流程，协调多个 Worker 并发生成代码文件
- **Worker** (`worker.py`): 执行具体的代码编写任务，一个 Worker 负责一个文件的实现
- **Integrator** (`integrator.py`): 验证生成的代码，检查语法错误、依赖问题，并运行初步测试

**工作流程**：
```
Research Proposal → Architect (Blueprint) → Manager (Task Distribution) 
→ Workers (Code Generation) → Integrator (Verification) → CodeManifest
```

### 2. Science Layer (科学层/实验执行层)
**职责**：使用生成的代码执行科学实验并分析结果

**核心组件**：
- **Architect** (`architect.py`): 基于研究目标设计实验计划（ExperimentPlan），定义实验步骤、参数空间和评估指标
- **Manager** (`manager.py`): 管理实验执行流程，调度实验任务并收集结果
- **Worker** (`worker.py`): 执行具体的实验脚本，运行训练/测试流程，记录实验数据
- **Integrator** (`integrator.py`): 分析实验结果，评估是否达到研究目标，生成优化建议

**工作流程**：
```
CodeManifest → Architect (ExperimentPlan) → Manager (Experiment Execution) 
→ Workers (Run Experiments) → Integrator (Result Analysis) → OptimizationTickets
```

### 3. 通信协议

#### CHP (Code Handover Protocol) - 代码交接协议
- **方向**：Code Layer → Science Layer
- **载体**：`CodeManifest`
- **内容**：项目根目录、入口点、可执行脚本清单、依赖信息

#### ORP (Optimization Request Protocol) - 优化请求协议
- **方向**：Science Layer → Code Layer
- **载体**：`OptimizationTicket`
- **内容**：问题类型（性能/Bug/配置等）、文件路径、问题描述、修复建议、优先级

## 主循环架构

SuperAgent 通过以下主循环实现科研自动化：

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Loop (max N iterations)            │
│                                                             │
│  ┌────────────────────┐                                    │
│  │  1. Engineering    │  生成代码                           │
│  │     Layer          │  → CodeManifest                    │
│  └────────┬───────────┘                                    │
│           │                                                 │
│           ▼                                                 │
│  ┌────────────────────┐                                    │
│  │  2. Science        │  运行实验                           │
│  │     Layer          │  → ExperimentResults               │
│  └────────┬───────────┘                                    │
│           │                                                 │
│           ▼                                                 │
│  ┌────────────────────┐                                    │
│  │  3. Analysis       │  分析结果                           │
│  │                    │  → OptimizationTickets             │
│  └────────┬───────────┘                                    │
│           │                                                 │
│           │  有问题？                                        │
│           ├─── Yes ──→ 优化代码（回到步骤1）                 │
│           │                                                 │
│           └─── No ──→ 科研目标达成！                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心数据模型

### Code Layer

#### Blueprint (代码蓝图)
```python
{
    "project_name": "实验项目名称",
    "description": "项目描述",
    "files": [                    # 文件清单
        {
            "path": "main.py",
            "purpose": "主程序入口",
            "dependencies": ["torch", "numpy"]
        }
    ],
    "dependencies": {...},        # 依赖管理
    "entry_point": "main.py",     # 程序入口
    "tech_stack": {...}           # 技术栈选择
}
```

#### CodeManifest (代码清单)
```python
{
    "project_root": "/path/to/project",
    "entry_point": "main.py",
    "entry_points": {             # 可执行脚本
        "train": "python train.py",
        "test": "python test.py"
    }
}
```

### Science Layer

#### ExperimentPlan (实验计划)
```python
{
    "goal": "验证模型在XX数据集上达到YY性能",
    "experiments": [
        {
            "name": "baseline",
            "script": "train.py",
            "params": {...},
            "expected_metrics": {"accuracy": 0.95}
        }
    ]
}
```

#### OptimizationTicket (优化票据)
```python
{
    "file_path": "model.py",
    "issue_type": "PerformanceIssue",
    "message": "模型收敛速度慢",
    "suggestion": "调整学习率或优化器",
    "priority": "HIGH",           # CRITICAL/HIGH/MEDIUM/LOW
    "metrics_context": {...}
}
```

## 状态管理与断点续传

### StateManager (状态管理器)

SuperAgent 实现了完善的状态管理系统，支持在任意阶段崩溃后恢复执行：

**状态持久化**：
```
workspaces/{experiment_id}/cached/
├── blueprints/              # 蓝图缓存
│   └── {hash}.json
└── execution/
    └── steps/               # 执行步骤（按步骤编号）
        ├── step_0000.json   # Code Layer 执行状态
        ├── step_0001.json   # Science Layer 执行状态
        └── ...
```

**步骤状态信息**：
- `step_index`: 步骤索引
- `phase`: 当前阶段（INIT/PLANNING/EXECUTION/VERIFICATION）
- `status`: 完成状态（RUNNING/COMPLETED/FAILED）
- `namespace`: 命名空间（"code" 或 "science"）
- `blueprint_id`: 关联的蓝图ID
- `tasks`: 任务状态字典（任务ID → 任务状态）
- `meta`: 额外元数据

### 断点续传机制

使用 `--resume` 参数可以从上次中断的地方继续执行：

```bash
python main.py --experiment exp001 --resume
```

**续传策略**：
1. 读取最新的 `step_*.json` 文件
2. 根据 `namespace` 判断中断在哪一层（Code/Science）
3. 根据 `tasks` 状态恢复未完成的任务
4. 跳过已完成的步骤，继续执行

## 目录结构

```
experiment_agent/
├── main.py                          # 主入口程序
├── __init__.py
│
├── layers/                          # 双层架构实现
│   ├── base/                        # 基础组件（所有层共享）
│   │   └── state.py                 # 状态管理器
│   │
│   ├── code/                        # Code Layer (工程层)
│   │   ├── entry.py                 # Code Layer 入口
│   │   ├── architect.py             # 蓝图设计师
│   │   ├── manager.py               # 代码管理器
│   │   ├── worker.py                # 代码工作者
│   │   ├── integrator.py            # 代码集成验证器
│   │   ├── traceback_parser.py      # 错误追踪解析
│   │   ├── schemas/                 # 数据模型
│   │   │   ├── blueprint.py         # 代码蓝图
│   │   │   ├── fix_blueprint.py     # 修复蓝图
│   │   │   ├── manifest.py          # 代码清单
│   │   │   ├── proposal.py          # 研究提案
│   │   │   ├── idea_parser.py       # 提案解析器
│   │   │   └── integration.py       # 集成结果
│   │   └── prompts/                 # LLM 提示模板
│   │
│   └── science/                     # Science Layer (科学层)
│       ├── entry.py                 # Science Layer 入口
│       ├── architect.py             # 实验设计师
│       ├── manager.py               # 实验管理器
│       ├── worker.py                # 实验执行者
│       ├── integrator.py            # 结果分析器
│       ├── schemas/                 # 数据模型
│       │   └── experiment.py        # 实验计划/结果/分析
│       └── prompts/                 # LLM 提示模板
│
├── shared/                          # 共享组件
│   ├── schemas/                     # 共享数据模型
│   │   ├── protocols.py             # CHP/ORP 协议定义
│   │   └── action_plan.py           # 执行计划
│   ├── tools/                       # 工具集
│   │   └── core.py                  # 核心工具（文件操作等）
│   ├── utils/                       # 工具函数
│   │   ├── config.py                # 配置管理
│   │   └── cache.py                 # 缓存管理
│   ├── logger/                      # 日志系统
│   └── exceptions.py                # 异常处理
│
├── docs/                            # 文档
│   ├── generate_diagrams.py        # 图表生成脚本
│   └── images/                      # 图片资源
│       └── dag_scheduling.png       # DAG调度示意图
│
└── workspaces/                      # 工作空间（运行时生成）
    └── {experiment_id}/             # 每个实验的工作空间
        ├── input/                   # 输入文件
        │   └── idea.json            # 研究提案
        ├── project/                 # 生成的代码项目
        ├── dataset/                 # 数据集
        ├── results/                 # 实验结果
        └── cached/                  # 缓存和状态
            ├── blueprints/          # 蓝图缓存
            └── execution/           # 执行状态
                └── steps/           # 步骤记录
```

## 使用方法

### 基本用法

```bash
# 运行完整的科研自动化流程
python main.py --experiment exp001

# 从断点续传
python main.py --experiment exp001 --resume

# 全新开始（清除之前的状态）
python main.py --experiment exp001 --fresh

# 只运行代码生成，跳过实验
python main.py --experiment exp001 --skip-science

# 限制最大循环次数
python main.py --experiment exp001 --max-loops 5

# 使用快速启发式分析（不使用LLM）
python main.py --experiment exp001 --quick-analysis

# 在 Science Architect 后停止（调试用）
python main.py --experiment exp001 --stop-after-science-architect

# 详细输出
python main.py --experiment exp001 --verbose
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--experiment, -e` | 实验ID（必需） | - |
| `--resume` | 从上次中断处继续 | False |
| `--fresh` | 全新开始，清除之前状态 | False |
| `--skip-science` | 跳过科学层，只生成代码 | False |
| `--max-loops` | 最大优化循环次数 | 3 |
| `--quick-analysis` | 使用快速分析替代LLM | False |
| `--stop-after-science-architect` | 在实验计划生成后停止 | False |
| `--verbose, -v` | 启用详细输出 | False |

### 输入格式

实验需要一个研究提案文件（JSON格式），位于：
```
workspaces/{experiment_id}/input/idea.json
```

**研究提案示例**：
```json
{
  "idea": {
    "title": "基于Transformer的时间序列预测",
    "description": "使用Transformer架构对多变量时间序列进行预测，评估其在金融数据上的表现",
    "hypothesis": "Transformer的自注意力机制能够更好地捕捉时间序列的长期依赖关系",
    "method": "在股票价格数据集上训练Transformer模型，与LSTM基线对比",
    "expected_outcome": "在MAE和RMSE指标上优于LSTM至少10%"
  },
  "requirements": {
    "dataset": "stock_prices.csv",
    "baseline": "LSTM",
    "metrics": ["MAE", "RMSE", "R2"],
    "constraints": {
      "max_training_time": "2h",
      "gpu_memory": "8GB"
    }
  }
}
```

## 工作流程详解

### Phase 1: Engineering Layer (工程层)

1. **Architect 阶段**
   - 读取研究提案
   - 分析研究目标和方法
   - 生成代码蓝图（Blueprint）：定义文件结构、依赖、技术栈

2. **Manager + Workers 阶段**
   - Manager 将蓝图分解为文件级任务
   - 为每个文件分配一个 Worker（支持并发）
   - Worker 使用 LLM 生成代码实现
   - 生成的代码写入项目目录

3. **Integrator 阶段**
   - 检查语法错误（pylint/flake8）
   - 验证依赖安装
   - 运行初步测试
   - 生成 CodeManifest 交接给 Science Layer

### Phase 2: Science Layer (科学层)

1. **Architect 阶段**
   - 分析研究目标和可用代码
   - 设计实验计划（ExperimentPlan）
   - 定义实验步骤、参数空间、评估指标

2. **Manager + Workers 阶段**
   - Manager 调度实验任务
   - Worker 执行训练/测试脚本
   - 记录实验日志、指标、输出

3. **Integrator 阶段**
   - 分析实验结果
   - 评估是否达到研究目标
   - 如果未达标，生成 OptimizationTickets

### Phase 3: Optimization Loop (优化循环)

1. **接收 OptimizationTickets**
   - Science Layer 发送问题列表
   - 按优先级（CRITICAL → HIGH → MEDIUM → LOW）排序

2. **Code Layer 修复**
   - 生成 FixBlueprint
   - Manager 协调 Workers 修复代码
   - Integrator 验证修复效果

3. **迭代或终止**
   - 如果修复成功，回到 Science Layer 重新实验
   - 如果达到最大循环次数或目标达成，终止

## 关键特性

### 1. 智能错误恢复
- 自动解析错误堆栈（`traceback_parser.py`）
- 精准定位问题代码行
- 生成针对性修复建议

### 2. 并发执行
- Code Workers 并发生成多个文件
- Science Workers 并发运行多组实验
- 提高执行效率

### 3. 缓存机制
- Blueprint 缓存：避免重复设计
- 实验结果缓存：支持增量分析
- 基于内容哈希的智能缓存

### 4. 安全沙箱
- `SecurityContext` 限制文件操作范围
- 防止恶意代码破坏系统
- 隔离实验环境

### 5. 灵活的分析模式
- **LLM分析**：深度分析实验结果，生成详细建议
- **快速分析**：基于启发式规则的快速评估（`--quick-analysis`）

## 高级功能

### 动态 DAG 调度

Science Layer 支持基于依赖关系的 DAG（有向无环图）调度：
- 自动识别实验间的依赖关系
- 并行执行独立实验
- 串行执行有依赖的实验
- 优化整体执行时间

参考图表：`docs/images/dag_scheduling.png`

### 多模态支持

- 支持不同类型的研究任务：
  - 机器学习训练与评估
  - 数据分析与可视化
  - 算法性能测试
  - 科学计算与模拟

### 实验追踪

- 每次实验自动记录：
  - 代码版本（Blueprint ID）
  - 超参数配置
  - 运行日志
  - 性能指标
  - 生成的可视化图表

## 依赖要求

### Python 环境
- Python >= 3.8

### 核心依赖
- `openai`: LLM API 调用
- `pydantic`: 数据验证和序列化
- `asyncio`: 异步并发执行

### 可选依赖（根据研究领域）
- **机器学习**：`torch`, `tensorflow`, `scikit-learn`
- **数据处理**：`pandas`, `numpy`
- **可视化**：`matplotlib`, `seaborn`

## 配置

配置文件位于 `shared/utils/config.py`，支持以下配置：

```python
# OpenAI API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# 工作空间配置
WORKSPACE_ROOT = "/path/to/workspaces"

# 并发配置
MAX_CONCURRENT_WORKERS = 5

# 缓存配置
ENABLE_CACHE = True
```

## 常见问题

### Q1: 如何处理 API Rate Limit？
SuperAgent 内置了速率限制检测和优雅退出机制（`exceptions.py`）。遇到速率限制时会自动保存状态，可以稍后使用 `--resume` 继续。

### Q2: 如何自定义研究领域的模板？
修改 `layers/{code,science}/prompts/` 目录下的提示模板，可以针对特定领域（如CV、NLP、RL）定制生成策略。

### Q3: 生成的代码质量如何保证？
- Integrator 会进行多重验证（语法、依赖、测试）
- 优化循环会根据实验结果迭代改进
- 支持人工审查和干预

### Q4: 如何扩展新的问题类型？
在 `shared/schemas/protocols.py` 中添加新的 `TicketType`，并在 Integrator 中实现相应的检测逻辑。

## 最佳实践

1. **清晰的研究提案**：提供详细的目标、方法、评估指标
2. **合理的资源约束**：在提案中指定时间、内存等限制
3. **渐进式优化**：从简单基线开始，逐步增加复杂度
4. **定期检查点**：利用 `--resume` 进行长时间实验
5. **日志分析**：详细查看生成的日志文件定位问题

## 开发指南

### 添加新的 Agent 组件

1. 在 `layers/{code,science}/` 下创建新模块
2. 实现标准接口（参考现有 Agent）
3. 在 `entry.py` 中注册新组件
4. 更新 Schema 和 Prompt

### 扩展新的通信协议

1. 在 `shared/schemas/protocols.py` 定义新协议
2. 在两层的 Integrator 中实现协议转换
3. 更新主循环逻辑（`main.py`）

### 调试技巧

```bash
# 查看详细日志
python main.py --experiment exp001 --verbose

# 只运行 Architect 阶段
python main.py --experiment exp001 --stop-after-science-architect

# 跳过耗时的实验
python main.py --experiment exp001 --quick-analysis

# 检查状态文件
cat workspaces/exp001/cached/execution/steps/step_*.json
```

## 性能优化

- **并发度调整**：修改 `MAX_CONCURRENT_WORKERS` 适配机器性能
- **缓存策略**：启用缓存避免重复计算
- **增量实验**：使用 `--resume` 继续未完成的实验
- **快速分析**：对于简单任务使用 `--quick-analysis`

## 许可与引用

（根据项目实际情况填写）

## 贡献指南

欢迎贡献！请遵循以下流程：
1. Fork 项目
2. 创建特性分支
3. 提交代码并通过测试
4. 发起 Pull Request

## 联系方式

（根据项目实际情况填写）

---

**版本**: 1.0  
**最后更新**: 2025-12  
**维护者**: Research Agent Team

