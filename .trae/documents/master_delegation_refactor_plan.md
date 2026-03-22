# Master Delegation 重构计划

## 目标与范围
- 将 `master.py` 从“外部 Python 迭代调用 `run_code_agent/run_science_agent`”重构为“Master 通过 delegation 工具在 agent 内部委派 Code/Science 子代理”。
- 让 Master 自主决策是否需要：
  - 生成并执行 code plan；
  - 生成并执行 science plan；
  - 或直接进入 ablation results 汇总（当证据已充分）。
- 在最终 `ablation_results.json` 的 `components` 每个组件中新增 `method_context` 字段，用于简述该组件对应的 memory/context。
- 在实验阶段明确要求检查并使用系统环境变量（如 API key）中“必要且最小”的子集。

## 现状确认
- 当前 Master 在 `run_orchestration()` 中通过 Python 逻辑调用 `_write_code_plan/_call_code_agent/_write_science_plan/_call_science_agent`，不是 delegation 驱动。
- 当前 Science 的 ablation JSON 模板与 schema 尚无 `method_context` 字段。
- 仓库存在 OpenHands delegation 能力（`DelegateTool` + `register_agent` + `spawn/delegate` 协议），可作为实现基线。

## 设计方案

### 1) Master 改为“单会话编排 + delegation 调用”
- 在 `MasterAgent` 工具集中加入 `DelegateTool`（并完成必要注册）。
- 为 delegation 注册两个子代理类型（例如 `code_agent`、`science_agent`），其工厂函数封装现有 Code/Science 执行入口，保证子代理独立上下文。
- 将 `run_orchestration()` 的主流程从“Python 外部迭代”改为“Master `self.run(...)` 内部执行”：
  - Master 读取 idea 与已有产物（`code_summary.md`、`science_report.md`、`ablation_results.json`）；
  - Master 自主决定下一动作；
  - 通过 delegation 调用对应子代理执行任务；
  - 在同一会话内根据返回结果继续决策，直到收敛或达到迭代上限。
- 保留状态文件写入（`agent.md`）与最终报告产出，避免破坏现有可观测性。

### 2) Master 决策策略显式化
- 在 Master system prompt 增加明确决策框架：
  - `CODE_NEEDED`：代码实现不足或错误需先补；
  - `EXP_NEEDED`：需要新增/补充实验验证；
  - `CONVERGED`：已有足够证据，可直接整理结论与 ablation results。
- 增加“跳过不必要规划”的指令：
  - 若当前证据足够，不强制再写 code/science plan，直接进入结果归档与结论阶段。
- 明确要求“每次决策基于真实文件内容”，避免仅基于短摘要做判断。

### 3) 子代理委派输入输出约定
- 统一 delegation 任务载荷：
  - Code 任务：包含 `idea`、已有问题、目标改动、验收标准、输出文件要求（`code_summary.md`）。
  - Science 任务：包含 `idea`、代码现状、实验目标、ablation 输出要求、输出文件要求（`science_report.md` + `ablation_results.json`）。
- 要求子代理返回结构化摘要（状态、关键产物路径、主要结论、失败原因），便于 Master 继续决策。
- 在失败分支定义兜底策略（重试/改写任务/终止并标记 IMPOSSIBLE）。

### 4) Ablation Schema 与生成约束升级（新增 method_context）
- 在 `layers/base/schemas.py` 的 `AblationResult` 增加字段：
  - `method_context: str`（组件 memory/context 简述）。
- 同步更新 `science_agent.py` 的用户提示 JSON 模板，要求每个组件包含 `method_context`。
- 保持与现有字段兼容，不移除旧字段；新增字段作为必填（若需平滑兼容则先设为可选并在 Master 侧补齐）。

### 5) 实验阶段环境变量检查与最小使用原则
- 在 Science 委派任务说明中加入硬性步骤：
  - 启动实验前先检查环境变量；
  - 仅选择实验必需的变量注入命令/流程（如 API keys、endpoint、token）；
  - 不在日志或报告中泄露敏感值，仅记录“已检测并使用的变量名”。
- 如项目已有统一配置入口，优先复用，避免重复实现 env 解析。

### 6) 兼容性与迁移处理
- 逐步移除（或降级为兼容包装）`master.py` 中直接调用 `run_code_agent/run_science_agent` 的路径，避免双轨逻辑长期共存。
- 保留对已有输出文件路径与 checkpoint/状态语义的兼容，降低历史实验目录受影响概率。

## 实施步骤（按执行顺序）
1. 改造 `master.py`：接入 delegation 工具与子代理注册。
2. 重写 Master 编排提示与执行入口，使决策与委派在 agent 内闭环完成。
3. 调整 `run_orchestration()`：以 delegation 结果驱动状态流转与收敛判定。
4. 更新 `AblationResult` schema，新增 `method_context` 字段。
5. 更新 `science_agent.py` 提示模板与输出约束，确保生成 `method_context`。
6. 补充 Master 对 ablation 结果的最终校验/补齐逻辑（缺失时给出合理 fallback 文本）。
7. 回归验证：最小样例跑通，检查委派调用链、决策分支与输出文件结构。

## 验证计划
- 前置环境：
  - 使用当前 conda 环境 `openhands` 执行验证与回归（不切换到其他环境）。
- 静态验证：
  - 类型检查与导入检查（确保 delegation 相关 API 可用）；
  - JSON schema 对齐检查（`method_context` 字段存在且可解析）。
- 运行验证（至少覆盖三类场景）：
  - 场景 A：需先 code 再 science；
  - 场景 B：仅需 science（已有代码）；
  - 场景 C：证据充足直接汇总结论与 ablation results。
- 产物验证：
  - `ablation_results.json` 中每个 `components.*` 均含 `method_context`；
  - `agent.md` 状态流转与最终 `final_report.md` 一致；
  - 实验日志不泄露敏感环境变量值。

## 风险与缓解
- delegation 工具接入差异（版本/API 变动）：先对齐现仓库可用实现，必要时做兼容分支。
- 子代理返回不稳定：增加结构化输出约束与重试策略。
- 新字段导致旧数据解析失败：采用兼容迁移（过渡期允许缺省并在 Master 统一补齐）。
