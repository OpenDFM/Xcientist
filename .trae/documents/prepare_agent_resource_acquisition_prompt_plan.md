# Prepare Agent 资源拉取强化计划

## 目标
- 强化 Prepare Agent 的提示与校验逻辑，确保其在 prepare 阶段充分使用可用工具，**实际完成** repo 与 dataset 下载（而非只写计划）。
- 明确要求 Prepare Agent 读取系统环境变量，并仅使用有价值变量（如 token、API endpoint、代理、缓存目录等）辅助下载与检索。
- 保持现有输出契约不变：`repos/`、`dataset_candidate/`、`project/venv/`、`idea.md`。

## 现状与问题
- 当前 `system.txt` 已要求“repos -> datasets -> venv -> idea.md”，也要求 bash 执行下载，但对“环境变量如何读取并实际使用”约束不足。
- 当前 `PrepareAgent._build_user_prompt()` 仅传入路径与 flags，未注入“候选环境变量清单 + 使用规则”。
- 当前 `_assert_prepare_outputs()` 只做产物存在性检查，未强约束“下载动作证据”与“环境变量使用记录”。

## 实施范围
- `src/agents/experiment_agent/layers/prepare/prompts/system.txt`
- `src/agents/experiment_agent/layers/prepare/agent.py`
- （可选）`src/agents/experiment_agent/layers/prepare/prompts/` 下新增辅助片段文件（如 env 使用规范模板）

## 详细实施步骤

### 1) 强化 Prepare System Prompt（核心）
- 在 `system.txt` 新增“环境变量读取与使用”章节，明确：
  - 在资源下载前执行环境变量检查（例如 `env | sort` 或按白名单读取）。
  - 只允许记录变量名与用途，不得在日志/idea.md 输出敏感值。
  - 必须在 `idea.md` 的 Meta/资源章节记录“已使用环境变量名 + 用途 + 影响步骤”。
- 在资源获取章节补充“可执行动作证据”要求：
  - 每个 repo 下载后提供验证动作（目录列举、git 远程信息、分支/提交信息）。
  - 每个 dataset 下载后提供验证动作（目录/文件清单、样本结构摘要）。
  - 若下载失败，必须记录失败原因、重试策略、替代来源与最终状态。
- 在工具使用策略中强调：
  - 优先 GitHub MCP 搜索 repo、Web 搜索补充；
  - 必须混合使用搜索 + bash 执行，不可只做网页调研。

### 2) 增强 User Prompt 注入（让执行更稳定）
- 在 `PrepareAgent._build_user_prompt()` 增加动态上下文：
  - 注入“候选环境变量名列表”（只名字，不含值）；
  - 注入“当前 skip 参数语义与必须执行项”；
  - 注入“下载完成判据清单”。
- 候选变量来源规则：
  - 从 `os.environ` 中筛选关键词（`TOKEN/KEY/API/ENDPOINT/PROXY/HF/GITHUB/CACHE` 等）；
  - 去重并限制数量，避免 prompt 过长。

### 3) 增加 Prepare 结果校验（防止空跑）
- 在 `agent.py` 中扩展准备完成校验：
  - 除现有目录存在性外，增加“非空与结构合理”检查；
  - repo 至少包含有效 clone 证据（目录项 + `.git` 或 remote 信息）；
  - dataset 至少包含一个成功下载数据目录或明确失败报告。
- 对 `idea.md` 增加最小结构检查：
  - 包含“Environment Variables Used”（或等价章节）；
  - 包含下载状态汇总（成功/失败/跳过及原因）。

### 4) 对齐“工具充分利用”的行为约束
- 在 prompt 内加入硬约束：
  - 必须至少一次使用 GitHub 搜索工具定位仓库；
  - 必须执行实际下载命令（git clone / python 下载）并验证；
  - 必须输出操作证据摘要（命令类别与结果，不泄露密钥）。
- 保持与现有 Operational Constraints 一致，不破坏原有流程顺序。

### 5) 文档输出规范完善
- 在 `idea.md` 模板要求中新增小节：
  - `## Environment Variables Used`：变量名、用途、应用步骤、是否必需。
  - `## Resource Acquisition Log`：repo/dataset 的来源、执行方式、验证结果、失败与替代方案。
- 要求最终 checklist 覆盖：
  - 是否读取并使用了必要环境变量；
  - 是否完成 repo/dataset 实际下载与验证。

## 验证计划（在 conda `openhands` 环境执行）
- 静态验证：
  - 检查修改后的 prompt 文本包含新增硬约束关键词；
  - 运行 `py_compile` 确认 `prepare/agent.py` 无语法错误。
- 行为验证（最小可复现）：
  - 执行一次 prepare（不 skip）；
  - 检查 `repos/`、`dataset_candidate/`、`project/venv/`、`idea.md` 是否满足新约束；
  - 检查 `idea.md` 中存在环境变量使用小节且无敏感值泄露。
- 失败路径验证：
  - 模拟不可下载数据源，确认会在 `idea.md` 记录失败原因与替代方案，而非静默通过。

## 风险与缓解
- 风险：环境变量过多导致 prompt 过长。  
  缓解：仅注入筛选后的变量名并限制上限。
- 风险：外部资源不稳定导致准备失败。  
  缓解：要求记录失败与替代来源，并以“可审计日志”方式保留证据。
- 风险：新增校验过严导致历史任务不兼容。  
  缓解：将新校验设计为“强提示 + 可解释失败”，避免误报硬中断。

## 交付结果
- Prepare 提示词更强，能驱动 agent 真正完成 repo/dataset 拉取。
- Prepare 输出可审计（有证据、有失败归因、有环境变量使用记录）。
- 不泄露密钥前提下，最大化利用系统环境变量提升资源获取成功率。
