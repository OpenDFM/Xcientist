# 计划：修复 Prepare 下载流程与 MCP 配置

## 目标

* 修复 `run_experiment.sh --prepare` 后 `repos/` 与 `dataset_candidate/` 未正确准备的问题。

* 严格固定 prepare 产出顺序：`repos` → `datasets` → `venv` → `idea.md`（最后）。

* 将 `/hpc_stor03/sjtu_home/hanqi.li/.claude.json` 中 MCP 配置完整迁移到 `src/agents/experiment_agent/layers/base/agent.py`，并改为非本地依赖写法。

## 已确认问题（基于现有代码与日志）

* `PrepareAgent` 当前存在对 `idea.json` 顶层字段的隐式依赖，遇到不同结构时容易导致资源阶段被跳过。

* prepare 指令中“写 `idea.md`”与“下载 repos/datasets”并列，缺少“`repos`→`datasets`→`venv`→`idea.md`”的硬顺序约束。

* 系统提示与工具能力存在偏差，影响资源发现与下载稳定性。

* `base/agent.py` 中 MCP 配置为部分硬编码，且存在本地化/不安全写法，不满足“完整迁移 + 非本地格式”要求。

## 实施步骤

### 1) 去除 idea.json 格式依赖，强制网络检索驱动资源发现

* 修改文件：

  * `src/agents/experiment_agent/layers/prepare/agent.py`

* 具体改动：

  * 仅把 `idea.json` 作为“可读上下文”，不再要求任何固定字段结构；将完整文本/对象传入提示上下文。

  * 增加硬约束：必须先执行网络搜索（论文、GitHub、HuggingFace）再确定 repos/datasets 候选，不能只依赖 `idea.json` 内已有链接。

  * 为 `skip_repos/skip_datasets` 以外场景增加显式约束：一旦搜索得到候选，必须尝试 clone/download（best-effort）并记录失败原因。

### 2) 重排 prepare 产出顺序（repos → datasets → venv → idea.md）

* 修改文件：

  * `src/agents/experiment_agent/layers/prepare/agent.py`

  * `src/agents/experiment_agent/layers/prepare/prompts/system.txt`

* 具体改动：

  * 在用户提示中将 Required Outputs 改为分阶段流程：先 repos，再 datasets，再创建 venv，最后写 `idea.md`。

  * 在系统提示中新增硬性规则：`idea.md` 为最终汇总产物，只能在 repos/datasets/venv 阶段结束后写入；若资源未完成，需在文档里记录状态与原因后收尾。

  * 统一文案，避免“并列任务导致提前写文档”。

### 3) 提升 prepare 成功判定与可观测性

* 修改文件：

  * `src/agents/experiment_agent/layers/prepare/agent.py`

  * `run_experiment.sh`

* 具体改动：

  * 在 prepare 完成后增加后置检查：按顺序核对 repos、datasets、venv、idea.md 四阶段结果，防止静默“成功但未准备”。

  * 调整脚本流程，使 prepare 失败时返回非零并停止主流程（避免继续进入 main 造成误判）。

  * 增加关键阶段日志（网络检索次数、候选数量、下载尝试次数、成功/失败统计），便于从 `log_mem.txt` 直接定位问题。

### 4) 完整迁移 MCP 配置到 BaseAgent，改为非本地格式

* 修改文件：

  * `src/agents/experiment_agent/layers/base/agent.py`

* 具体改动：

  * 读取并映射 `.claude.json` 的 MCP server 集合到 `mcp_config`，覆盖 `github/thinking/filesystem/fetch/playwright/memory/MiniMax/context7`。

  * 将“本地依赖写法”改为非本地格式（重点是 command 必须使用 npx/uvx 拉起，而不是依赖预装可执行文件）：

    * 不写死本地绝对路径参数。

    * 不在源码硬编码密钥，改为环境变量注入。

    * 对必须路径参数改为运行时配置（环境变量或工作目录派生），保证可迁移性。

  * 按“可下载 command”改写 MCP：
    * `fetch` 从 `command: "mcp-fetch"` 改为 npx/uvx 形式（优先 `npx -y @kazuph/mcp-fetch`，保留 uvx 兼容方案）。
    * 对其余 server 逐一校验 command 是否为 npx/uvx 拉起；若不是，则替换为对应包启动方式。

  * 实施“下载并可用性预热”：
    * 在代码中新增 MCP 预检/预热步骤（首次启动前执行 `npx -y <pkg> --help` 或 uvx 等价命令）。
    * 预热失败时给出明确错误信息和修复建议，避免运行时才暴露问题。

  * 保持与 OpenHands `mcp_config` 结构兼容，避免回归已有 MCP 超时与工具发现逻辑。

### 5) 验证与回归

* 代码级验证：

  * 对修改的 Python 文件执行语法检查。

  * 校验 MCP 配置对象中各 server 的 `command` 均为 `npx` 或 `uvx`。

* 端到端验证：

  * 运行用户提供命令（含 `--prepare`）复现一次，确认：

    * `repos/` 有 clone 结果或明确失败记录；

    * `dataset_candidate/` 有下载结果或明确失败记录；

    * `idea.md` 的生成时间与内容位于流程末尾，并包含最终状态汇总。

    * MCP 启动日志中不再出现 `mcp-fetch` 这类预装命令直调，改为 npx/uvx 拉起并成功可用。

* 日志验证：

  * 在 `log_mem.txt` 中确认阶段顺序与统计信息符合预期。

## 交付结果

* 修复后，prepare 不再对 `idea.json` 结构有格式要求，只做读取与理解。

* prepare 将先做网络搜索，再按 `repos` → `datasets` → `venv` → `idea.md` 执行并落盘。

* `idea.md` 变为“最后产出、汇总全流程状态”的文档。

* MCP 配置完整且可迁移，不依赖本机绝对路径与源码内密钥。
