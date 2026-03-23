# Experiment Agent 5 项优化实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 优化 experiment_agent 架构：统一基类复用、Agent 实例复用、路径集中管理、状态结构化、配置集中管理

**Architecture:** 基于 OpenHands SDK 范式，改进现有双层架构，减少代码重复，提高可维护性

**Tech Stack:** Python, OpenHands SDK, Pydantic

---

## Task 1: 重构 MasterAgent 使用 UnifiedExperimentAgent

**Files:**
- Modify: `src/agents/experiment_agent/master.py`

**Step 1: Read MasterAgent class definition**

Run: Read master.py lines 185-260 to understand current MasterAgent initialization

**Step 2: Modify MasterAgent to inherit from UnifiedExperimentAgent**

```python
# Change from:
from src.agents.experiment_agent.layers.base.agent import OpenHandsBaseAgent

class MasterAgent(OpenHandsBaseAgent):

# To:
from src.agents.experiment_agent.shared.agents.base import UnifiedExperimentAgent

class MasterAgent(UnifiedExperimentAgent):
```

**Step 3: Update __init__ to use UnifiedExperimentAgent**

Replace MasterAgent.__init__ with:
```python
def __init__(
    self,
    experiment_id: str,
    idea_path: str,
    workspace_root: str,
    project_root: str,
    model: str = MASTER_AGENT_MODEL,
    verbose: bool = True,
    max_iterations: int = 10,
):
    self.project_root = project_root

    super().__init__(
        agent_type="Master",
        model=model,
        workspace_root=workspace_root,
        experiment_id=experiment_id,
        idea_path=idea_path,
        verbose=verbose,
    )

    self.max_iterations = max_iterations

    # State files
    self.agent_md_path = os.path.join(workspace_root, "agent.md")
    self.agent_json_path = os.path.join(workspace_root, "agent_state.json")

    # Current state
    self.current_iteration = 1
    self.state = AgentState(
        iteration=1,
        phase="planning_code",
        decision="",
    )
```

**Step 4: Verify syntax**

Run: `python3 -m py_compile master.py`

---

## Task 2: 复用 Agent 实例 (Agent Pool)

**Files:**
- Modify: `src/agents/experiment_agent/master.py`

**Step 1: Add agent cache to MasterAgent**

Add to MasterAgent.__init__:
```python
# Agent instance cache for reuse
self._code_agent_instance = None
self._science_agent_instance = None
```

**Step 2: Add get_code_agent method**

Add after __init__:
```python
def get_code_agent(self, plan: str):
    """Get or create cached CodeAgent instance."""
    if self._code_agent_instance is None:
        from src.agents.experiment_agent.code_agent import CodeAgent
        self._code_agent_instance = CodeAgent(
            experiment_id=self.experiment_id,
            idea_path=self.idea_path,
            project_root=self.project_root,
            workspace_root=self.workspace_root,
            plan=plan,
            model=CODE_ARCHITECT_MODEL,
            verbose=self.verbose,
        )
    else:
        # Update plan for reuse
        self._code_agent_instance.plan = plan
    return self._code_agent_instance

def get_science_agent(self, plan: str, code_summary: str, code_usage: str):
    """Get or create cached ScienceAgent instance."""
    if self._science_agent_instance is None:
        from src.agents.experiment_agent.science_agent import ScienceAgent
        self._science_agent_instance = ScienceAgent(
            experiment_id=self.experiment_id,
            idea_path=self.idea_path,
            project_root=self.project_root,
            workspace_root=self.workspace_root,
            plan=plan,
            code_summary=code_summary,
            code_usage=code_usage,
            model=SCIENCE_AGENT_MODEL,
            verbose=self.verbose,
        )
    else:
        # Update parameters for reuse
        self._science_agent_instance.plan = plan
        self._science_agent_instance.code_summary = code_summary
        self._science_agent_instance.code_usage = code_usage
    return self._science_agent_instance
```

**Step 3: Update _call_code_agent to use cached instance**

Replace `_call_code_agent` method:
```python
async def _call_code_agent(self, plan: str) -> Dict[str, Any]:
    """Call the Code Agent with the plan (with retry)."""
    self.state.phase = "coding"
    self._save_agent_md()

    agent = self.get_code_agent(plan)

    async def run_agent():
        return await agent.execute()

    result = await self.run_with_retry(run_agent, max_retries=MAX_RETRIES)

    # Check if summary file exists and read it
    summary_content = self.read_code_summary()

    if summary_content:
        result["summary"] = summary_content
    elif result.get("summary"):
        self.write_code_summary(result["summary"])

    self.state.code_result = result.get("summary", "")
    self._save_agent_md()
    return result
```

**Step 4: Verify syntax**

Run: `python3 -m py_compile master.py`

---

## Task 3: 使用 WorkspaceManager 集中管理路径

**Files:**
- Modify: `src/agents/experiment_agent/master.py`
- Modify: `src/agents/experiment_agent/shared/agents/base.py`

**Step 1: Import WorkspaceManager in master.py**

Add after existing imports:
```python
from src.agents.experiment_agent.shared.utils.workspace import WorkspaceManager
```

**Step 2: Add WorkspaceManager to MasterAgent**

Add to __init__:
```python
# Centralized workspace management
self.workspace = WorkspaceManager(workspace_root, experiment_id)
```

**Step 3: Update file paths to use WorkspaceManager**

Replace hardcoded paths:
```python
# Before:
self.agent_md_path = os.path.join(workspace_root, "agent.md")
self.agent_json_path = os.path.join(workspace_root, "agent_state.json")

# After:
self.agent_md_path = os.path.join(self.workspace.workspace_dir, "agent.md")
self.agent_json_path = os.path.join(self.workspace.workspace_dir, "agent_state.json")
```

**Step 4: Add WorkspaceManager properties to base class**

In shared/agents/base.py, add:
```python
@property
def workspace_manager(self) -> 'WorkspaceManager':
    """Get workspace manager instance."""
    if not hasattr(self, '_workspace') or self._workspace is None:
        from src.agents.experiment_agent.shared.utils.workspace import WorkspaceManager
        self._workspace = WorkspaceManager(self.workspace_root, self.experiment_id)
    return self._workspace
```

**Step 5: Verify syntax**

Run: `python3 -m py_compile master.py shared/agents/base.py`

---

## Task 4: 状态持久化改为 JSON 格式

**Files:**
- Modify: `src/agents/experiment_agent/master.py`

**Step 1: Add AgentState.to_json and from_json methods**

Add to AgentState class:
```python
def to_json(self) -> dict:
    """Convert to JSON-serializable dict."""
    return {
        "iteration": self.iteration,
        "phase": self.phase,
        "decision": self.decision,
        "code_plan": self.code_plan,
        "code_result": self.code_result,
        "science_plan": self.science_plan,
        "science_result": self.science_result,
        "conclusion": self.conclusion,
    }

@classmethod
def from_json(cls, data: dict) -> 'AgentState':
    """Create AgentState from JSON dict."""
    return cls(
        iteration=data.get("iteration", 1),
        phase=data.get("phase", "planning_code"),
        decision=data.get("decision", ""),
        code_plan=data.get("code_plan", ""),
        code_result=data.get("code_result", ""),
        science_plan=data.get("science_plan", ""),
        science_result=data.get("science_result", ""),
        conclusion=data.get("conclusion", ""),
    )
```

**Step 2: Update _save_agent_md to also save JSON**

Replace _save_agent_md:
```python
def _save_agent_md(self):
    """Save current state to agent.md and agent_state.json."""
    os.makedirs(os.path.dirname(self.agent_md_path), exist_ok=True)

    # Save Markdown
    with open(self.agent_md_path, "w", encoding="utf-8") as f:
        f.write(self.state.to_markdown())

    # Save JSON for reliable restoration
    json_path = self.agent_json_path
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(self.state.to_json(), f, indent=2, ensure_ascii=False)
```

**Step 3: Update _load_agent_md to prefer JSON**

Replace _load_agent_md:
```python
def _load_agent_md(self) -> Optional[AgentState]:
    """Load previous state from agent_state.json (preferred) or agent.md (fallback)."""
    # Try JSON first (more reliable)
    if os.path.exists(self.agent_json_path):
        try:
            with open(self.agent_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return AgentState.from_json(data)
        except Exception as e:
            logger.warning(f"Failed to load agent_state.json: {e}")

    # Fallback to Markdown
    if os.path.exists(self.agent_md_path):
        try:
            content = open(self.agent_md_path, "r", encoding="utf-8").read()
            # ... existing markdown parsing logic
        except Exception as e:
            logger.warning(f"Failed to load agent.md: {e}")

    return None
```

**Step 4: Add json import**

Add to master.py imports:
```python
import json
```

**Step 5: Verify syntax**

Run: `python3 -m py_compile master.py`

---

## Task 5: 统一配置管理

**Files:**
- Modify: `src/agents/experiment_agent/shared/utils/config.py`

**Step 1: Add dataclasses for configuration**

Add to config.py:
```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    model: str = "MiniMax-M2.5"
    max_turns: int = 10000
    condenser_max_size: int = 20
    condenser_keep_first: int = 2

@dataclass
class ExperimentConfig:
    """Main configuration for experiment agent."""
    experiment_id: str = ""
    workspace_root: str = ""
    project_root: str = ""
    max_iterations: int = 10
    max_retries: int = 3
    retry_delay_base: float = 2.0
    retry_backoff: float = 1.5

    # Agent configurations
    master: AgentConfig = field(default_factory=lambda: AgentConfig(model=MASTER_AGENT_MODEL))
    code: AgentConfig = field(default_factory=lambda: AgentConfig(model=CODE_ARCHITECT_MODEL))
    science: AgentConfig = field(default_factory=lambda: AgentConfig(model=SCIENCE_AGENT_MODEL))

    @classmethod
    def from_experiment(cls, experiment_id: str, workspace_root: str, project_root: str) -> 'ExperimentConfig':
        """Create config from experiment parameters."""
        return cls(
            experiment_id=experiment_id,
            workspace_root=workspace_root,
            project_root=project_root,
        )

    def get_agent_config(self, agent_type: str) -> AgentConfig:
        """Get config for specific agent type."""
        return getattr(self, agent_type.lower(), self.master)
```

**Step 2: Add global config instance**

Add at end of config.py:
```python
# Global experiment config (set during initialization)
_current_config: Optional[ExperimentConfig] = None

def get_experiment_config() -> Optional[ExperimentConfig]:
    """Get current experiment configuration."""
    return _current_config

def set_experiment_config(config: ExperimentConfig):
    """Set current experiment configuration."""
    global _current_config
    _current_config = config
```

**Step 3: Update MasterAgent to use config**

In master.py, add:
```python
from src.agents.experiment_agent.shared.utils.config import get_experiment_config, set_experiment_config, ExperimentConfig
```

In MasterAgent.__init__, add at the end:
```python
# Register this config globally
config = ExperimentConfig.from_experiment(
    experiment_id=experiment_id,
    workspace_root=workspace_root,
    project_root=project_root,
)
config.master.model = self.model
set_experiment_config(config)
```

**Step 4: Verify syntax**

Run: `python3 -m py_compile shared/utils/config.py master.py`

---

## Verification Steps

After all tasks complete:

1. **Syntax check:**
```bash
python3 -m py_compile master.py code_agent.py science_agent.py shared/agents/base.py shared/utils/workspace.py shared/utils/config.py
```

2. **Import check:**
```bash
python3 -c "from src.agents.experiment_agent.master import MasterAgent; print('OK')"
```

3. **Run a quick test:**
```bash
cd /aistor/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent && python3 -c "
from src.agents.experiment_agent.master import MasterAgent, AgentState
from src.agents.experiment_agent.shared.utils.workspace import WorkspaceManager
from src.agents.experiment_agent.shared.utils.config import ExperimentConfig, get_experiment_config

# Test AgentState JSON serialization
state = AgentState(iteration=3, phase='experimenting', decision='EXP_NEEDED')
json_data = state.to_json()
restored = AgentState.from_json(json_data)
assert restored.iteration == 3
print('AgentState JSON: OK')

# Test WorkspaceManager
wm = WorkspaceManager('/tmp/test_workspace', 'test_exp')
assert wm.workspace_dir == '/tmp/test_workspace'
print('WorkspaceManager: OK')

# Test ExperimentConfig
config = ExperimentConfig.from_experiment('exp1', '/ws', '/proj')
assert config.experiment_id == 'exp1'
print('ExperimentConfig: OK')

print('All tests passed!')
"
```
