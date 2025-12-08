"""
Experiment Execute Agent - Autonomous Researcher.

This agent acts as an autonomous researcher who can execute code, monitor results,
and iteratively tune hyperparameters to achieve research goals.
"""

import os
from typing import Optional, Any

from agents import Agent, Runner, RunConfig, ModelSettings

from src.agents.experiment_agent.sub_agents.experiment_execute.output_schemas import (
    ExperimentExecuteOutput,
)
from src.agents.experiment_agent.logger import create_verbose_hooks
from src.agents.experiment_agent.utils.print_utils import *
from src.agents.experiment_agent.utils.json_utils import (
    extract_and_parse_json,
    JSONParseError,
)


# Hand-written JSON output instruction for ExperimentExecuteOutput
EXPERIMENT_EXECUTE_JSON_OUTPUT_INSTRUCTION = """
## Required JSON Output Format: ExperimentExecuteOutput

You MUST output a JSON object with this EXACT structure:

```json
{
  "execution_status": "success",
  "has_error": false,
  "error_message": null,
  "output_files": [
    {
      "file_path": "/path/to/project/logs/baseline_exp.log",
      "file_type": "log",
      "description": "Baseline experiment training log",
      "run_command": "python train.py --method baseline --dataset mnist",
      "run_config": "lr=0.001, epochs=10, batch_size=32"
    },
    {
      "file_path": "/path/to/project/results/model_best.pt",
      "file_type": "checkpoint",
      "description": "Best model checkpoint",
      "run_command": "",
      "run_config": ""
    }
  ],
  "log_path": "/path/to/project/logs/baseline_exp.log",
  "experiment_metrics": "{\\"accuracy\\": 0.95, \\"loss\\": 0.23}",
  "execution_summary": "Ran baseline and proposed method on MNIST. Proposed achieved 95% accuracy vs baseline 87%.",
  "stdout_preview": "Epoch 10/10: loss=0.23, acc=0.95",
  "stderr_preview": ""
}
```

### Field Descriptions:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `execution_status` | string | YES | "success", "partial", "error", "timeout", "interrupted", "skipped" |
| `has_error` | boolean | YES | Whether any error occurred |
| `error_message` | string or null | NO | Error message if failed |
| `output_files` | array or null | NO | List of ExperimentFile objects |
| `log_path` | string | NO | Path to primary log file |
| `experiment_metrics` | string | NO | JSON string of best metrics |
| `execution_summary` | string | NO | Human-readable summary |
| `stdout_preview` | string | NO | Preview of stdout |
| `stderr_preview` | string | NO | Preview of stderr if errors |

### ExperimentFile Object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | string | YES | Absolute path to file |
| `file_type` | string | YES | "log", "result", "checkpoint", "config", "plot", "other" |
| `description` | string | YES | What this file contains |
| `run_command` | string | NO | Command used to generate this file |
| `run_config` | string | NO | Key hyperparameters |

⚠️ **CRITICAL**: Output ONLY valid JSON, no markdown explanations!
"""


# --- AGENT FACTORIES ---


def create_execute_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    log_dir: str = "./experiment_logs",
    tools: Optional[list] = None,
) -> Agent:
    """
    Create the experiment execution agent.
    """

    instructions = f"""You are an autonomous AI researcher executing experiments.

## WORKSPACE
| Path | Description |
|------|-------------|
| `{working_dir}/project` | Project root |
| `{working_dir}/project/logs` | Experiment logs (stdout/stderr) |
| `{working_dir}/project/results` | Experiment results (metrics, checkpoints) |

---

## WORKFLOW: EXPLORE → EXECUTE → REPORT

### 1️⃣ EXPLORE
1. `list_files` → understand project structure
2. Find entry scripts: `main.py`, `train.py`, `run_*.py`
3. `read_file` → understand arguments, configs, data paths
4. Verify datasets exist

### 2️⃣ EXECUTE
For EACH experiment:
1. Create directories: `mkdir -p {working_dir}/project/logs {working_dir}/project/results`
2. Construct command with proper arguments
3. Execute: `python script.py [args] > {working_dir}/project/logs/exp_name.log 2>&1`
4. **IMMEDIATELY verify**: Read log file, check for errors/metrics
5. Record result before next experiment

**Verification Protocol:**
- Read log file (last 100 lines)
- SUCCESS: "Training finished", final metrics present
- FAILURE: `Error`, `Traceback`, `OOM`, no final metrics

### 3️⃣ REPORT
Parse logs, extract metrics, build comparison tables.

---

## RULES
- **Baseline first**: Run baseline before proposed method
- **Same conditions**: Identical seeds, data splits for fair comparison
- **NEVER fabricate metrics** - Only report actual numbers from logs
- **NEVER claim success without verification**
- **Document ALL failures** - Don't skip failed experiments
- **NO .md files** - Do NOT create any markdown files.

---

## ⚠️ CRITICAL: REQUIRED OUTPUT FORMAT

🚨🚨🚨 **YOUR FINAL OUTPUT MUST BE ONLY A JSON OBJECT** 🚨🚨🚨

🚨 **CRITICAL**: 
- **DO NOT** use `write_file` to save your final result JSON!
- **DO NOT** call any tool to output the result!
- **JUST PRINT** the JSON directly in your response message!

**DO NOT** write markdown summaries like "I have completed the experiments..." or "Here are my findings...".
**DO NOT** write any explanatory text after completing tool calls.
**ONLY** output a valid JSON wrapped in ```json ... ``` code block.

{EXPERIMENT_EXECUTE_JSON_OUTPUT_INSTRUCTION}
"""

    agent = Agent(
        name="Experiment Executor",
        instructions=instructions,
        tools=tools or [],
        model=model,
    )

    return agent


class ExperimentExecuteAgent:
    """
    Main experiment execution agent.
    Wrapper around the autonomous researcher agent.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        working_dir: str = None,
        tools: Optional[list] = None,
        log_dir: str = "./experiment_logs",
        timeout: int = 3600,
        max_iterations: Optional[int] = None,  # None = unlimited turns
        verbose: bool = False,
    ):
        self.model = model
        self.working_dir = working_dir
        self.log_dir = log_dir
        self.timeout = timeout
        self.max_iterations = max_iterations  # None means no limit
        self.verbose = verbose
        # Always create hooks to show tool arguments
        # verbose mode controls whether to show detailed responses and results
        self.hooks = create_verbose_hooks(
            show_llm_responses=verbose,
            show_tools=verbose,
            show_tool_args=True,  # Always show tool arguments
        )

        # Auto-load recommended tools if not provided
        if tools is None:
            from src.agents.experiment_agent.sub_agents.experiment_execute import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize the autonomous researcher
        self.researcher_agent = create_execute_agent(
            model=model, working_dir=working_dir, log_dir=log_dir, tools=self.tools
        )

        # Expose agent
        self.agent = self.researcher_agent

    async def process(self, context: Any, **kwargs) -> ExperimentExecuteOutput:
        """
        Process the execution step.
        Let the agent autonomously explore and execute the experiment.
        """
        print_section("EXPERIMENT EXECUTION WORKFLOW", "=")

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        print_info(f"Log directory: {self.log_dir}")

        # 提取研究目标摘要
        print_subsection("Preparing Execution Context")
        summary = ""
        if hasattr(context, "pre_analysis_output") and context.pre_analysis_output:
            # Unified access
            if hasattr(context.pre_analysis_output, "get"):
                 summary = context.pre_analysis_output.get("summary", "")
            else:
                 summary = getattr(context.pre_analysis_output, "summary", "")
        print_info(f"Research goal loaded: {len(summary)} chars")

        # 提取代码计划信息（如果有）
        code_plan_info = ""
        experiment_plan_info = ""
        if hasattr(context, "code_plan_output") and context.code_plan_output:
            plan = context.code_plan_output
            # Unified access helper
            def get_val(obj, key, default=None):
                if hasattr(obj, "get"):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            # Extract experiment plan
            exp_plan = get_val(plan, "experiment_plan")
            if exp_plan:
                experiment_plan_info = self._format_experiment_plan(exp_plan)
                
        if code_plan_info:
            print_info("Code plan context extracted")
        if experiment_plan_info:
            print_info(
                "Experiment plan extracted - will execute ALL planned experiments"
            )

        print_info(f"Project Root: {self.working_dir}/project")
        print_info(
            f"Max Turns: {'Unlimited' if self.max_iterations is None else self.max_iterations}"
        )

        # Build task prompt
        task_prompt = f"""## Task
Execute experiments to validate the research implementation.

## Research Goal
{summary if summary else "Validate the implemented code through systematic experiments."}

## Environment
- Project: `{self.working_dir}/project`
- Logs: `{self.log_dir}`
{f'''
## Context
{code_plan_info}''' if code_plan_info else ""}
{f'''
## Experiment Plan
{experiment_plan_info}''' if experiment_plan_info else ""}

## Instructions
1. Explore the project structure to understand how to run experiments
2. Execute the experiments (baseline + proposed method, all datasets, multiple configs)
3. Create comparison tables and identify the best configuration
4. Report all results with log file paths

Begin by exploring the project with `list_files`.
"""

        print_subsection("Running Autonomous Researcher")

        effective_max_turns = (
            self.max_iterations if self.max_iterations is not None else 1000
        )

        run_config = RunConfig(
            model_settings=ModelSettings(max_tokens=128*1024)
        )

        result_stream = Runner.run_streamed(
            self.researcher_agent,
            task_prompt,
            hooks=self.hooks,
            max_turns=effective_max_turns,
            run_config=run_config
        )

        final_text = ""

        async for event in result_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if hasattr(event.data, "delta"):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                        final_text += delta.content
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
                        final_text += delta.text

        if hasattr(result_stream, "final_output") and isinstance(
            result_stream.final_output, str
        ):
            final_text = result_stream.final_output
        elif (
            not final_text
            and hasattr(result_stream, "chat_history")
            and result_stream.chat_history
        ):
            final_text = result_stream.chat_history[-1].content

        # Build full log from chat history if available
        if hasattr(result_stream, "chat_history") and result_stream.chat_history:
            full_log = "\n".join(
                [f"{msg.role}: {msg.content}" for msg in result_stream.chat_history]
            )
        else:
            full_log = final_text

        print_success("Research session completed")

        print_subsection("Parsing JSON Output")
        
        # Extract and parse JSON from the execution output
        # Use raise_on_failure=True to trigger retry in master agent
        try:
            final_output = extract_and_parse_json(final_text, ExperimentExecuteOutput, raise_on_failure=True)
        except JSONParseError as e:
            # Re-raise JSONParseError to trigger retry in master agent
            print_error(f"JSON parsing failed, will trigger retry: {e}")
            raise

        # 填充日志预览
        if final_output.log_path and os.path.exists(final_output.log_path):
            log_content = self.get_log_content(final_output.log_path)
            lines = log_content.splitlines()
            if lines:
                final_output.stdout_preview = "\n".join(lines[:50])
                if len(lines) > 50:
                    final_output.stdout_preview += "\n..."
                if final_output.has_error:
                    final_output.stderr_preview = "\n".join(lines[-50:])
            print_info(f"Log preview loaded from: {final_output.log_path}")

        print_success("Execution output parsed")

        # Display execution status
        if hasattr(final_output, "execution_status"):
            status = final_output.execution_status
            if status == "success":
                print(
                    f"\n{Colors.OKGREEN}{Colors.BOLD}✓ EXECUTION SUCCESS{Colors.ENDC}\n"
                )
            elif status == "error":
                print(f"\n{Colors.FAIL}{Colors.BOLD}✗ EXECUTION FAILED{Colors.ENDC}\n")
            else:
                print(
                    f"\n{Colors.WARNING}{Colors.BOLD}⚠ STATUS: {status.upper()}{Colors.ENDC}\n"
                )

        print_section("EXPERIMENT EXECUTION COMPLETE", "=")

        return final_output

    def _format_experiment_plan(self, exp_plan) -> str:
        """Format experiment plan into readable text for the agent."""
        lines = []

        # Baseline
        baseline = exp_plan.get("baseline_method", "")
        if baseline:
            lines.append(f"**Baseline**: {baseline}")

        # Datasets
        datasets = exp_plan.get("datasets", [])
        if datasets:
            lines.append(
                f"**Datasets** ({len(datasets)}): {', '.join(str(d) for d in datasets)}"
            )

        # Hyperparameters
        hp_space = exp_plan.get("hyperparameter_space", "")
        if hp_space:
            lines.append(f"**Hyperparameters**: {hp_space}")

        # Experiment matrix
        exp_matrix = exp_plan.get("experiment_matrix", [])
        if exp_matrix:
            lines.append(f"\n**Experiment Matrix** ({len(exp_matrix)} experiments):")
            lines.append("| Exp ID | Method | Dataset | Config | Seeds |")
            lines.append("|--------|--------|---------|--------|-------|")
            for exp in exp_matrix:
                exp_id = exp.get("exp_id", "?")
                method = exp.get("method", "?")
                dataset = exp.get("dataset", "?")
                hp = exp.get("hyperparameters", "default")
                seeds = exp.get("seeds", [42])
                seeds_str = (
                    ",".join(map(str, seeds)) if isinstance(seeds, list) else str(seeds)
                )
                lines.append(
                    f"| {exp_id} | {method} | {dataset} | {hp} | {seeds_str} |"
                )

        # Metrics
        primary = exp_plan.get("primary_metrics", [])
        if primary:
            lines.append(f"\n**Primary Metrics**: {', '.join(primary)}")

        return "\n".join(lines)

    def get_log_content(self, log_path: str) -> str:
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading log file: {str(e)}"


def create_experiment_execute_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: Optional[list] = None,
    log_dir: str = "./experiment_logs",
    timeout: int = 3600,
    max_iterations: Optional[int] = None,  # None = unlimited turns
    verbose: bool = False,
) -> ExperimentExecuteAgent:
    """
    Factory function to create an experiment execute agent.

    Args:
        max_iterations: Maximum number of turns. None means unlimited.
    """
    return ExperimentExecuteAgent(
        model=model,
        working_dir=working_dir,
        tools=tools,
        log_dir=log_dir,
        timeout=timeout,
        max_iterations=max_iterations,
        verbose=verbose,
    )
