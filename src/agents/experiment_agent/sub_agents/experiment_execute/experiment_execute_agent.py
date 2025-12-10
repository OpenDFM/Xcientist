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


# Unifier instruction for ExperimentExecuteOutput
EXPERIMENT_EXECUTE_UNIFIER_INSTRUCTION = """You are an Output Formatter. Convert the structured execution output into JSON.

## Input Format
The input follows this structure:
```
=== EXPERIMENT EXECUTION OUTPUT ===
EXECUTION_STATUS: success/partial/error
HAS_ERROR: true/false
ERROR_MESSAGE: ...
=== OUTPUT FILES ===
FILE #N:
FILE_PATH: ...
FILE_TYPE: ...
...
=== PRIMARY LOG ===
LOG_PATH: ...
=== EXPERIMENT METRICS ===
metric: value
...
=== EXECUTION SUMMARY ===
...
=== STDOUT PREVIEW ===
...
=== STDERR PREVIEW ===
...
```

## Required JSON Output Format

```json
{
  "execution_status": "success",
  "has_error": false,
  "error_message": null,
  "output_files": [
    {
      "file_path": "/path/to/logs/exp.log",
      "file_type": "log",
      "description": "Description",
      "run_command": "python train.py",
      "run_config": "lr=0.001"
    }
  ],
  "log_path": "/path/to/logs/exp.log",
  "experiment_metrics": "{\\"accuracy\\": 0.95}",
  "execution_summary": "Summary text",
  "stdout_preview": "Output preview",
  "stderr_preview": ""
}
```

### Rules:
1. Parse EXECUTION_STATUS, HAS_ERROR, ERROR_MESSAGE
2. Parse each FILE block -> `output_files` array
3. Parse LOG_PATH -> `log_path`
4. Parse EXPERIMENT METRICS -> `experiment_metrics` (as JSON string)
5. Parse summaries and previews

Output ONLY valid JSON wrapped in ```json ... ``` block.
"""


def create_execute_output_unifier(model: str = None) -> Agent:
    """Create unifier agent to format execution output."""
    if model is None:
        from src.agents.experiment_agent.config import UNIFIER_MODEL
        model = UNIFIER_MODEL
    return Agent(
        name="Experiment Execute Output Unifier",
        instructions=EXPERIMENT_EXECUTE_UNIFIER_INSTRUCTION,
        model=model,
    )


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
3. `file_viewer` → understand arguments, configs, data paths
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

**⛔ ABSOLUTELY PROHIBITED - DO NOT CREATE THESE FILES:**
- `STEP*.json`, `*_COMPLETION*.json`, `*_EVALUATION*.json`, `*_SUMMARY*.json`
- `*_RESULT*.json`, `*_ANALYSIS*.json`, `*_STATUS*.json`
- **ANY `.md` files**, **ANY summary/status/progress files**
The orchestrator handles step tracking externally.

---

## 4️⃣ OUTPUT (MANDATORY - AS CHAT RESPONSE, NOT FILE!)

**🚨 CRITICAL OUTPUT RULES:**
- **DO NOT use `write_file` to write the output below!**
- **DO NOT write any JSON/summary/progress files to disk!**
- The format below is your **FINAL CHAT RESPONSE** - just type it directly as text!
- A separate unifier agent will convert your text response to JSON.

After completing all experiments, **STOP calling tools** and output this TEXT directly in chat:

```
=== EXPERIMENT EXECUTION OUTPUT ===

EXECUTION_STATUS: success  # success, partial, error, timeout, interrupted, skipped
HAS_ERROR: false
ERROR_MESSAGE: null  # or error description if failed

=== OUTPUT FILES ===

FILE #1:
FILE_PATH: /workspace/project/logs/baseline_exp.log
FILE_TYPE: log  # log, result, checkpoint, config, plot, other
DESCRIPTION: Baseline experiment training log
RUN_COMMAND: python train.py --method baseline --dataset mnist
RUN_CONFIG: lr=0.001, epochs=10, batch_size=32

FILE #2:
FILE_PATH: /workspace/project/results/model_best.pt
FILE_TYPE: checkpoint
DESCRIPTION: Best model checkpoint
RUN_COMMAND: 
RUN_CONFIG: 

=== PRIMARY LOG ===
LOG_PATH: /workspace/project/logs/baseline_exp.log

=== EXPERIMENT METRICS ===
accuracy: 0.95
loss: 0.23
f1_score: 0.92

=== EXECUTION SUMMARY ===
Ran baseline and proposed method on MNIST dataset.
Baseline achieved 87% accuracy.
Proposed method achieved 95% accuracy, outperforming baseline by 8%.

=== STDOUT PREVIEW ===
Epoch 10/10: loss=0.23, acc=0.95
Training completed successfully.

=== STDERR PREVIEW ===
[empty if no errors]
```

**🚨 REMINDER**: 
- This output format is your **CHAT RESPONSE** - just type it out!
- **DO NOT call write_file() with this content!**
- Include ALL output files with their full paths!
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

        # Initialize output unifier agent
        self.output_unifier = create_execute_output_unifier(model=model)

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

        print_subsection("Unifying Output")

        # Use unifier agent to convert raw output to structured JSON
        unifier_prompt = f"""Convert the following experiment execution output to JSON:

=== RAW OUTPUT START ===
{final_text}
=== RAW OUTPUT END ===

Extract all execution results, file paths, and metrics. Output the structured JSON:"""

        unifier_result = await Runner.run(
            self.output_unifier,
            unifier_prompt,
            run_config=RunConfig(model_settings=ModelSettings(max_tokens=64*1024)),
        )

        unified_text = ""
        if hasattr(unifier_result, "final_output") and isinstance(unifier_result.final_output, str):
            unified_text = unifier_result.final_output
        elif hasattr(unifier_result, "chat_history") and unifier_result.chat_history:
            unified_text = unifier_result.chat_history[-1].content

        print_subsection("Parsing JSON Output")
        
        # Extract and parse JSON from the unified output
        try:
            final_output = extract_and_parse_json(unified_text, ExperimentExecuteOutput, raise_on_failure=True)
        except JSONParseError as e:
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
