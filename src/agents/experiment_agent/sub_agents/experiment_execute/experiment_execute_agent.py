"""
Experiment Execute Agent - Autonomous Researcher.

This agent acts as an autonomous researcher who can execute code, monitor results,
and iteratively tune hyperparameters to achieve research goals.
"""

import os
from typing import Optional, Any

from agents import Agent, Runner

from src.agents.experiment_agent.sub_agents.experiment_execute.output_schemas import (
    ExperimentExecuteOutput,
)
from src.agents.experiment_agent.config import OUTPUT_UNIFIER_MODEL
from src.agents.experiment_agent.logger import create_verbose_hooks
from src.agents.experiment_agent.utils.print_utils import *


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

    instructions = f"""You are an autonomous AI researcher executing experiments to validate research ideas.

## Environment
- Project Root: `{working_dir}/project`
- Log Directory: `{log_dir}`

## Your Workflow

1. **Explore**: Understand the codebase structure and find entry scripts
2. **Execute**: Run experiments according to the provided plan (or design reasonable experiments if no plan)
3. **Analyze**: Read logs, extract metrics, compare results
4. **Report**: Summarize findings with evidence

## Core Principles

**Scientific Rigor**:
- Always include baseline comparison when evaluating a new method
- Test on all relevant datasets, not just one
- Try multiple hyperparameter configurations to find the best setting
- Record seeds for reproducibility

**Completeness**:
- Execute all experiments specified in the plan
- Don't stop after one successful run - complete the full experiment matrix
- Log every run to a file for later analysis

**Clear Reporting**:
- Create comparison tables (baseline vs proposed, per dataset)
- Identify the best configuration and explain why
- List all output files with their paths

## Output Format

End with a structured report containing:
1. Experiment results table (all runs with metrics)
2. Baseline vs proposed comparison (per dataset)
3. Best configuration identified
4. All output file paths
5. Conclusion: Does the proposed method work? By how much?

## Tips
- Redirect output to log files: `python script.py > {log_dir}/run_name.log 2>&1`
- Read log files after each run to check for errors
- If a run fails, try to fix it or note the failure and continue
"""

    agent = Agent(
        name="Experiment Executor",
        instructions=instructions,
        tools=tools or [],
        model=model,
    )

    return agent


def create_execute_unifier_agent(model: str = "gpt-4o") -> Agent:
    return Agent(
        name="Execute Output Unifier",
        instructions="""You are an expert data structuring assistant.
Your task is to convert the researcher's experiment session into a structured `ExperimentExecuteOutput` object.

### INPUT
The session log contains the researcher's exploration, execution attempts, and final report.

### YOUR TASK
Extract and structure the following information:

1. **execution_status**: 'success', 'error', 'timeout', or 'skipped'
2. **has_error**: True if any critical error occurred
3. **error_message**: Error details if failed (None if successful)

4. **output_files**: A LIST of all generated files. For EACH file, extract:
   - `file_path`: Full absolute path to the file
   - `file_type`: 'log', 'result', 'checkpoint', 'config', 'plot', or 'other'
   - `description`: What this file contains
   - `run_command`: The exact command used to generate this file
   - `run_config`: Key hyperparameters (e.g., "lr=0.001, epochs=100, batch_size=32")

5. **log_path**: Path to the PRIMARY/BEST log file

6. **experiment_metrics**: JSON string of best metrics, e.g., '{"loss": 0.23, "accuracy": 0.87}'

7. **execution_summary**: Concise summary of what was done and key findings

### EXAMPLE OUTPUT STRUCTURE
```json
{
  "execution_status": "success",
  "has_error": false,
  "output_files": [
    {
      "file_path": "/workspace/logs/run_1.log",
      "file_type": "log",
      "description": "Initial run with default parameters, converged but slow",
      "run_command": "python train.py --lr 0.01 --epochs 100",
      "run_config": "lr=0.01, epochs=100, batch_size=32, dataset=a9a"
    },
    {
      "file_path": "/workspace/logs/run_2.log",
      "file_type": "log",
      "description": "Best run with tuned learning rate",
      "run_command": "python train.py --lr 0.001 --epochs 200",
      "run_config": "lr=0.001, epochs=200, batch_size=32, dataset=a9a"
    }
  ],
  "log_path": "/workspace/logs/run_2.log",
  "experiment_metrics": "{\\"loss\\": 0.23, \\"accuracy\\": 0.87}",
  "execution_summary": "Successfully trained model on a9a dataset. Best results with lr=0.001."
}
```

### IMPORTANT
- Extract ALL files mentioned in the session (logs, results, checkpoints, etc.)
- Include the FULL ABSOLUTE PATH for each file
- Include the EXACT command and configuration for each run
- Identify the BEST run and set it as log_path
""",
        output_type=ExperimentExecuteOutput,
        model=OUTPUT_UNIFIER_MODEL,
    )


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

        # Initialize output unifier
        self.output_unifier = create_execute_unifier_agent(model=model)

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
            if hasattr(context.pre_analysis_output, "summary"):
                summary = context.pre_analysis_output.summary
            elif isinstance(context.pre_analysis_output, dict):
                summary = context.pre_analysis_output.get("summary", "")
        print_info(f"Research goal loaded: {len(summary)} chars")

        # 提取代码计划信息（如果有）
        code_plan_info = ""
        experiment_plan_info = ""
        if hasattr(context, "code_plan_output") and context.code_plan_output:
            plan = context.code_plan_output
            if isinstance(plan, dict):
                if plan.get("research_summary"):
                    code_plan_info += (
                        f"Research Summary: {plan.get('research_summary')}\n"
                    )
                if plan.get("testing_plan"):
                    code_plan_info += f"Testing Plan: {plan.get('testing_plan')}\n"
                # Extract experiment plan
                exp_plan = plan.get("experiment_plan")
                if exp_plan:
                    experiment_plan_info = self._format_experiment_plan(exp_plan)
            elif hasattr(plan, "research_summary"):
                code_plan_info += f"Research Summary: {plan.research_summary}\n"
                if hasattr(plan, "testing_plan"):
                    code_plan_info += f"Testing Plan: {plan.testing_plan}\n"
                # Extract experiment plan
                if hasattr(plan, "experiment_plan") and plan.experiment_plan:
                    experiment_plan_info = self._format_experiment_plan(
                        plan.experiment_plan
                    )
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

        result_stream = Runner.run_streamed(
            self.researcher_agent,
            task_prompt,
            hooks=self.hooks,
            max_turns=effective_max_turns,
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

        # 获取完整对话历史
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

        # 使用 Unifier 整理输出
        print_subsection("Unifying Output Format")
        unifier_input = f"""
Please analyze this research session log and extract the details of the BEST/FINAL run.

=== SESSION LOG ===
{full_log}
"""
        unifier_stream = Runner.run_streamed(
            self.output_unifier, unifier_input, hooks=None
        )

        async for _ in unifier_stream.stream_events():
            pass

        final_output = unifier_stream.final_output

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

        print_success("Execution output unified")

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

        def get_attr(obj, key, default=""):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # Baseline
        baseline = get_attr(exp_plan, "baseline_method", "")
        if baseline:
            lines.append(f"**Baseline**: {baseline}")
            impl = get_attr(exp_plan, "baseline_implementation", "")
            if impl:
                lines.append(f"  Implementation: {impl}")

        # Datasets
        datasets = get_attr(exp_plan, "datasets", [])
        if datasets:
            lines.append(
                f"**Datasets** ({len(datasets)}): {', '.join(str(d) for d in datasets)}"
            )

        # Hyperparameters
        hp_space = get_attr(exp_plan, "hyperparameter_space", "")
        if hp_space:
            lines.append(f"**Hyperparameters**: {hp_space}")

        # Experiment matrix
        exp_matrix = get_attr(exp_plan, "experiment_matrix", [])
        if exp_matrix:
            lines.append(f"\n**Experiment Matrix** ({len(exp_matrix)} experiments):")
            lines.append("| Exp ID | Method | Dataset | Config | Seeds |")
            lines.append("|--------|--------|---------|--------|-------|")
            for exp in exp_matrix:
                exp_id = get_attr(exp, "exp_id", "?")
                method = get_attr(exp, "method", "?")
                dataset = get_attr(exp, "dataset", "?")
                hp = get_attr(exp, "hyperparameters", "default")
                seeds = get_attr(exp, "seeds", [42])
                seeds_str = (
                    ",".join(map(str, seeds)) if isinstance(seeds, list) else str(seeds)
                )
                lines.append(
                    f"| {exp_id} | {method} | {dataset} | {hp} | {seeds_str} |"
                )

        # Metrics
        primary = get_attr(exp_plan, "primary_metrics", [])
        if primary:
            lines.append(f"\n**Primary Metrics**: {', '.join(primary)}")

        success = get_attr(exp_plan, "success_criteria", "")
        if success:
            lines.append(f"**Success Criteria**: {success}")

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
