"""
Experiment Execute Agent - Autonomous Researcher.

This agent acts as an autonomous researcher who can execute code, monitor results,
and iteratively tune hyperparameters to achieve research goals.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Any, Dict, List

from agents import Agent, Runner

from src.agents.experiment_agent.sub_agents.experiment_execute.output_schemas import (
    ExperimentExecuteOutput,
)
from src.agents.experiment_agent.config import OUTPUT_UNIFIER_MODEL
from src.agents.experiment_agent.utils.print_utils import *


# --- AGENT FACTORIES ---


def create_execute_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    log_dir: str = "./experiment_logs",
    tools: Optional[list] = None,
) -> Agent:
    """
    Create the autonomous experiment researcher agent.
    """

    instructions = f"""You are an Expert AI Researcher.
Your goal is to successfully execute the provided Code Implementation to verify the Research Idea.

### YOUR CAPABILITIES
1. **Run Experiments**: You can execute the code using `run_shell_command`.
2. **Analyze Logs**: You can read the output logs using `read_file` to see loss, accuracy, and errors.
3. **Iterate**: You are expected to NOT just run once. You should:
   - Run with initial/default parameters.
   - Check the results (Is it converging? Is there an OOM error? Is performance satisfying?).
   - **Adjust parameters** (LR, Batch Size, Epochs, Model Config) and RUN AGAIN.
   - Repeat until you achieve a satisfactory result or determine the idea fails.

### ENVIRONMENT
- **Workspace**: `{working_dir}`
- **Project Root**: `{working_dir}/project`
- **Log Dir**: `{log_dir}` (You MUST save logs here).

### PROTOCOL
1. **Understand**: Read the `entry_script` content first to see what `argparse` arguments are available!
2. **Execute**: 
   - Construct a command like: `python train.py --epochs 10 --lr 0.01 > {log_dir}/run_1.log 2>&1`
   - **CRITICAL**: Always use redirection (`> log_file 2>&1`) to capture output.
   - Run it using `run_shell_command` from the Project Root.
3. **Observe**: Read the log file immediately after running.
4. **Reflect**: 
   - *Error?* Fix args (e.g., reduce batch size if OOM).
   - *Poor Performance?* Tune args (e.g., increase epochs, change LR).
   - *Success?* Great, you are done.
5. **Report**: When you are satisfied, provide a final summary.

**IMPORTANT**: You are autonomous. Do not ask the user for permission to run again. Just do it.
"""

    agent = Agent(
        name="Experiment Researcher",
        instructions=instructions,
        tools=tools or [],
        model=model,
    )

    return agent


def create_execute_unifier_agent(model: str = "gpt-4o") -> Agent:
    return Agent(
        name="Execute Output Unifier",
        instructions="""You are an expert data structuring assistant.
Your task is to convert the researcher's final report into a structured `ExperimentExecuteOutput` object.

Input text will contain the conversation history of the researcher.
Identify the **BEST/FINAL** run from the history.

Extract:
- Log Path (of the best run)
- Status (Success/Fail)
- Metrics (JSON string of the best metrics found)
- Summary (What was achieved, what parameters worked best)
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
        max_iterations: int = 10,  # Passed to Runner as max_turns
    ):
        self.model = model
        self.working_dir = working_dir
        self.log_dir = log_dir
        self.timeout = timeout
        self.max_iterations = max_iterations

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
        """

        # 1. Identify Entry Script
        entry_script = self._find_entry_script(context)
        if not entry_script:
            return self._create_skipped_output()

        # 2. Prepare Context
        summary = None
        if hasattr(context.pre_analysis_output, "summary"):
            summary = context.pre_analysis_output.summary

        print_section("STARTING AUTONOMOUS EXPERIMENT RESEARCH", "=")
        print_info(f"Entry Script: {entry_script}")
        print_info(f"Max Turns: {self.max_iterations}")

        # 3. Construct the initial prompt
        # We give the agent the context and tell it to GO.
        task_prompt = f"""
START RESEARCH TASK

**Target Script**: `{entry_script}`
**Research Goal**: {summary}
**Log Directory**: `{self.log_dir}`

**Instructions**:
1. Check `{entry_script}` content to understand arguments.
2. Construct and run commands like: `python {entry_script} --arg val > {self.log_dir}/run_1.log 2>&1`
3. Analyze logs and iterate until the goal is met.
"""

        # 4. Run the Agent (ReAct Loop)
        # The Runner will handle the loop: Agent -> Tool -> Agent -> Tool ...
        result_stream = Runner.run_streamed(
            self.researcher_agent, task_prompt, max_turns=self.max_iterations
        )

        final_text = ""
        chat_history = []

        async for event in result_stream.stream_events():
            if hasattr(event, "data") and hasattr(event.data, "delta"):
                delta = event.data.delta
                if hasattr(delta, "content") and delta.content:
                    print(delta.content, end="", flush=True)
                    final_text += delta.content

        # Capture the full history for the unifier to analyze
        if hasattr(result_stream, "chat_history"):
            chat_history = result_stream.chat_history
            # Append final text if not in history yet
            full_log = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
        else:
            full_log = final_text

        print_success("\nResearch Session Completed.")

        # 5. Unify Output
        # We send the entire conversation log to the unifier so it can pick the best run.
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

        # Hydrate log preview
        if final_output.log_path and os.path.exists(final_output.log_path):
            log_content = self.get_log_content(final_output.log_path)
            lines = log_content.splitlines()
            if lines:
                final_output.stdout_preview = "\n".join(lines[:50])
                if len(lines) > 50:
                    final_output.stdout_preview += "\n..."
                if final_output.has_error:
                    final_output.stderr_preview = "\n".join(lines[-50:])

        return final_output

    def _find_entry_script(self, context: Any) -> Optional[str]:
        if context.code_plan_output:
            plan_output = context.code_plan_output
            file_structure = []
            if isinstance(plan_output, dict):
                file_structure = plan_output.get("file_structure", [])
            elif hasattr(plan_output, "file_structure"):
                file_structure = plan_output.file_structure

            for item in file_structure:
                path = ""
                if hasattr(item, "path"):
                    path = item.path
                elif isinstance(item, dict):
                    path = item.get("path", "")
                else:
                    continue

                if "main.py" in path or "train.py" in path or "run.py" in path:
                    return path.split("/")[-1] if "/" in path else path
        return None

    def _create_skipped_output(self) -> ExperimentExecuteOutput:
        return ExperimentExecuteOutput(
            log_path="",
            has_error=False,
            execution_status="skipped",
            exit_code=0,
            execution_time=0.0,
            stdout_preview="N/A",
            stderr_preview="",
            experiment_metrics={},
            execution_summary="Execution skipped - no entry script found.",
        )

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
    max_iterations: int = 20,  # Give it enough turns to think and act
) -> ExperimentExecuteAgent:
    """
    Factory function to create an experiment execute agent.
    """
    return ExperimentExecuteAgent(
        model=model,
        working_dir=working_dir,
        tools=tools,
        log_dir=log_dir,
        timeout=timeout,
        max_iterations=max_iterations,
    )
