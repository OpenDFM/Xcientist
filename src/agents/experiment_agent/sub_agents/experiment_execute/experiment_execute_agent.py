"""
Experiment Execute Agent - Executes experiment code and captures logs.

This agent executes the implemented code in the runtime environment,
captures execution logs, and reports whether any errors occurred.

Architecture:
- Execute Agent: Runs code and monitors execution
- Uses tools to execute code in controlled environment
- Captures stdout/stderr to log files
- Reports execution status and error information
"""

import os
import time
from datetime import datetime
from typing import Optional

from agents import Agent, Runner

from src.agents.experiment_agent.sub_agents.experiment_execute.output_schemas import (
    ExperimentExecuteOutput,
)


def create_execute_agent(model: str = "gpt-4o", tools: Optional[list] = None) -> Agent:
    """
    Create the experiment execution agent.

    Args:
        model: Model to use for the agent
        tools: List of tools for code execution (to be implemented)

    Returns:
        Agent configured for experiment execution
    """

    instructions = """You are an experiment execution agent responsible for running 
implemented code and capturing execution results.

YOUR RESPONSIBILITIES:

CRITICAL - EXECUTION DIRECTORY:
The code MUST be executed from the working_dir/project directory.
All Python scripts expect to run with project/ as the working directory.
- Change directory to project/ before execution
- Ensure PYTHONPATH includes the project/ directory
- All relative imports in the code assume execution from project/

Example execution:
  cd /path/to/working_dir/project
  python train.py --args

NOT:
  cd /path/to/working_dir
  python project/train.py --args

1. CODE EXECUTION
   - Use provided tools to execute the experiment code
   - Set up proper execution environment (virtualenv, dependencies)
   - Handle different execution scenarios:
     * Training scripts
     * Testing/evaluation scripts
     * Data preprocessing scripts
     * Complete pipelines

2. LOG MANAGEMENT
   - Create timestamped log file for each execution
   - Capture stdout and stderr separately
   - Write comprehensive execution logs including:
     * Timestamp and execution command
     * Environment information
     * Complete stdout output
     * Error messages and stack traces
     * Execution time and resource usage
   - Store logs in organized directory structure

3. ERROR DETECTION
   - Monitor execution process for errors
   - Detect different error types:
     * Syntax errors (failed import, parse errors)
     * Runtime errors (exceptions during execution)
     * Import errors (missing dependencies)
     * Timeout (execution exceeds time limit)
     * Resource errors (out of memory, GPU errors)
   - Capture error messages and stack traces
   - Set has_error flag appropriately

4. EXECUTION MONITORING
   - Track execution time
   - Monitor process exit codes
   - Detect execution interruptions
   - Extract performance metrics if available:
     * Training/validation accuracy
     * Loss values
     * Evaluation metrics (F1, precision, recall, etc.)
     * Training time per epoch

5. RESULT SUMMARIZATION
   - Provide clear execution summary
   - Highlight key results and metrics
   - Explain any errors that occurred
   - Suggest next steps if execution failed

TOOL USAGE GUIDELINES:

All tools return a dictionary with the following structure:
- success (bool): Indicates if the operation succeeded
- If successful: Contains relevant data fields (stdout, stderr, exit_code, status, etc.)
- If failed: Contains an "error" field with error message

Example successful execution:
{{
  "success": true,
  "exit_code": 0,
  "stdout": "Training completed...",
  "stderr": "",
  "execution_time": 125.5
}}

Example failed execution:
{{
  "success": false,
  "exit_code": 1,
  "stdout": "Starting training...",
  "stderr": "ImportError: No module named 'torch'",
  "error": "Execution failed with exit code 1"
}}

Always check the "success" field and "exit_code" before determining execution status.
A tool may return success=true but exit_code!=0, indicating the process ran but failed.

AVAILABLE TOOLS:
- run_python_script: Execute Python script (returns dict with "success", "exit_code", "stdout", "stderr")
- run_shell_command: Execute shell command (returns dict with "success", "exit_code", "stdout", "stderr")
- run_python_code: Execute Python code snippet (returns dict with "success", "exit_code", "stdout", "stderr")
- create_log_file: Create timestamped log file (returns dict with "success", "log_path", "filename")
- append_to_log: Append content to log (returns dict with "success", "log_path", "bytes_written")
- get_environment_info: Get Python environment info (returns dict with "success", "python_version", "platform")
- check_python_syntax: Check Python syntax (returns dict with "success", "valid_syntax", "syntax_error")

EXECUTION PROCESS:

Step 1: Prepare execution environment
   - Create log directory if not exists
   - Generate unique log file name with timestamp
   - Verify code path and dependencies

Step 2: Execute code
   - Run code using provided tools
   - Capture stdout/stderr in real-time
   - Monitor for errors and timeouts

Step 3: Write logs
   - Write all output to log file
   - Include execution metadata (time, command, environment)
   - Preserve formatting and stack traces

Step 4: Analyze results
   - Check exit code and error status
   - Extract metrics from output
   - Classify error type if applicable
   - Generate preview of output

Step 5: Generate output
   - Return structured ExperimentExecuteOutput
   - Include log path and error status
   - Provide execution summary

OUTPUT FORMAT:

You must output a structured ExperimentExecuteOutput with:
- log_path: Path to the complete log file
- has_error: Boolean indicating if errors occurred
- execution_status: 'success', 'error', 'timeout', or 'interrupted'
- exit_code: Process exit code (0 for success)
- error_message: Detailed error message if applicable
- error_type: Classification of error
- execution_time: Total execution time in seconds
- stdout_preview: Preview of stdout (first/last lines)
- stderr_preview: Preview of stderr if errors occurred
- experiment_metrics: Extracted metrics dictionary
- execution_summary: Human-readable summary

Be thorough in capturing all execution information and accurate in error detection."""

    agent = Agent(
        name="Experiment Execute Agent",
        instructions=instructions,
        tools=tools or [],
        output_type=ExperimentExecuteOutput,
        model=model,
    )

    return agent


class ExperimentExecuteAgent:
    """
    Main experiment execution agent that runs code and captures logs.

    This agent:
    1. Receives code path and execution parameters
    2. Executes code in controlled environment
    3. Captures all output to log files
    4. Detects and reports errors
    5. Extracts metrics if available
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        tools: Optional[list] = None,
        log_dir: str = "./experiment_logs",
        timeout: int = 3600,
    ):
        """
        Initialize the experiment execute agent.

        Args:
            model: Model to use for execution monitoring
            tools: Optional list of tools for code execution.
                   If None, automatically loads recommended tools.
            log_dir: Directory to store execution logs
            timeout: Maximum execution time in seconds (default: 1 hour)
        """
        self.model = model
        self.log_dir = log_dir
        self.timeout = timeout

        # Auto-load recommended tools if not provided
        if tools is None:
            from src.agents.experiment_agent.sub_agents.experiment_execute import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Create log directory if not exists
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize execute agent
        self.execute_agent = create_execute_agent(model=model, tools=self.tools)

        # Expose execute agent as main agent for handoff compatibility
        self.agent = self.execute_agent

    async def execute(
        self,
        code_path: str,
        entry_script: str,
        execution_args: Optional[dict] = None,
    ) -> ExperimentExecuteOutput:
        """
        Execute experiment code and capture results.

        Args:
            code_path: Path to the implemented codebase
            entry_script: Main script to execute (e.g., "train.py")
            execution_args: Optional execution arguments (e.g., {"epochs": 10})

        Returns:
            ExperimentExecuteOutput with execution results and log path
        """
        # Generate log file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"execution_{timestamp}.log"
        log_path = os.path.join(self.log_dir, log_filename)

        # Prepare execution arguments
        args_str = ""
        if execution_args:
            args_str = " ".join(
                [f"--{key} {value}" for key, value in execution_args.items()]
            )

        # Prepare input for execute agent
        execute_input = f"""
EXECUTE EXPERIMENT CODE

Code Path: {code_path}
Entry Script: {entry_script}
Execution Arguments: {args_str if args_str else "None"}
Log File: {log_path}
Timeout: {self.timeout} seconds

=== EXECUTION INSTRUCTIONS ===

1. Navigate to the code directory: {code_path}

2. Execute the entry script:
   Command: python {entry_script} {args_str}

3. Capture all output (stdout and stderr) and write to: {log_path}

4. Monitor execution for:
   - Exit code (0 = success, non-zero = error)
   - Error messages and stack traces
   - Execution time
   - Any performance metrics printed to stdout

5. Detect error types:
   - Import errors (ModuleNotFoundError, ImportError)
   - Syntax errors (SyntaxError)
   - Runtime errors (Exception stack traces)
   - Timeout (execution exceeds {self.timeout}s)

6. Extract metrics from output if available:
   - Look for patterns like "accuracy: 0.95", "loss: 0.23"
   - Training progress (epoch, steps)
   - Evaluation results

7. Generate execution summary including:
   - Whether execution succeeded or failed
   - Key results and metrics
   - Error description if applicable
   - Execution time

Use the available tools to execute the code and capture all information.
"""

        # Run execute agent
        # Use streamed version for real-time output
        result_stream = Runner.run_streamed(
            self.execute_agent, execute_input, max_turns=100
        )
        async for event in result_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type and hasattr(
                    event.data, "delta"
                ):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
        result = result_stream  # The stream object is the result

        return result.final_output

    def execute_sync(
        self,
        code_path: str,
        entry_script: str,
        execution_args: Optional[dict] = None,
    ) -> ExperimentExecuteOutput:
        """
        Synchronous version of execute method.

        Args:
            code_path: Path to the implemented codebase
            entry_script: Main script to execute (e.g., "train.py")
            execution_args: Optional execution arguments

        Returns:
            ExperimentExecuteOutput with execution results and log path
        """
        import asyncio

        return asyncio.run(self.execute(code_path, entry_script, execution_args))

    def get_log_content(self, log_path: str) -> str:
        """
        Read and return the content of a log file.

        Args:
            log_path: Path to the log file

        Returns:
            Log file content as string
        """
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading log file: {str(e)}"


def create_experiment_execute_agent(
    model: str = "gpt-4o",
    tools: Optional[list] = None,
    log_dir: str = "./experiment_logs",
    timeout: int = 3600,
) -> ExperimentExecuteAgent:
    """
    Factory function to create an experiment execute agent.

    Args:
        model: Model to use for execution monitoring
        tools: List of tools for code execution
        log_dir: Directory to store execution logs
        timeout: Maximum execution time in seconds

    Returns:
        ExperimentExecuteAgent instance
    """
    return ExperimentExecuteAgent(
        model=model, tools=tools, log_dir=log_dir, timeout=timeout
    )


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create the experiment execute agent
        agent = create_experiment_execute_agent(
            model="gpt-4o", log_dir="./experiment_logs", timeout=3600
        )

        # Execute experiment
        result = await agent.execute(
            code_path="/path/to/implemented/code",
            entry_script="train.py",
            execution_args={"epochs": 10, "batch_size": 32},
        )

        print("Execution Result:")
        print(f"Status: {result.execution_status}")
        print(f"Has Error: {result.has_error}")
        print(f"Log Path: {result.log_path}")
        print(f"Execution Time: {result.execution_time}s")

        if result.has_error:
            print(f"Error Type: {result.error_type}")
            print(f"Error Message: {result.error_message}")

        if result.experiment_metrics:
            print(f"Metrics: {result.experiment_metrics}")

        print(f"\nSummary: {result.execution_summary}")

    asyncio.run(main())
