"""
Output schemas for experiment execute agent.

Defines structured output formats for experiment execution results,
including log file paths, result files, and execution details.
"""

from typing import Optional, List

from pydantic import BaseModel, Field

from src.agents.experiment_agent.sub_agents.base.output_schemas import BaseDictModel


class ExperimentFile(BaseDictModel):
    """Information about a single experiment output file."""

    file_path: str = Field(description="Absolute path to the file")
    file_type: str = Field(
        description="Type of file: 'log', 'result', 'checkpoint', 'config', 'plot', 'other'"
    )
    description: str = Field(description="Brief description of what this file contains")
    run_command: str = Field(
        default="",
        description="The command used to generate this file (if applicable)",
    )
    run_config: str = Field(
        default="",
        description="Key configuration/hyperparameters for this run (e.g., 'lr=0.001, epochs=10, batch_size=32')",
    )


class ExperimentExecuteOutput(BaseDictModel):
    """
    Output structure for experiment execution.

    This structure contains the execution results including all output files
    with their paths, descriptions, and run configurations.
    """

    # Execution status
    execution_status: str = Field(
        description="Overall execution status: 'success', 'partial', 'error', 'timeout', 'interrupted', 'skipped'. Use 'partial' when some experiments succeeded but others failed or were incomplete."
    )
    has_error: bool = Field(description="Whether any error occurred during execution")
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if execution failed (None if successful)",
    )

    # Files and outputs - THE MAIN CONTENT
    output_files: List[ExperimentFile] = Field(
        default_factory=list,
        description="List of all output files with their paths, types, descriptions, and run configurations",
    )

    # Primary log path (for backward compatibility)
    log_path: str = Field(
        default="",
        description="Path to the primary/best log file",
    )

    # Metrics and results
    experiment_metrics: str = Field(
        default="",
        description='Best metrics from execution as JSON string (e.g., \'{"accuracy": 0.95, "loss": 0.23}\')',
    )

    # Summary
    execution_summary: str = Field(
        default="",
        description="Human-readable summary: what was run, what worked, key findings",
    )

    # Previews (optional, for quick inspection)
    stdout_preview: str = Field(
        default="", description="Preview of stdout from the best run"
    )
    stderr_preview: str = Field(
        default="", description="Preview of stderr if any errors occurred"
    )
