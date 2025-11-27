"""
Output schemas for experiment execute agent.

Defines structured output formats for experiment execution results,
including log file paths and error status.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ExperimentExecuteOutput(BaseModel):
    """
    Output structure for experiment execution.

    This structure contains the execution results including log file path,
    error status, and execution summary.
    """

    log_path: str = Field(
        description="Path to the log file containing execution output"
    )

    has_error: bool = Field(description="Whether any error occurred during execution")

    execution_status: str = Field(
        description="Overall execution status: 'success', 'error', 'timeout', 'interrupted', 'skipped'"
    )

    exit_code: Optional[int] = Field(
        default=None,
        description="Exit code of the execution process (None if not applicable)",
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Error message if execution failed (None if successful)",
    )

    error_type: Optional[str] = Field(
        default=None,
        description="Type of error: 'runtime_error', 'import_error', 'syntax_error', 'timeout', etc.",
    )

    execution_time: float = Field(description="Total execution time in seconds")

    stdout_preview: str = Field(
        default="", description="Preview of stdout (first/last lines)"
    )

    stderr_preview: str = Field(
        default="", description="Preview of stderr if any errors occurred"
    )

    experiment_metrics: str = Field(
        default="",
        description='Extracted metrics from execution as JSON string if available (e.g., \'{"accuracy": 0.95, "loss": 0.23}\')',
    )

    execution_summary: str = Field(
        description="Human-readable summary of the execution result"
    )
