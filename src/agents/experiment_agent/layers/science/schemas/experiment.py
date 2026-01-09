from typing import List, Dict, Optional, Any, Union, Literal
from pydantic import BaseModel, Field


class MetricSpec(BaseModel):
    """
    Explicit metric extraction spec (no guessing).

    - kind="json_key": read a JSON file and extract one numeric key (optionally nested one level via subkey)
    - kind="jsonl_last": read a JSONL file, optionally filter rows by `where`, and extract the last matching row's key (optionally nested via subkey)
    - kind="regex": search a text/log file using an explicit regex with one capturing group
    """

    kind: Literal["json_key", "jsonl_last", "regex"] = Field(
        description="Metric extraction method"
    )
    file_path: str = Field(description="Relative path to the file (from project_root)")
    name: str = Field(description="Metric name to report")
    key: Optional[str] = Field(default=None, description="JSON key (for kind=json_key)")
    subkey: Optional[str] = Field(
        default=None, description="Nested JSON key (for kind=json_key)"
    )
    where: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Optional equality filters for JSONL rows (for kind=jsonl_last), e.g. {"split": "train"}',
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern with one capture group (for kind=regex)",
    )


class ExperimentTask(BaseModel):
    """A single experiment run."""

    task_id: str = Field(description="Unique ID for this task")
    command: str = Field(description="Shell command to execute")
    description: str = Field(description="Purpose of this experiment")
    result_dir: str = Field(
        default="",
        description=(
            "Directory (relative to project_root) where this task should write its outputs. "
            "May contain placeholders like {step}, {step4}, {task_id}, {plan_id}."
        ),
    )
    dependencies: List[str] = Field(
        default_factory=list, description="IDs of tasks that must finish first"
    )
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration overrides"
    )
    expected_output_files: List[str] = Field(
        default_factory=list, description="Files expected to be generated"
    )
    metric_specs: List[MetricSpec] = Field(
        default_factory=list,
        description="Explicit metric extraction specs (no guessing). Empty means no metrics extracted.",
    )


class ExperimentPlan(BaseModel):
    """The full plan of experiments."""

    tasks: List[ExperimentTask] = Field(description="List of tasks to execute")
    analysis_goal: str = Field(
        description="What specific questions these experiments should answer"
    )


class ExperimentResult(BaseModel):
    """Result of a single task."""

    task_id: str
    success: bool
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    result_dir: Optional[str] = None
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


class ScienceAnalysis(BaseModel):
    """Final analysis from the Integrator."""

    success: bool = Field(description="Whether the scientific goal was achieved")
    summary: str = Field(description="High-level summary of findings")
    key_findings: List[str] = Field(default_factory=list)
    optimization_tickets: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tickets for Code Layer optimization"
    )
    next_experiments: Optional[ExperimentPlan] = Field(
        default=None, description="Plan for next round of experiments (if needed)"
    )

    # --- Science iteration reporting (spec-coding friendly) ---
    verdict: Optional[str] = Field(
        default=None,
        description="Idea validation verdict: supported/refuted/inconclusive (optional)",
    )
    report_md: str = Field(
        default="",
        description="Markdown report text for this iteration (optional)",
    )
    feedback_md: str = Field(
        default="",
        description="Markdown feedback to Architect for next iteration (optional; only when more experiments needed)",
    )
