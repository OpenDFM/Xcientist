from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class FixIssueSpec(BaseModel):
    """A single issue record used to justify a fix task."""

    severity: str = Field(default="error", description="Issue severity: error|warning")
    issue_type: str = Field(description="Issue type, e.g. TestFailure, SyntaxError")
    message: str = Field(description="Human-readable error message")
    suggestion: str = Field(default="", description="Suggested fix direction")


class FixTaskSpec(BaseModel):
    """A file-level fix task (aligned with implementation tasks)."""

    task_id: str = Field(description="Unique task id (use file_path by default)")
    file_path: str = Field(description="Project-relative file path to edit")
    title: str = Field(description="Short title of the fix task")
    description: str = Field(description="What needs to be fixed and why")
    issues: List[FixIssueSpec] = Field(
        default_factory=list, description="Issues motivating this task"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Other task_ids this task depends on"
    )


class FixBlueprint(BaseModel):
    """
    Fix blueprint: a DAG of file-level fix tasks generated from integration verification.

    This is NOT for creating files; it is for fixing an already generated codebase.
    """

    entry_point: str = Field(description="Entry point file (for context)")
    trigger: str = Field(
        description="What triggered the fix blueprint, e.g. integration_verify"
    )
    tasks: List[FixTaskSpec] = Field(description="Fix tasks to execute")
    dependency_graph: Dict[str, List[str]] = Field(
        default_factory=dict, description="task_id -> dependencies"
    )
    notes: Optional[str] = Field(
        default=None, description="Optional notes for the manager/workers"
    )
