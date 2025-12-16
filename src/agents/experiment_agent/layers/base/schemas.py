"""
Base Schemas - Common schema definitions for all layers.

Provides:
- Result types for agent operations
- Common status enums
- Base configuration types

These are inherited by both Code Layer and Science Layer schemas.
"""

from typing import Optional, Dict, Any, List
from enum import Enum

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """
    Base result type for agent operations.

    All agent execution results should inherit from this.
    """

    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(default="", description="Human-readable status message")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Result data")


class TaskResult(BaseModel):
    """
    Result of a single task execution.

    Used by Worker agents to report task completion status.
    """

    task_id: str = Field(description="Unique identifier for the task")
    success: bool = Field(description="Whether the task succeeded")
    output: Optional[str] = Field(default=None, description="Task output content")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    attempts: int = Field(default=1, description="Number of attempts made")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )


class BatchResult(BaseModel):
    """
    Result of a batch of tasks.

    Used by Manager agents to report overall execution status.
    """

    total: int = Field(description="Total number of tasks")
    completed: int = Field(description="Number of completed tasks")
    failed: int = Field(description="Number of failed tasks")
    skipped: int = Field(default=0, description="Number of skipped tasks")
    results: Dict[str, TaskResult] = Field(
        default_factory=dict, description="Per-task results"
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total == 0:
            return 0.0
        return self.completed / self.total

    @property
    def all_succeeded(self) -> bool:
        """Check if all tasks succeeded."""
        return self.completed == self.total and self.failed == 0


class ExecutionStatus(str, Enum):
    """
    Status of an execution (task, phase, or overall).

    Used for tracking progress across all layers.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class Priority(str, Enum):
    """
    Priority levels for tasks.

    Used by both Code Layer and Science Layer for task ordering.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FunctionSignature(BaseModel):
    """
    Function signature specification.

    Used in blueprints and code generation.
    """

    name: str = Field(description="Name of the function")
    args: str = Field(default="", description="Arguments string, e.g. 'x: int, y: int'")
    return_type: str = Field(default="", description="Return type annotation")
    docstring: str = Field(default="", description="Docstring describing the function")


class ClassSignature(BaseModel):
    """
    Class signature specification.

    Used in blueprints and code generation.
    """

    name: str = Field(description="Name of the class")
    methods: List[FunctionSignature] = Field(
        default_factory=list, description="List of methods"
    )
    docstring: str = Field(default="", description="Docstring describing the class")
    attributes: Optional[Dict[str, str]] = Field(
        default=None, description="Class attributes"
    )


class FileSpecBase(BaseModel):
    """
    Base specification for a file.

    Extended by Code Layer and Science Layer with layer-specific fields.
    """

    file_path: str = Field(description="Relative path to the file")
    description: str = Field(
        default="", description="Description of the file's purpose"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Files this depends on"
    )


class ValidationResult(BaseModel):
    """
    Result of a validation operation.

    Used by validation tools and integrator agents.
    """

    valid: bool = Field(description="Whether validation passed")
    errors: List[str] = Field(
        default_factory=list, description="List of error messages"
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of warning messages"
    )

    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0


class ToolResult(BaseModel):
    """
    Standard result type for tool operations.

    All tools should return this format.
    """

    success: bool = Field(description="Whether the tool operation succeeded")
    data: Optional[Any] = Field(default=None, description="Result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    @classmethod
    def ok(cls, data: Any = None) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> "ToolResult":
        """Create a failed result."""
        return cls(success=False, error=error)


__all__ = [
    "AgentResult",
    "TaskResult",
    "BatchResult",
    "ExecutionStatus",
    "Priority",
    "FunctionSignature",
    "ClassSignature",
    "FileSpecBase",
    "ValidationResult",
    "ToolResult",
]
