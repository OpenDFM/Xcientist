"""
Exceptions - Unified exception handling for SuperAgent.

Provides:
- Base exception classes
- Layer-specific exceptions
- Helper functions for exception handling and logging

All exceptions should inherit from SuperAgentError.
"""

import logging
import traceback
import sys
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class SuperAgentError(Exception):
    """
    Base exception for all SuperAgent errors.

    All custom exceptions should inherit from this.
    """

    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __str__(self):
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details,
        }


class AgentError(SuperAgentError):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        message: str,
        agent_type: str = "Agent",
        code: str = "AGENT_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, details)
        self.agent_type = agent_type

    def __str__(self):
        return f"[{self.code}] {self.agent_type}: {self.message}"


class ArchitectError(AgentError):
    """Exception raised by Architect agents."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "Architect", "ARCHITECT_ERROR", details)


class ManagerError(AgentError):
    """Exception raised by Manager agents."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "Manager", "MANAGER_ERROR", details)


class WorkerError(AgentError):
    """Exception raised by Worker agents."""

    def __init__(
        self,
        message: str,
        task_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if task_id:
            details["task_id"] = task_id
        super().__init__(message, "Worker", "WORKER_ERROR", details)
        self.task_id = task_id


class IntegratorError(AgentError):
    """Exception raised by Integrator agents."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "Integrator", "INTEGRATOR_ERROR", details)


class TaskError(SuperAgentError):
    """Base exception for task-related errors."""

    def __init__(
        self,
        message: str,
        task_id: str,
        code: str = "TASK_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["task_id"] = task_id
        super().__init__(message, code, details)
        self.task_id = task_id


class TaskFailedError(TaskError):
    """Exception raised when a task fails after all retries."""

    def __init__(
        self,
        message: str,
        task_id: str,
        attempts: int = 0,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["attempts"] = attempts
        super().__init__(message, task_id, "TASK_FAILED", details)
        self.attempts = attempts


class TaskTimeoutError(TaskError):
    """Exception raised when a task times out."""

    def __init__(
        self,
        message: str,
        task_id: str,
        timeout_seconds: float = 0,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["timeout_seconds"] = timeout_seconds
        super().__init__(message, task_id, "TASK_TIMEOUT", details)
        self.timeout_seconds = timeout_seconds


class DependencyError(TaskError):
    """Exception raised when task dependencies cannot be satisfied."""

    def __init__(
        self,
        message: str,
        task_id: str,
        missing_deps: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["missing_deps"] = missing_deps or []
        super().__init__(message, task_id, "DEPENDENCY_ERROR", details)
        self.missing_deps = missing_deps or []


class ValidationError(SuperAgentError):
    """Exception raised when validation fails."""

    def __init__(
        self,
        message: str,
        errors: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["validation_errors"] = errors or []
        super().__init__(message, "VALIDATION_ERROR", details)
        self.errors = errors or []


class SyntaxValidationError(ValidationError):
    """Exception raised when code has syntax errors."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["file_path"] = file_path
        details["line_number"] = line_number
        super().__init__(message, [], details)
        self.file_path = file_path
        self.line_number = line_number


class SpecValidationError(ValidationError):
    """Exception raised when code doesn't match specification."""

    def __init__(
        self,
        message: str,
        missing_items: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["missing_items"] = missing_items or []
        super().__init__(message, missing_items, details)
        self.missing_items = missing_items or []


class ConfigurationError(SuperAgentError):
    """Exception raised for configuration errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["config_key"] = config_key
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key


class APIKeyError(ConfigurationError):
    """Exception raised when API key is missing or invalid."""

    def __init__(
        self,
        message: str = "API key not configured",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, "api_key", details)


class FileOperationError(SuperAgentError):
    """Exception raised for file operation errors."""

    def __init__(
        self,
        message: str,
        file_path: str,
        operation: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["file_path"] = file_path
        details["operation"] = operation
        super().__init__(message, "FILE_ERROR", details)
        self.file_path = file_path
        self.operation = operation


class DAGError(SuperAgentError):
    """Exception raised for DAG-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "DAG_ERROR", details)


class CyclicDependencyError(DAGError):
    """Exception raised when a cyclic dependency is detected."""

    def __init__(
        self,
        message: str,
        cycle: Optional[list] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        details["cycle"] = cycle or []
        super().__init__(message, details)
        self.cycle = cycle or []


def log_exception(
    exc: Exception,
    context: str = "",
    level: int = logging.ERROR,
) -> None:
    """
    Log an exception with context.

    Args:
        exc: The exception to log
        context: Additional context string
        level: Logging level (default: ERROR)
    """
    if context:
        logger.log(level, f"{context}: {exc}")
    else:
        logger.log(level, str(exc))

    if level >= logging.ERROR:
        logger.debug(traceback.format_exc())


def format_exception_chain(exc: Exception) -> str:
    """
    Format an exception chain into a readable string.

    Args:
        exc: The exception to format

    Returns:
        Formatted exception string
    """
    lines = []
    current = exc

    while current is not None:
        if isinstance(current, SuperAgentError):
            lines.append(f"  [{current.code}] {current.message}")
        else:
            lines.append(f"  {type(current).__name__}: {current}")

        current = current.__cause__

    return "\n".join(lines)


def safe_execute(func, *args, default=None, log_error: bool = True, **kwargs):
    """
    Execute a function safely, catching and logging exceptions.

    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value to return on error
        log_error: Whether to log errors
        **kwargs: Keyword arguments

    Returns:
        Function result or default on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_error:
            log_exception(e, f"Error in {func.__name__}")
        return default


def is_rate_limit_error(error: object) -> bool:
    """Return True if error indicates API rate limiting / quota exhaustion."""
    text = str(error)
    lower = text.lower()
    return ("429" in text) or ("rate_limit" in lower) or ("usage limit exceeded" in lower)


def exit_on_rate_limit(error: object, exit_code: int = 1) -> None:
    """Exit the process if the error indicates API rate limiting / quota exhaustion."""
    if not is_rate_limit_error(error):
        return

    error_str = str(error)
    print("\n" + "=" * 60)
    print("❌ FATAL ERROR: API Rate Limit Exceeded (429)")
    print("=" * 60)
    print(f"Error: {error_str}")
    print("\nThe program will now exit.")
    print("Please wait before retrying or check your API quota.")
    print("=" * 60)
    sys.exit(int(exit_code))


__all__ = [
    "SuperAgentError",
    "AgentError",
    "ArchitectError",
    "ManagerError",
    "WorkerError",
    "IntegratorError",
    "TaskError",
    "TaskFailedError",
    "TaskTimeoutError",
    "DependencyError",
    "ValidationError",
    "SyntaxValidationError",
    "SpecValidationError",
    "ConfigurationError",
    "APIKeyError",
    "FileOperationError",
    "DAGError",
    "CyclicDependencyError",
    "log_exception",
    "format_exception_chain",
    "safe_execute",
    "is_rate_limit_error",
    "exit_on_rate_limit",
]
