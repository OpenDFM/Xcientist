"""
Base layer - Common base classes for Code and Science layers.

Provides:
- BaseAgent: Abstract base class for all agents
- BaseManager: Base class for Manager agents with DAG scheduling
- Shared schemas: Common data types

All layer-specific agents inherit from these base classes.
"""

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.layers.base.manager import BaseManager, TaskWrapper
from src.agents.experiment_agent.layers.base.schemas import (
    AgentResult,
    TaskResult,
    BatchResult,
    ExecutionStatus,
    Priority,
    FunctionSignature,
    ClassSignature,
    FileSpecBase,
    ValidationResult,
    ToolResult,
)

__all__ = [
    # Agent base classes
    "BaseAgent",
    "PromptBuilder",
    # Manager base classes
    "BaseManager",
    "TaskWrapper",
    # Schemas
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
