"""
Shared utilities for SuperAgent.

Modules:
- tools: Core tools, validation utilities, and parsing helpers
- utils: Configuration, caching, DAG scheduling, LLM helpers
- schemas: Protocol definitions for inter-layer communication
- logger: Logging hooks for agent execution monitoring
- exceptions: Unified exception handling

Usage:
    from shared.tools import bash, file_viewer, run_linter
    from shared.utils import Cache, DAGScheduler, ProjectContext
    from shared.schemas import OptimizationTicket
    from shared.exceptions import SuperAgentError
"""

from src.agents.experiment_agent.shared.exceptions import (
    SuperAgentError,
    AgentError,
    TaskError,
    ValidationError,
    log_exception,
)

__all__ = [
    "SuperAgentError",
    "AgentError",
    "TaskError",
    "ValidationError",
    "log_exception",
]
