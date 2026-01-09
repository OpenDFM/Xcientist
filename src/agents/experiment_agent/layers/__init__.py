"""
SuperAgent Layers

IMPORTANT: This module must NOT eagerly import subpackages (code/science), otherwise
we create circular imports:

- shared.tools.core -> shared.tools.validation -> layers.base.schemas -> layers(__init__)
- layers(__init__) importing code/science -> code.* importing shared.tools.core

To keep imports safe, we expose a lazy attribute loader via __getattr__.
"""

from typing import TYPE_CHECKING, Any

from src.agents.experiment_agent.layers.base import BaseAgent, BaseManager

if TYPE_CHECKING:
    from src.agents.experiment_agent.layers.code.architect import CodeArchitectAgent
    from src.agents.experiment_agent.layers.code.manager import CodeManagerAgent
    from src.agents.experiment_agent.layers.code.worker import CodeWorkerAgent
    from src.agents.experiment_agent.layers.code.integrator import CodeIntegratorAgent
    from src.agents.experiment_agent.layers.science.architect import ExpArchitectAgent
    from src.agents.experiment_agent.layers.science.worker import ExpWorkerAgent
    from src.agents.experiment_agent.layers.science.integrator import ExpIntegratorAgent


__all__ = [
    "BaseAgent",
    "BaseManager",
    "CodeArchitectAgent",
    "CodeManagerAgent",
    "CodeWorkerAgent",
    "CodeIntegratorAgent",
    "ExpArchitectAgent",
    "ExpWorkerAgent",
    "ExpIntegratorAgent",
]


def __getattr__(name: str) -> Any:
    if name == "CodeArchitectAgent":
        from src.agents.experiment_agent.layers.code.architect import CodeArchitectAgent

        return CodeArchitectAgent
    if name == "CodeManagerAgent":
        from src.agents.experiment_agent.layers.code.manager import CodeManagerAgent

        return CodeManagerAgent
    if name == "CodeWorkerAgent":
        from src.agents.experiment_agent.layers.code.worker import CodeWorkerAgent

        return CodeWorkerAgent
    if name == "CodeIntegratorAgent":
        from src.agents.experiment_agent.layers.code.integrator import (
            CodeIntegratorAgent,
        )

        return CodeIntegratorAgent
    if name == "ExpArchitectAgent":
        from src.agents.experiment_agent.layers.science.architect import (
            ExpArchitectAgent,
        )

        return ExpArchitectAgent
    if name == "ExpWorkerAgent":
        from src.agents.experiment_agent.layers.science.worker import ExpWorkerAgent

        return ExpWorkerAgent
    if name == "ExpIntegratorAgent":
        from src.agents.experiment_agent.layers.science.integrator import (
            ExpIntegratorAgent,
        )

        return ExpIntegratorAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
