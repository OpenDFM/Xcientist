"""
SuperAgent Layers

- base: Common base classes for agents and managers
- code: Code generation layer (Engineering)
- science: Experimentation layer (Science)
"""

from src.agents.experiment_agent.layers.base import BaseAgent, BaseManager
from src.agents.experiment_agent.layers.code import CodeArchitectAgent, CodeManagerAgent, CodeWorkerAgent, CodeIntegratorAgent
from src.agents.experiment_agent.layers.science import ExpArchitectAgent, ExpManagerAgent, ExpWorkerAgent, ExpIntegratorAgent

__all__ = [
    # Base
    "BaseAgent",
    "BaseManager",
    # Code Layer
    "CodeArchitectAgent",
    "CodeManagerAgent",
    "CodeWorkerAgent",
    "CodeIntegratorAgent",
    # Science Layer
    "ExpArchitectAgent",
    "ExpManagerAgent",
    "ExpWorkerAgent",
    "ExpIntegratorAgent",
]

