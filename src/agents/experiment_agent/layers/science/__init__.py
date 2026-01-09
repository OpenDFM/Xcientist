"""
Science Layer - Experimentation Layer

Agents:
- ExpArchitectAgent: Experiment design
- ExpWorkerAgent: Experiment execution (single worker, sequential)
- ExpIntegratorAgent: Result analysis + report/feedback
"""

from src.agents.experiment_agent.layers.science.architect import ExpArchitectAgent
from src.agents.experiment_agent.layers.science.worker import ExpWorkerAgent
from src.agents.experiment_agent.layers.science.integrator import ExpIntegratorAgent

__all__ = [
    "ExpArchitectAgent",
    "ExpWorkerAgent",
    "ExpIntegratorAgent",
]

