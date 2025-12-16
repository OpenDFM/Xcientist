"""
Science Layer - Experimentation Layer

Agents:
- ExpArchitectAgent: Experiment design
- ExpManagerAgent: Experiment execution with DAG scheduling
- ExpWorkerAgent: Experiment execution
- ExpIntegratorAgent: Result analysis
"""

from src.agents.experiment_agent.layers.science.architect import ExpArchitectAgent
from src.agents.experiment_agent.layers.science.manager import ExpManagerAgent
from src.agents.experiment_agent.layers.science.worker import ExpWorkerAgent
from src.agents.experiment_agent.layers.science.integrator import ExpIntegratorAgent

__all__ = [
    "ExpArchitectAgent",
    "ExpManagerAgent",
    "ExpWorkerAgent",
    "ExpIntegratorAgent",
]

