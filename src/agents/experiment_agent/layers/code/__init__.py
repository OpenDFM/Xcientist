"""
Code Layer - Engineering/Code Generation Layer

Agents:
- ArchitectAgent: System architecture design
- ManagerAgent: Task orchestration with DAG scheduling
- WorkerAgent: Code implementation
- IntegratorAgent: Integration and verification
"""

from src.agents.experiment_agent.layers.code.architect import CodeArchitectAgent
from src.agents.experiment_agent.layers.code.manager import CodeManagerAgent
from src.agents.experiment_agent.layers.code.worker import CodeWorkerAgent
from src.agents.experiment_agent.layers.code.integrator import CodeIntegratorAgent

__all__ = [
    "CodeArchitectAgent",
    "CodeManagerAgent",
    "CodeWorkerAgent",
    "CodeIntegratorAgent",
]
