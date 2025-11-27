"""Experiment analysis agent module."""

from src.agents.experiment_agent.sub_agents.experiment_analysis.experiment_analysis_agent import (
    create_experiment_analysis_agent,
)
from src.agents.experiment_agent.sub_agents.experiment_analysis.output_schemas import (
    ExperimentAnalysisOutput,
    MetricAnalysis,
)


def get_recommended_tools():
    """
    Get recommended tools for experiment analysis agent.

    Returns:
        List of tools for log analysis
    """
    from src.agents.experiment_agent.tools import (
        FILE_TOOLS,
        DOCUMENT_TOOLS,
        CODE_ANALYSIS_TOOLS,
    )

    return FILE_TOOLS + DOCUMENT_TOOLS + CODE_ANALYSIS_TOOLS[:3]


__all__ = [
    "create_experiment_analysis_agent",
    "ExperimentAnalysisOutput",
    "MetricAnalysis",
    "get_recommended_tools",
]
