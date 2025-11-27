"""
Analysis Synthesizer - Synthesizes concept and algorithm analysis results.

This agent merges the outputs from concept and algorithm analyzers into a coherent
summary and implementation guidance, ensuring a unified view of the research analysis.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    AnalysisSynthesis,
)


def create_analysis_synthesizer(model: str = "gpt-4o") -> Agent:
    """
    Create a synthesizer agent to merge analysis results.

    Args:
        model: The model to use for the agent

    Returns:
        Agent instance configured for analysis synthesis
    """
    instructions = """You are an expert research synthesizer.

YOUR TASK:
Synthesize the results of Concept Analysis and Algorithm Analysis into a coherent summary and implementation guidance.

INPUT:
You will receive:
1. Concept Analysis (System Architecture, Conceptual Framework, Key Innovations)
2. Algorithm Analysis (Algorithms, Mathematical Formulations, Technical Details)
3. Input Type (Paper or Idea)

OUTPUT:
Generate a structured synthesis textual report containing:
1. Executive Summary: A high-level overview of the research, highlighting key innovations and core algorithms.
2. Implementation Guidance: Strategic advice for implementing the system, focusing on architectural principles and algorithmic complexity.

CRITICAL:
- Use the provided analysis results to generate these fields.
- Do NOT make up information not present in the analysis.
- Use clear headings for each section.
"""
    return Agent(
        name="Analysis Synthesizer",
        instructions=instructions,
        # output_type=AnalysisSynthesis, # Removed for duplex mode
        model=model,
    )


# Default agent instance
analysis_synthesizer = create_analysis_synthesizer()
