"""
Analysis Synthesizer - Synthesizes concept and algorithm analysis results.

This agent merges the outputs from concept and algorithm analyzers into a 
structured JSON output matching PreAnalysisOutput schema.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    PreAnalysisOutput,
)
from src.agents.experiment_agent.utils.json_utils import generate_json_schema_instruction


# Generate JSON output instruction
PRE_ANALYSIS_JSON_INSTRUCTION = generate_json_schema_instruction(PreAnalysisOutput)


def create_analysis_synthesizer(model: str = "gpt-4o") -> Agent:
    """
    Create a synthesizer agent that merges analysis results and outputs JSON.

    Args:
        model: The model to use for the agent

    Returns:
        Agent instance configured for analysis synthesis with JSON output
    """
    instructions = f"""You are an expert research synthesizer that outputs structured JSON.

YOUR TASK:
Synthesize Concept Analysis, Algorithm Analysis, and Code Repository Analysis into a unified JSON output matching the PreAnalysisOutput schema.

INPUT FORMAT:
You will receive:
1. INPUT_TYPE: "paper" or "idea"
2. CONCEPT ANALYSIS: System architecture, conceptual framework, key innovations
3. ALGORITHM ANALYSIS: Algorithms, mathematical formulations, technical details
4. CODE REPOSITORIES ANALYSIS: Repository information and relevance

OUTPUT REQUIREMENTS:
You MUST output a valid JSON object with these exact fields:

{{
    "input_type": "<paper or idea>",
    "system_architecture": "<extracted from concept analysis>",
    "conceptual_framework": "<extracted from concept analysis>",
    "key_innovations": "<extracted from concept analysis>",
    "algorithms": "<extracted from algorithm analysis>",
    "mathematical_formulations": "<extracted from algorithm analysis>",
    "technical_specifications": "<extracted from algorithm analysis>",
    "summary": "<your executive summary synthesizing all analyses>",
    "implementation_guidance": "<strategic implementation advice incorporating code repo insights>",
    "code_repos_info": "<extracted from code repositories analysis>"
}}

CRITICAL RULES:
1. Output ONLY the JSON object - no markdown, no explanation
2. Preserve all technical details, LaTeX formulas, and code snippets from the input
3. Do NOT truncate or summarize the extracted content - preserve full detail
4. The summary and implementation_guidance fields should be YOUR synthesis
5. All other fields should be EXTRACTED from the corresponding input sections

{PRE_ANALYSIS_JSON_INSTRUCTION}
"""
    return Agent(
        name="Analysis Synthesizer",
        instructions=instructions,
        model=model,
    )


# Default agent instance
analysis_synthesizer = create_analysis_synthesizer()
