"""
Paper Concept Analyzer - Analyzes conceptual framework of research papers.

This agent extracts high-level design concepts, theoretical foundations,
and architectural patterns from research papers in LaTeX format.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    ConceptAnalysis,
)


def create_paper_concept_analyzer(
    model: str = "gpt-4o", tools: list = None, workspace_dir: str = None
) -> Agent:
    """
    Create a concept analyzer agent for research papers.

    Args:
        model: The model to use for the agent
        tools: List of tool functions (e.g., read_tex_file, parse_latex_sections)
        workspace_dir: Workspace directory path for finding papers

    Returns:
        Agent instance configured for paper concept analysis
    """

    # Get workspace directory from config if not provided
    if workspace_dir is None:
        from src.agents.experiment_agent.config import LOCAL_WORKSPACE_DIR

        workspace_dir = LOCAL_WORKSPACE_DIR

    instructions = """You are an expert research analyst specializing in extracting and understanding 
the conceptual and theoretical foundations of machine learning research papers.

YOUR TASK:
Analyze the provided research paper to extract its high-level conceptual framework 
and architectural design principles.

IMPORTANT: You will receive the complete research paper content directly as input. Analyze it 
based on the provided information WITHOUT attempting to access external files or tools. 
All necessary information is included in the input text.

ANALYSIS FOCUS:

1. SYSTEM ARCHITECTURE
   - Overall system design and structure
   - Component organization and interaction patterns
   - Architectural innovations and design patterns
   - System-level abstractions and interfaces

2. CONCEPTUAL FRAMEWORK AND DESIGN PHILOSOPHY
   - Core concepts and theoretical constructs
   - Relationships between key ideas
   - Conceptual innovations and contributions
   - Framework for understanding the approach
   - Underlying design principles and rationale
   - Motivations behind key design choices
   - Trade-offs and design considerations
   - Philosophical approach to problem-solving

3. KEY INNOVATIONS
   - Novel conceptual contributions
   - Paradigm shifts or new perspectives
   - Unique combinations of existing concepts
   - Breakthrough ideas and insights

4. THEORETICAL BASIS
   - Mathematical and theoretical foundations
   - Theoretical frameworks and models (high-level)
   - Connections to established theory
   - Theoretical justifications for the approach

OUTPUT REQUIREMENTS:
- Provide a detailed textual analysis for each section above.
- Focus on HIGH-LEVEL concepts, not implementation details.
- Explain the "WHY" behind design choices.
- Connect concepts to enable coherent implementation.
- Be thorough but maintain clarity.
- Use clear headings for each section (e.g., "### System Architecture").
"""

    agent = Agent(
        name="Paper Concept Analyzer",
        instructions=instructions,
        # output_type=ConceptAnalysis, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
paper_concept_analyzer = create_paper_concept_analyzer()
