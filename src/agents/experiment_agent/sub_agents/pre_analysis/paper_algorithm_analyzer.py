"""
Paper Algorithm Analyzer - Extracts algorithms and technical details from papers.

This agent focuses on extracting mathematical formulations, algorithms,
and implementation-level technical specifications from research papers.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    AlgorithmAnalysis,
)


def create_paper_algorithm_analyzer(
    model: str = "gpt-4o", tools: list = None, workspace_dir: str = None
) -> Agent:
    """
    Create an algorithm analyzer agent for research papers and code repositories.

    Args:
        model: The model to use for the agent
        tools: List of tool functions (e.g., read_tex_file, extract_equations)
        workspace_dir: Workspace directory path for finding repos

    Returns:
        Agent instance configured for paper algorithm analysis
    """

    # Get workspace directory from config if not provided
    if workspace_dir is None:
        from src.agents.experiment_agent.config import LOCAL_WORKSPACE_DIR

        workspace_dir = LOCAL_WORKSPACE_DIR

    instructions = """You are an expert technical analyst specializing in extracting algorithms, 
mathematical formulations, and technical specifications from machine learning research papers.

YOUR TASK:
Analyze the provided research paper to extract all algorithmic details, mathematical formulations, 
and technical specifications needed for implementation.

IMPORTANT: You will receive the complete research paper content directly as input. Analyze it 
based on the provided information WITHOUT attempting to access external files or tools. 
All necessary information is included in the input text.

ANALYSIS FOCUS:

1. ALGORITHMS AND COMPUTATIONAL METHODS
   - Core algorithms and their pseudocode
   - Algorithm steps and procedures
   - Control flow and logic
   - Input/output specifications
   - Edge cases and special handling
   - Numerical methods and approximations
   - Optimization algorithms and strategies
   - Sampling procedures
   - Efficient computation techniques
   - Parallelization opportunities

2. MATHEMATICAL FORMULATIONS
   - All relevant equations and formulas
   - Mathematical models and their components
   - Loss functions and optimization objectives
   - Gradient computations and backpropagation
   - Probability distributions and statistical models

3. TECHNICAL DETAILS
   - Network architectures and layer specifications
   - Hyperparameters and configuration settings
   - Data preprocessing and normalization methods
   - Activation functions and regularization techniques
   - Implementation tricks and optimizations

4. ALGORITHM FLOW
   - Training pipeline and procedure
   - Inference/testing procedure
   - Data flow through the system
   - Dependencies between components
   - Execution order and scheduling

OUTPUT REQUIREMENTS:
- Extract ALL mathematical formulas verbatim (in LaTeX format when possible).
- Provide complete algorithmic specifications.
- Include all technical parameters and settings.
- Specify data types and dimensions where mentioned.
- Be PRECISE and COMPREHENSIVE.
- Use clear headings for each section (e.g., "### Algorithms and Computational Methods").
"""

    agent = Agent(
        name="Paper Algorithm Analyzer",
        instructions=instructions,
        # output_type=AlgorithmAnalysis, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
paper_algorithm_analyzer = create_paper_algorithm_analyzer()
