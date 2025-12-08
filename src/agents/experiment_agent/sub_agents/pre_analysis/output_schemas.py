"""
Output schemas for pre-analysis agents.

Defines structured output formats for concept analysis, algorithm analysis,
and the unified output format.
"""

from pydantic import BaseModel, Field

from src.agents.experiment_agent.sub_agents.base.output_schemas import BaseDictModel


class ConceptAnalysis(BaseDictModel):
    """Concept analysis output structure."""

    system_architecture: str = Field(
        description="High-level system architecture and design patterns"
    )
    conceptual_framework: str = Field(
        description="Conceptual framework, theoretical foundations, design philosophy and principles"
    )
    key_innovations: str = Field(
        description="Key innovative concepts and contributions"
    )
    theoretical_basis: str = Field(
        description="Theoretical basis and mathematical foundations"
    )


class AlgorithmAnalysis(BaseDictModel):
    """Algorithm analysis output structure."""

    algorithms: str = Field(
        description="Core algorithms, procedures, and computational methods"
    )
    mathematical_formulations: str = Field(
        description="Mathematical models, formulas, and equations"
    )
    technical_details: str = Field(
        description="Implementation-level technical specifications"
    )
    algorithm_flow: str = Field(description="Algorithm flow and execution pipeline")


class AnalysisSynthesis(BaseDictModel):
    """Synthesis of analysis results."""

    summary: str = Field(description="Executive summary of the analysis")
    implementation_guidance: str = Field(
        description="Guidance for implementation and code planning"
    )


class CodeRepoInfo(BaseDictModel):
    """Information about a single code repository."""
    
    repo_name: str = Field(description="Name of the repository")
    repo_path: str = Field(description="Path to the repository")
    purpose: str = Field(description="Brief description of what this repository does")
    key_components: str = Field(description="Key components and their descriptions")
    relevance: str = Field(description="How this repo relates to the research idea/paper")
    useful_patterns: str = Field(description="Useful code patterns that could be reused")


class PreAnalysisOutput(BaseDictModel):
    """
    Unified analysis output for both paper and idea inputs.

    This structure provides a consistent interface for downstream
    code planning agents regardless of input type.
    """

    input_type: str = Field(description="Type of input: 'paper' or 'idea'")

    # Concept analysis
    system_architecture: str = Field(
        description="System architecture and high-level design"
    )
    conceptual_framework: str = Field(
        description="Conceptual framework, theoretical foundations, design philosophy and principles"
    )
    key_innovations: str = Field(description="Key innovations and contributions")

    # Algorithm analysis
    algorithms: str = Field(
        description="Core algorithms, procedures, and computational methods"
    )
    mathematical_formulations: str = Field(
        description="Mathematical models and formulas"
    )
    technical_specifications: str = Field(
        description="Technical specifications and implementation details"
    )

    # Additional metadata
    summary: str = Field(description="Executive summary of the analysis")
    implementation_guidance: str = Field(
        description="Guidance for implementation and code planning"
    )
    
    # Code repository analysis
    code_repos_info: str = Field(
        default="",
        description="Analysis of reference code repositories: their purpose, structure, and relevance to the research"
    )
    

