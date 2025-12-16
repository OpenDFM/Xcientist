"""
Schemas for Code Layer.

Provides data structures for:
- Proposal/Idea: Research proposal input
- Blueprint: System architecture specification
- CodeManifest: Code Handover Protocol (CHP) interface
"""

from src.agents.experiment_agent.layers.code.schemas.proposal import Proposal, Idea
from src.agents.experiment_agent.layers.code.schemas.blueprint import (
    Blueprint,
    FileSpec,
    ClassSignature,
    FunctionSignature,
)
from src.agents.experiment_agent.layers.code.schemas.idea_parser import (
    load_idea_file,
    parse_idea_markdown,
    validate_idea_markdown,
)
from src.agents.experiment_agent.layers.code.schemas.manifest import (
    CodeManifest,
    ConfigurationSpec,
    HyperparameterSpec,
    MetricsSpec,
)

__all__ = [
    # Proposal
    "Proposal",
    "Idea",
    # Blueprint
    "Blueprint",
    "FileSpec",
    "ClassSignature",
    "FunctionSignature",
    # Idea Parser
    "load_idea_file",
    "parse_idea_markdown",
    "validate_idea_markdown",
    # Manifest (CHP Interface)
    "CodeManifest",
    "ConfigurationSpec",
    "HyperparameterSpec",
    "MetricsSpec",
]
