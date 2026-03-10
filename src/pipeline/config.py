"""Pipeline configuration."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    """Configuration for the Idea <-> Experiment pipeline."""

    # Idea Agent configuration
    idea_agent_root: Path = Field(
        default=Path("src/agents/idea_agent"),
        description="Root directory of Idea Agent",
    )
    symbolic_memory_path: Path = Field(
        default=Path("idea_skill_priors"),
        description="Path to symbolic memory storage",
    )

    # Experiment Agent configuration
    experiment_agent_root: Path = Field(
        default=Path("src/agents/experiment_agent"),
        description="Root directory of Experiment Agent",
    )
    experiment_workspace_root: Path = Field(
        default=Path("workspaces"),
        description="Root directory for experiment workspaces",
    )

    # Pipeline configuration
    default_max_iterations: int = Field(
        default=3,
        description="Maximum number of iterations for the pipeline loop",
    )
    ablation_result_filename: str = Field(
        default="ablation_results.json",
        description="Filename for ablation results",
    )

    # Component family mapping
    default_macro_roles: list[str] = Field(
        default_factory=lambda: [
            "generator",
            "encoder",
            "decoder",
            "loss",
            "regularizer",
            "optimizer",
            "architecture",
            "attention",
            "embedding",
            "representation",
            "constraint",
            "retrieval",
            "augmentation",
        ],
        description="List of known macro roles for component family inference",
    )


# Global default config instance
_default_config: Optional[PipelineConfig] = None


def get_default_config() -> PipelineConfig:
    """Get the default pipeline configuration."""
    global _default_config
    if _default_config is None:
        _default_config = PipelineConfig()
    return _default_config


def update_config(**kwargs) -> PipelineConfig:
    """Update the default configuration with custom values."""
    global _default_config
    current = get_default_config()
    for key, value in kwargs.items():
        if hasattr(current, key):
            setattr(current, key, value)
    return current
