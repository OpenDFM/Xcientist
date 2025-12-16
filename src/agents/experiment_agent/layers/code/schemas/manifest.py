"""
Code Manifest Schema

Defines the CodeManifest structure for the Code Handover Protocol (CHP).
This is the primary interface from Code Layer to Science Layer.

Direction: Code Integrator -> Exp Architect
Purpose: Tell the Science Layer "the tool is ready, here's the manual"
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class HyperparameterSpec(BaseModel):
    """Specification for a single hyperparameter."""

    type: str = Field(description="Data type: 'float', 'int', 'str', 'bool', 'list'")
    default: Any = Field(description="Default value")
    description: Optional[str] = Field(
        default=None, description="Human-readable description"
    )
    min_value: Optional[float] = Field(
        default=None, description="Minimum value (for numeric types)"
    )
    max_value: Optional[float] = Field(
        default=None, description="Maximum value (for numeric types)"
    )
    choices: Optional[List[Any]] = Field(
        default=None, description="Valid choices (for enum-like params)"
    )


class ConfigurationSpec(BaseModel):
    """Configuration specification for the generated code."""

    config_file: Optional[str] = Field(
        default=None, description="Path to main configuration file"
    )
    config_format: str = Field(
        default="yaml", description="Configuration file format: 'yaml', 'json', 'py'"
    )
    hyperparameters: Dict[str, HyperparameterSpec] = Field(
        default_factory=dict,
        description="Map of hyperparameter names to their specifications",
    )


class MetricsSpec(BaseModel):
    """Specification for metrics output."""

    log_file: Optional[str] = Field(
        default=None, description="Path to metrics log file"
    )
    log_format: str = Field(
        default="json", description="Log file format: 'json', 'csv', 'tensorboard'"
    )
    keys: List[str] = Field(
        default_factory=list, description="List of metric keys to extract"
    )
    primary_metric: Optional[str] = Field(
        default=None, description="Primary metric for optimization"
    )
    higher_is_better: bool = Field(
        default=True, description="Whether higher values are better for primary metric"
    )


class CodeManifest(BaseModel):
    """
    Manifest describing the generated codebase for the Science Layer.

    This is the primary interface for the Code Handover Protocol (CHP).
    The Science Layer uses this to understand how to use the generated code.
    """

    # Basic Information
    project_root: str = Field(description="Absolute path to the project root")
    entry_point: str = Field(description="Main entry point file (e.g., main.py)")
    description: str = Field(
        default="", description="Description of what this code does"
    )

    # Entry Points / Scripts
    entry_points: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of script names to commands (e.g., 'train': 'python train.py --config config.yaml')",
    )

    # Legacy alias for backward compatibility
    @property
    def scripts(self) -> Dict[str, str]:
        return self.entry_points

    # Configuration
    configuration: Optional[ConfigurationSpec] = Field(
        default=None,
        description="Configuration specification",
    )

    # Metrics
    metrics: Optional[MetricsSpec] = Field(
        default=None,
        description="Metrics output specification",
    )

    # Dependencies
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of Python package dependencies",
    )

    # File Structure
    source_files: List[str] = Field(
        default_factory=list,
        description="List of source files in the project",
    )

    # Legacy field for backward compatibility
    config_file: Optional[str] = Field(
        default=None,
        description="Path to configuration file (deprecated, use configuration.config_file)",
    )

    # Legacy field for backward compatibility
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Exposed hyperparameters (deprecated, use configuration.hyperparameters)",
    )

    def get_train_command(self, **overrides) -> str:
        """Get the training command with optional overrides."""
        base_cmd = self.entry_points.get("train", f"python {self.entry_point}")

        # Add overrides as command line arguments
        for key, value in overrides.items():
            base_cmd += f" --{key} {value}"

        return base_cmd

    def get_eval_command(self, **overrides) -> str:
        """Get the evaluation command with optional overrides."""
        base_cmd = self.entry_points.get("eval", self.entry_points.get("evaluate", ""))

        if not base_cmd:
            return ""

        for key, value in overrides.items():
            base_cmd += f" --{key} {value}"

        return base_cmd

    def get_hyperparameter_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get hyperparameter ranges for experiment design."""
        if not self.configuration or not self.configuration.hyperparameters:
            return {}

        ranges = {}
        for name, spec in self.configuration.hyperparameters.items():
            ranges[name] = {
                "type": spec.type,
                "default": spec.default,
                "min": spec.min_value,
                "max": spec.max_value,
                "choices": spec.choices,
            }
        return ranges

    @classmethod
    def from_blueprint(
        cls, blueprint, project_root: str, description: str = ""
    ) -> "CodeManifest":
        """
        Create a CodeManifest from a Blueprint.

        Args:
            blueprint: The Blueprint object from code generation
            project_root: Absolute path to the project root
            description: Optional description

        Returns:
            CodeManifest instance
        """
        return cls(
            project_root=project_root,
            entry_point=blueprint.entry_point,
            description=description,
            source_files=blueprint.file_tree,
        )
