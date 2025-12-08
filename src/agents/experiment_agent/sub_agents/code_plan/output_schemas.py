"""
Output schemas for code planning agents.

Defines structured output formats for code plans and intermediate outputs.
"""

from pydantic import Field
from typing import List, Optional

from src.agents.experiment_agent.sub_agents.base.output_schemas import BaseDictModel


class FileStructureItem(BaseDictModel):
    """
    Single file or directory in the structure.

    NOTE: DO NOT include 'tests/' directory or test files (test_*.py).
    Testing is managed separately by Code Judge Agent.
    """

    path: str = Field(description="Relative path of the file/directory (EXCLUDE tests/)")
    description: str = Field(description="Purpose of this file/directory")


class ChecklistItem(BaseDictModel):
    """Single item in the implementation checklist."""

    step_id: int = Field(description="Unique identifier for this step")
    title: str = Field(description="Brief title of the implementation step")
    description: str = Field(description="What needs to be implemented in this step")
    files_to_create: Optional[List[str]] = Field(
        default=None,
        description="List of specific file paths to create (e.g., ['data/dataset.py'])",
    )
    files_to_modify: Optional[List[str]] = Field(
        default=None,
        description="List of specific file paths to modify",
    )
    acceptance_criteria: Optional[List[str]] = Field(
        default=None,
        description="Criteria to verify this step is correctly implemented",
    )


class ExperimentItem(BaseDictModel):
    """Single experiment in the experiment matrix."""

    exp_id: str = Field(description="Unique experiment identifier (e.g., 'E1', 'E2')")
    method: str = Field(description="Method name: 'baseline' or 'proposed' or specific method name")
    dataset: str = Field(description="Dataset name")
    hyperparameters: str = Field(description="Key hyperparameters (e.g., 'lr=0.01, batch_size=32')")
    seeds: Optional[List[int]] = Field(default=None, description="Random seeds for reproducibility")


class ExperimentPlan(BaseDictModel):
    """
    Experiment plan defining all experiments to run.

    This is SEPARATE from CodePlan and focuses on experimental validation.
    """

    baseline_method: str = Field(description="Name of the baseline method")
    datasets: Optional[List[str]] = Field(default=None, description="List of ALL datasets to use (paths)")
    hyperparameter_space: str = Field(description="Complete hyperparameter search space definition")
    experiment_matrix: Optional[List[ExperimentItem]] = Field(default=None, description="Complete list of ALL experiments to run")
    primary_metrics: Optional[List[str]] = Field(default=None, description="Primary evaluation metrics")


class CodePlanOutput(BaseDictModel):
    """
    Unified code plan output in YAML-compatible format.

    This structure contains BOTH the Code Plan and Experiment Plan.
    The Code Plan must support all experiments in the Experiment Plan.
    """

    # Metadata
    plan_type: str = Field(description="Type of plan: 'initial', 'error_feedback', 'analysis_feedback'")

    # File structure
    file_structure: Optional[List[FileStructureItem]] = Field(
        default=None,
        description="Complete file and directory structure (EXCLUDE tests/ directory)",
    )

    # Implementation plans (CODE PLAN)
    dataset_plan: str = Field(description="Dataset preparation and loading plan")
    model_plan: str = Field(description="Model implementation plan")
    training_plan: str = Field(description="Training pipeline plan")

    # Checklist for iterative implementation
    implementation_checklist: Optional[List[ChecklistItem]] = Field(
        default=None,
        description="Detailed checklist for step-by-step iterative implementation",
    )

    # Additional guidance
    implementation_notes: str = Field(description="Important notes and considerations for implementation")

    # EXPERIMENT PLAN
    experiment_plan: ExperimentPlan = Field(
        default=None,
        description="Complete experiment plan including baseline, datasets, hyperparameters, and experiment matrix",
    )
