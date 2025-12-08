"""
Output schemas for code planning agents.

Defines structured output formats for code plans and intermediate outputs.
"""

from pydantic import Field
from typing import List

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
    files_to_create: List[str] = Field(
        description="List of specific file paths to create (e.g., ['data/dataset.py'])",
        default_factory=list,
    )
    files_to_modify: List[str] = Field(
        description="List of specific file paths to modify",
        default_factory=list,
    )
    acceptance_criteria: List[str] = Field(
        description="Criteria to verify this step is correctly implemented"
    )


class ExperimentItem(BaseDictModel):
    """Single experiment in the experiment matrix."""

    exp_id: str = Field(description="Unique experiment identifier (e.g., 'E1', 'E2')")
    method: str = Field(description="Method name: 'baseline' or 'proposed' or specific method name")
    dataset: str = Field(description="Dataset name")
    hyperparameters: str = Field(description="Key hyperparameters (e.g., 'lr=0.01, batch_size=32')")
    seeds: List[int] = Field(description="Random seeds for reproducibility", default_factory=lambda: [42])


class ExperimentPlan(BaseDictModel):
    """
    Experiment plan defining all experiments to run.

    This is SEPARATE from CodePlan and focuses on experimental validation.
    """

    baseline_method: str = Field(description="Name of the baseline method")
    datasets: List[str] = Field(description="List of ALL datasets to use (paths)")
    hyperparameter_space: str = Field(description="Complete hyperparameter search space definition")
    experiment_matrix: List[ExperimentItem] = Field(description="Complete list of ALL experiments to run")
    primary_metrics: List[str] = Field(description="Primary evaluation metrics")


class CodePlanOutput(BaseDictModel):
    """
    Unified code plan output in YAML-compatible format.

    This structure contains BOTH the Code Plan and Experiment Plan.
    The Code Plan must support all experiments in the Experiment Plan.
    """

    # Metadata
    plan_type: str = Field(description="Type of plan: 'initial', 'error_feedback', 'analysis_feedback'")

    # File structure
    file_structure: List[FileStructureItem] = Field(
        description="Complete file and directory structure (EXCLUDE tests/ directory)"
    )

    # Implementation plans (CODE PLAN)
    dataset_plan: str = Field(description="Dataset preparation and loading plan")
    model_plan: str = Field(description="Model implementation plan")
    training_plan: str = Field(description="Training pipeline plan")

    # Checklist for iterative implementation
    implementation_checklist: List[ChecklistItem] = Field(
        description="Detailed checklist for step-by-step iterative implementation"
    )

    # Additional guidance
    implementation_notes: str = Field(description="Important notes and considerations for implementation")

    # EXPERIMENT PLAN
    experiment_plan: ExperimentPlan = Field(
        default=None,
        description="Complete experiment plan including baseline, datasets, hyperparameters, and experiment matrix",
    )
