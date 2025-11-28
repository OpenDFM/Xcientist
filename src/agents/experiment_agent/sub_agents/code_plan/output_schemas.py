"""
Output schemas for code planning agents.

Defines structured output formats for code plans and intermediate outputs.
"""

from pydantic import BaseModel, Field
from typing import List, Dict


class FileStructureItem(BaseModel):
    """
    Single file or directory in the structure.

    NOTE: DO NOT include 'tests/' directory or test files (test_*.py).
    Testing is managed separately by Code Judge Agent.
    """

    path: str = Field(
        description="Relative path of the file/directory (EXCLUDE tests/)"
    )
    type: str = Field(description="'file' or 'directory'")
    description: str = Field(description="Purpose of this file/directory")


class ChecklistItem(BaseModel):
    """Single item in the implementation checklist."""

    step_id: int = Field(description="Unique identifier for this step")
    title: str = Field(description="Brief title of the implementation step")
    description: str = Field(
        description="Clear description of what needs to be implemented in this step"
    )
    files_to_create: List[str] = Field(
        description="MUST be a list of specific file paths (e.g., ['data/dataset.py', 'models/model.py']). DO NOT use descriptive text like 'ALL files' or 'Include config files'. Each item must be a concrete file path relative to project root.",
        default_factory=list,
    )
    files_to_modify: List[str] = Field(
        description="MUST be a list of specific file paths that already exist and need modification (e.g., ['config.yaml', 'main.py']). DO NOT use descriptive text.",
        default_factory=list,
    )
    acceptance_criteria: List[str] = Field(
        description="Criteria to verify this step is correctly implemented"
    )
    dependencies: List[int] = Field(
        description="Step IDs that must be completed before this step",
        default_factory=list,
    )
    estimated_complexity: str = Field(
        description="Complexity level: 'low', 'medium', 'high'"
    )


class ExperimentItem(BaseModel):
    """Single experiment in the experiment matrix."""

    exp_id: str = Field(description="Unique experiment identifier (e.g., 'E1', 'E2')")
    method: str = Field(
        description="Method name: 'baseline' or 'proposed' or specific method name"
    )
    dataset: str = Field(description="Dataset name")
    hyperparameters: str = Field(
        description="Key hyperparameters for this run (e.g., 'lr=0.01, batch_size=32')"
    )
    seeds: List[int] = Field(
        description="Random seeds for reproducibility", default_factory=lambda: [42]
    )
    priority: str = Field(
        description="Priority: 'high', 'medium', 'low'", default="medium"
    )


class ExperimentPlan(BaseModel):
    """
    Experiment plan defining all experiments to run.

    This is SEPARATE from CodePlan and focuses on experimental validation.
    """

    # Baseline definition
    baseline_method: str = Field(description="Name of the baseline method")
    baseline_justification: str = Field(description="Why this baseline is appropriate")
    baseline_implementation: str = Field(
        description="How baseline is implemented (file/function)"
    )

    # Dataset coverage
    datasets: List[str] = Field(description="List of ALL datasets to use (paths)")
    dataset_preprocessing: str = Field(
        description="Preprocessing steps for each dataset"
    )

    # Hyperparameter search
    hyperparameter_space: str = Field(
        description="Complete hyperparameter search space definition"
    )
    tuning_strategy: str = Field(
        description="Tuning strategy: 'grid', 'random', 'manual'"
    )

    # Experiment matrix
    experiment_matrix: List[ExperimentItem] = Field(
        description="Complete list of ALL experiments to run"
    )

    # Evaluation
    primary_metrics: List[str] = Field(description="Primary evaluation metrics")
    secondary_metrics: List[str] = Field(
        description="Secondary metrics", default_factory=list
    )
    success_criteria: str = Field(
        description="How to determine if proposed method succeeds"
    )

    # Runtime estimate
    estimated_runtime: str = Field(
        description="Estimated total runtime for all experiments", default=""
    )


class CodePlanOutput(BaseModel):
    """
    Unified code plan output in YAML-compatible format.

    This structure contains BOTH the Code Plan and Experiment Plan.
    The Code Plan must support all experiments in the Experiment Plan.
    """

    # Metadata
    plan_type: str = Field(
        description="Type of plan: 'initial', 'judge_feedback', 'error_feedback', 'analysis_feedback'"
    )
    timestamp: str = Field(description="Plan generation timestamp")

    # Research context
    research_summary: str = Field(description="Summary of research to implement")
    key_innovations: str = Field(description="Key innovations to implement")

    # File structure
    file_structure: List[FileStructureItem] = Field(
        description="Complete file and directory structure (EXCLUDE tests/ directory)"
    )

    # Implementation plans (CODE PLAN)
    dataset_plan: str = Field(description="Dataset preparation and loading plan")
    model_plan: str = Field(description="Model implementation plan")
    training_plan: str = Field(description="Training pipeline plan")
    testing_plan: str = Field(
        description="ML model evaluation plan (metrics, inference, validation) - NOT unit testing"
    )

    # Checklist for iterative implementation
    implementation_checklist: List[ChecklistItem] = Field(
        description="Detailed checklist for step-by-step iterative implementation"
    )

    # Additional guidance
    implementation_notes: str = Field(
        description="Important notes and considerations for implementation"
    )
    potential_challenges: str = Field(
        description="Potential challenges and mitigation strategies"
    )

    # EXPERIMENT PLAN (NEW - MANDATORY)
    experiment_plan: ExperimentPlan = Field(
        default=None,
        description="Complete experiment plan including baseline, datasets, hyperparameters, and experiment matrix",
    )

    # Feedback-specific (optional, depending on plan_type)
    addressed_issues: str = Field(
        default="",
        description="How this plan addresses feedback/errors from previous iteration",
    )

    # Performance targets
    performance_targets: str = Field(
        default="",
        description="Performance targets and metrics to achieve",
    )
