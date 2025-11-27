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


class CodePlanOutput(BaseModel):
    """
    Unified code plan output in YAML-compatible format.

    This structure is used by all code planning agents and
    will be formatted into YAML for downstream consumption.
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

    # Implementation plans
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
