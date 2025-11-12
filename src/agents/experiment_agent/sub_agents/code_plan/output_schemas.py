"""
Output schemas for code planning agents.

Defines structured output formats for code plans and intermediate outputs.
"""

from pydantic import BaseModel, Field
from typing import List, Dict


class FileStructureItem(BaseModel):
    """Single file or directory in the structure."""

    path: str = Field(description="Relative path of the file/directory")
    type: str = Field(description="'file' or 'directory'")
    description: str = Field(description="Purpose of this file/directory")


class ImplementationStep(BaseModel):
    """Single step in the implementation roadmap."""

    step_number: int = Field(description="Step order")
    title: str = Field(description="Step title")
    description: str = Field(description="Detailed description of what to implement")
    files_involved: List[str] = Field(description="Files to create or modify")
    dependencies: List[int] = Field(
        description="Step numbers this step depends on", default_factory=list
    )


class ChecklistItem(BaseModel):
    """Single item in the implementation checklist."""

    step_id: int = Field(description="Unique identifier for this step")
    title: str = Field(description="Brief title of the implementation step")
    description: str = Field(
        description="Clear description of what needs to be implemented in this step"
    )
    files_to_create: List[str] = Field(
        description="Files that should be created in this step", default_factory=list
    )
    files_to_modify: List[str] = Field(
        description="Existing files that should be modified in this step",
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
        description="Complete file and directory structure"
    )

    project_structure_tree: str = Field(
        description="Visual tree representation of the complete project structure (ASCII art style)"
    )

    # Implementation plans
    dataset_plan: str = Field(description="Dataset preparation and loading plan")
    model_plan: str = Field(description="Model implementation plan")
    training_plan: str = Field(description="Training pipeline plan")
    testing_plan: str = Field(description="Testing and evaluation plan")

    # Roadmap
    implementation_roadmap: List[ImplementationStep] = Field(
        description="Step-by-step implementation roadmap"
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


class IntermediatePlanOutput(BaseModel):
    """Intermediate output from scenario-specific agents before formatting."""

    research_summary: str
    key_innovations: str
    file_structure_description: str
    project_structure_tree: str = Field(
        description="Visual ASCII tree of complete project structure"
    )
    dataset_plan: str
    model_plan: str
    training_plan: str
    testing_plan: str
    implementation_steps: str
    implementation_checklist: str = Field(
        description="Detailed checklist in text format for iterative implementation"
    )
    implementation_notes: str
    potential_challenges: str
    addressed_issues: str = ""
