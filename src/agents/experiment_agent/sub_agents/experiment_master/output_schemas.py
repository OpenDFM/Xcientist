"""
Output schemas for experiment master agent.

Defines structured output formats for the complete experiment workflow.
"""

from typing import List, Optional

from pydantic import BaseModel, Field

from src.agents.experiment_agent.sub_agents.base.output_schemas import BaseDictModel


class WorkflowStep(BaseDictModel):
    """Record of a single workflow step."""

    step_number: int = Field(description="Sequential step number")
    agent_name: str = Field(description="Name of the agent executed")
    status: str = Field(description="Status: 'success', 'failed', 'needs_revision'")
    summary: str = Field(description="Brief summary of this step's result")


class ExperimentMasterOutput(BaseDictModel):
    """
    Output structure for the complete experiment workflow.

    This structure contains the final results and complete workflow history.
    """

    workflow_completed: bool = Field(
        description="Whether the entire workflow completed successfully"
    )

    final_status: str = Field(
        description="Final status: 'success', 'failed', 'max_iterations_reached'"
    )

    total_iterations: int = Field(
        description="Total number of iterations through the workflow"
    )

    workflow_history: List[WorkflowStep] = Field(
        default_factory=list,
        description="Complete history of all workflow steps",
    )

    # Final outputs from each stage
    pre_analysis_output: Optional[dict] = Field(
        default=None, description="Final pre-analysis output"
    )

    code_plan_output: Optional[dict] = Field(
        default=None, description="Final code plan output"
    )

    code_implement_output: Optional[dict] = Field(
        default=None, description="Final code implementation output"
    )

    code_judge_output: Optional[dict] = Field(
        default=None, description="Final code judge output"
    )

    experiment_execute_output: Optional[dict] = Field(
        default=None, description="Final experiment execution output"
    )

    experiment_analysis_output: Optional[dict] = Field(
        default=None, description="Final experiment analysis output"
    )

    # Summary
    overall_summary: str = Field(
        description="High-level summary of the entire experiment workflow"
    )

    key_findings: List[str] = Field(
        default_factory=list,
        description="Key findings from the experiment",
    )

    final_recommendations: str = Field(
        default="",
        description="Final recommendations based on all iterations",
    )
