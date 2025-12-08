"""
Output schemas for experiment analysis agent.

Defines structured output formats for experiment results analysis,
including improvement suggestions for ideas and code plans.
"""

from typing import List, Optional

from pydantic import Field

from src.agents.experiment_agent.sub_agents.base.output_schemas import BaseDictModel


class MetricAnalysis(BaseDictModel):
    """Analysis of a specific metric."""

    metric_name: str = Field(description="Name of the metric (e.g., 'accuracy', 'loss')")
    actual_value: Optional[float] = Field(default=None, description="Actual value achieved in experiment")
    analysis: str = Field(description="Brief analysis of this metric's performance and whether it meets expectations")


class ExperimentAnalysisOutput(BaseDictModel):
    """
    Simplified output structure for experiment analysis.

    Analysis always triggers iteration to code_plan_agent for the next round.
    """

    # === Core Assessment ===
    meets_requirements: bool = Field(
        description="Whether the experiment meets the requirements from pre-analysis and plan"
    )

    overall_analysis: str = Field(
        description="High-level analysis including: summary, strengths, unexpected findings"
    )

    # === Metrics Analysis ===
    metrics_analysis: Optional[List[MetricAnalysis]] = Field(
        default=None,
        description="Detailed analysis of individual metrics",
    )

    # === Plan Feedback (for code_plan_agent iteration) ===
    plan_improvements: str = Field(
        default="",
        description="Specific improvements for the code plan: what to change, add, or fix",
    )

    potential_issues: Optional[List[str]] = Field(
        default=None,
        description="Issues identified that need to be addressed in next iteration",
    )

    # === Next Steps ===
    next_steps: str = Field(
        description="Recommended next steps with prioritized actions"
    )

    # === Iteration Control (for workflow state machine) ===
    @property
    def needs_iteration(self) -> bool:
        """Analysis always triggers iteration."""
        return True

    @property
    def iteration_target(self) -> str:
        """Always iterate back to plan."""
        return "plan"

    @property
    def feedback(self) -> str:
        """Combined feedback formatted for code_plan_agent."""
        parts = []
        
        parts.append(f"### Overall Analysis\n{self.overall_analysis}")
        
        if self.plan_improvements:
            parts.append(f"### Plan Improvements\n{self.plan_improvements}")
        
        if self.potential_issues:
            parts.append(f"### Issues to Address\n" + "\n".join(f"- {issue}" for issue in self.potential_issues))
        
        if self.next_steps:
            parts.append(f"### Recommended Actions\n{self.next_steps}")
        
        return "\n\n".join(parts)
