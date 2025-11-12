"""
Output schemas for code judge agent.

Defines structured output formats for code consistency evaluation
and implementation feedback.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class CodeIssue(BaseModel):
    """Individual code issue identified during review."""

    file_path: str = Field(description="Path to the file with the issue")
    issue_type: str = Field(
        description="Type of issue: 'logic_error', 'missing_implementation', 'inconsistency', 'quality'"
    )
    severity: str = Field(description="Severity level: 'critical', 'major', 'minor'")
    description: str = Field(description="Detailed description of the issue")
    expected: str = Field(description="Expected implementation based on plan/analysis")
    actual: str = Field(description="Actual implementation found in code")
    suggestion: str = Field(description="Suggestion for fixing the issue")
    line_numbers: Optional[str] = Field(
        default=None, description="Line numbers where the issue occurs (if applicable)"
    )


class UnitTestSpec(BaseModel):
    """Specification for a unit test to verify implementation."""

    test_file_path: str = Field(
        description="Path where the unit test file should be created (e.g., 'tests/test_module.py')"
    )
    test_code: str = Field(
        description="Complete Python unit test code (using pytest or unittest)"
    )
    test_description: str = Field(description="Description of what this test validates")
    target_files: List[str] = Field(
        description="List of implementation files that this test validates"
    )
    time_limit_seconds: int = Field(
        default=30,
        description="Maximum allowed execution time for this test in seconds",
    )
    data_subset_size: Optional[int] = Field(
        default=None,
        description="If testing with datasets, max number of samples to use (e.g., 10, 100)",
    )


class CodeJudgeOutput(BaseModel):
    """
    Output structure for code consistency evaluation.

    This structure contains the evaluation result, detailed feedback,
    and unit tests for code implementation review.
    """

    is_consistent: bool = Field(
        description="Whether the code implementation is consistent with plan and analysis"
    )

    overall_assessment: str = Field(
        description="High-level assessment of the code implementation quality and consistency"
    )

    # Consistency evaluation
    plan_consistency_score: float = Field(
        description="Score (0-1) indicating consistency with the code plan"
    )
    analysis_consistency_score: float = Field(
        description="Score (0-1) indicating consistency with the pre-analysis"
    )

    # Detailed feedback
    issues: List[CodeIssue] = Field(
        default_factory=list,
        description="List of issues found in the implementation",
    )

    strengths: List[str] = Field(
        default_factory=list,
        description="Aspects of the implementation that are well done",
    )

    missing_components: List[str] = Field(
        default_factory=list,
        description="Components specified in plan but missing in implementation",
    )

    extra_components: List[str] = Field(
        default_factory=list,
        description="Components implemented but not specified in plan",
    )

    # Unit tests for verification
    unit_tests: List[UnitTestSpec] = Field(
        default_factory=list,
        description="Unit tests generated to verify the current step implementation",
    )

    # Recommendations
    priority_fixes: List[str] = Field(
        default_factory=list,
        description="High-priority fixes that should be addressed first",
    )

    implementation_suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improving the implementation",
    )

    next_steps: str = Field(
        description="Recommended next steps based on the evaluation"
    )
