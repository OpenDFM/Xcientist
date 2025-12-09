"""
Output schemas for code implementation agents.

Defines structured output formats for code implementation results.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from src.agents.experiment_agent.sub_agents.base.output_schemas import BaseDictModel


class GeneratedFile(BaseDictModel):
    """Information about a generated file."""

    file_path: str = Field(description="Relative path of the generated file")
    content: str = Field(default="", description="Complete file content (may be empty for references)")
    description: str = Field(default="", description="Purpose and description of this file")
    dependencies: List[str] = Field(default_factory=list, description="List of files this file depends on")


class ImplementationSummary(BaseDictModel):
    """Summary of implementation work done."""

    files_created: int = Field(description="Number of files created")
    files_modified: int = Field(description="Number of files modified")
    total_lines: int = Field(description="Total lines of code generated")
    key_components: List[str] = Field(description="List of key components implemented")


class CodeImplementOutput(BaseDictModel):
    """
    Unified code implementation output.

    This structure is used by all code implementation agents.
    """

    # Metadata
    implementation_type: str = Field(description="Type: 'initial' or 'fix'")

    # Generated code
    generated_files: List[GeneratedFile] = Field(
        description="All generated or modified files"
    )

    # Implementation details
    implementation_summary: ImplementationSummary = Field(
        description="Summary of implementation work"
    )

    # Testing and validation
    test_files: Optional[List[GeneratedFile]] = Field(
        default=None, description="Test files generated"
    )

    # Fix-specific (optional)
    issues_addressed: str = Field(
        default="", description="Issues that were addressed (for fix implementations)"
    )
