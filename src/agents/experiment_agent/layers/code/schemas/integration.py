"""
Integration-related data models.

Contains dataclasses for representing:
- Stack frames from tracebacks
- Test failure information
- Integration issues
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class StackFrame:
    """A single frame in the call stack."""

    file_path: str
    line_num: int
    func_name: str
    code_line: str = ""
    is_project_file: bool = False

    def format(self) -> str:
        """Format this frame for display."""
        loc = f"{self.file_path}:{self.line_num} in {self.func_name}"
        if self.code_line:
            return f"{loc}\n    > {self.code_line.strip()}"
        return loc


@dataclass
class TestFailureInfo:
    """Detailed information about a test failure."""

    test_file: str
    test_name: str
    error_type: str
    error_message: str
    root_cause: str
    call_stack: List[StackFrame] = field(default_factory=list)
    assertion_details: str = ""
    impl_files: List[str] = field(default_factory=list)
    raw_traceback: str = ""

    def __str__(self):
        return f"[{self.test_file}] {self.test_name}: {self.error_type} - {self.root_cause[:100]}"

    def format_for_llm(self) -> str:
        """Format this failure for LLM consumption."""
        lines = [f"**{self.test_file}::{self.test_name}**", ""]
        lines.append(f"**Root Cause:** {self.error_type}: {self.root_cause}")
        if self.assertion_details:
            lines.append(f"**Assertion:** {self.assertion_details}")
        if self.call_stack:
            lines.append("")
            lines.append("**Call Stack (project files):**")
            project_frames = [f for f in self.call_stack if f.is_project_file]
            for i, frame in enumerate(project_frames[:8], 1):
                lines.append(f"  {i}. {frame.format()}")
        if self.impl_files:
            lines.append("")
            lines.append(f"**Files to Fix:** {', '.join(self.impl_files[:5])}")
        return "\n".join(lines)


class IntegrationIssue:
    """Represents an integration issue found."""

    def __init__(
        self,
        severity: str,  # "error" | "warning"
        file_path: str,
        issue_type: str,
        message: str,
        suggestion: str = "",
    ):
        self.severity = severity
        self.file_path = file_path
        self.issue_type = issue_type
        self.message = message
        self.suggestion = suggestion

    def __str__(self):
        return f"[{self.severity.upper()}] {self.file_path}: {self.issue_type} - {self.message}"
