from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class TicketPriority(str, Enum):
    """Priority levels for optimization tickets."""

    CRITICAL = "critical"  # Blocking issue, must fix immediately
    HIGH = "high"  # Significant performance issue
    MEDIUM = "medium"  # Moderate issue, should fix
    LOW = "low"  # Minor issue, nice to have


class TicketType(str, Enum):
    """Types of optimization tickets."""

    PERFORMANCE = "PerformanceIssue"  # Model not converging, poor metrics
    IMPLEMENTATION = "ImplementationBug"  # Errors in execution, crashes
    CONFIGURATION = "ConfigurationIssue"  # Wrong hyperparameters, settings
    DATA = "DataIssue"  # Problems with data loading, preprocessing
    ARCHITECTURE = "ArchitectureIssue"  # Design problems, structural issues
    MEMORY = "MemoryIssue"  # Out of memory, memory leaks
    NUMERICAL = "NumericalIssue"  # NaN, Inf, numerical instability


class OptimizationTicket(BaseModel):
    """
    A single optimization request from Science Layer to Code Layer.

    This is the primary interface for the Optimization Request Protocol (ORP).
    The Code Layer uses these tickets to understand what needs to be fixed.
    """

    # Required fields
    file_path: str = Field(
        description="Path to the file that needs modification (relative to project root)"
    )
    issue_type: str = Field(
        description="Type of issue: PerformanceIssue, ImplementationBug, ConfigurationIssue, etc."
    )
    message: str = Field(description="Detailed description of the issue")

    # Recommended fields
    suggestion: str = Field(default="", description="Specific fix recommendation")
    priority: TicketPriority = Field(
        default=TicketPriority.MEDIUM, description="Priority level of this ticket"
    )

    # Context fields
    experiment_id: Optional[str] = Field(
        default=None, description="ID of the experiment that revealed this issue"
    )
    metrics_context: Dict[str, float] = Field(
        default_factory=dict, description="Relevant metrics when issue was discovered"
    )
    error_trace: Optional[str] = Field(
        default=None, description="Stack trace if this is a crash/error"
    )

    # Code context
    line_number: Optional[int] = Field(
        default=None, description="Specific line number if known"
    )
    code_snippet: Optional[str] = Field(
        default=None, description="Relevant code snippet"
    )

    def to_fix_ticket(self) -> Dict[str, Any]:
        """Convert to the format expected by Code Manager.fix_files()."""
        return {
            "file_path": self.file_path,
            "issue_type": self.issue_type,
            "message": self.message,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationTicket":
        """Create an OptimizationTicket from a dictionary."""
        return cls(**data)


class OptimizationBatch(BaseModel):
    """
    A batch of optimization tickets sent from Science Layer to Code Layer.
    """

    tickets: List[OptimizationTicket] = Field(
        description="List of optimization tickets"
    )
    experiment_id: str = Field(description="Experiment ID that generated these tickets")
    iteration: int = Field(
        default=1, description="Iteration number in the optimization loop"
    )
    summary: str = Field(
        default="", description="Summary of why these optimizations are needed"
    )

    @property
    def critical_count(self) -> int:
        """Count of critical priority tickets."""
        return sum(1 for t in self.tickets if t.priority == TicketPriority.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count of high priority tickets."""
        return sum(1 for t in self.tickets if t.priority == TicketPriority.HIGH)

    def to_fix_tickets(self) -> List[Dict[str, Any]]:
        """Convert all tickets to fix format for Code Manager."""
        return [t.to_fix_ticket() for t in self.tickets]

    def group_by_file(self) -> Dict[str, List[OptimizationTicket]]:
        """Group tickets by file path."""
        groups: Dict[str, List[OptimizationTicket]] = {}
        for ticket in self.tickets:
            if ticket.file_path not in groups:
                groups[ticket.file_path] = []
            groups[ticket.file_path].append(ticket)
        return groups

    def sort_by_priority(self) -> List[OptimizationTicket]:
        """Return tickets sorted by priority (critical first)."""
        priority_order = {
            TicketPriority.CRITICAL: 0,
            TicketPriority.HIGH: 1,
            TicketPriority.MEDIUM: 2,
            TicketPriority.LOW: 3,
        }
        return sorted(self.tickets, key=lambda t: priority_order.get(t.priority, 4))


def create_performance_ticket(
    file_path: str,
    metric_name: str,
    expected_value: float,
    actual_value: float,
    suggestion: str = "",
) -> OptimizationTicket:
    """Create a performance issue ticket."""
    return OptimizationTicket(
        file_path=file_path,
        issue_type=TicketType.PERFORMANCE.value,
        message=f"Metric '{metric_name}' is {actual_value:.4f}, expected ~{expected_value:.4f}",
        suggestion=suggestion or f"Review the implementation to improve {metric_name}",
        priority=TicketPriority.HIGH,
        metrics_context={metric_name: actual_value},
    )


def create_crash_ticket(
    file_path: str,
    error_message: str,
    stack_trace: str = "",
    line_number: Optional[int] = None,
) -> OptimizationTicket:
    """Create a crash/error ticket."""
    return OptimizationTicket(
        file_path=file_path,
        issue_type=TicketType.IMPLEMENTATION.value,
        message=error_message,
        suggestion="Fix the error causing the crash",
        priority=TicketPriority.CRITICAL,
        error_trace=stack_trace,
        line_number=line_number,
    )


def create_numerical_ticket(
    file_path: str,
    issue_description: str,
    suggestion: str = "",
) -> OptimizationTicket:
    """Create a numerical stability ticket (NaN, Inf, etc.)."""
    return OptimizationTicket(
        file_path=file_path,
        issue_type=TicketType.NUMERICAL.value,
        message=issue_description,
        suggestion=suggestion or "Check for numerical stability issues",
        priority=TicketPriority.HIGH,
    )


def tickets_from_science_analysis(
    analysis_tickets: List[Dict[str, Any]],
    experiment_id: str = "",
) -> List[OptimizationTicket]:
    """
    Convert optimization tickets from ScienceAnalysis to OptimizationTicket objects.

    Args:
        analysis_tickets: List of ticket dicts from ScienceAnalysis.optimization_tickets
        experiment_id: Optional experiment ID

    Returns:
        List of OptimizationTicket objects
    """
    tickets = []
    for ticket_data in analysis_tickets:
        ticket = OptimizationTicket(
            file_path=ticket_data.get("file_path", "unknown"),
            issue_type=ticket_data.get("issue_type", "UnknownIssue"),
            message=ticket_data.get("message", "No message provided"),
            suggestion=ticket_data.get("suggestion", ""),
            experiment_id=experiment_id,
        )
        tickets.append(ticket)
    return tickets
