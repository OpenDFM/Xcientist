"""
Shared Schemas for SuperAgent.

Contains inter-layer communication protocols:
- OptimizationTicket: Request from Science Layer to Code Layer
- OptimizationBatch: Batch of optimization tickets
- Helper functions for ticket creation
"""

from src.agents.experiment_agent.shared.schemas.protocols import (
    OptimizationTicket,
    OptimizationBatch,
    TicketPriority,
    TicketType,
    create_performance_ticket,
    create_crash_ticket,
    create_numerical_ticket,
    tickets_from_science_analysis,
)

__all__ = [
    "OptimizationTicket",
    "OptimizationBatch",
    "TicketPriority",
    "TicketType",
    "create_performance_ticket",
    "create_crash_ticket",
    "create_numerical_ticket",
    "tickets_from_science_analysis",
]

