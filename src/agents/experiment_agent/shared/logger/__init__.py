"""
Logger module for SuperAgent system.

Provides:
- Custom hooks for intercepting agent execution
- Colored terminal output for agent activities
"""

from typing import Optional

from src.agents.experiment_agent.shared.logger.hooks import (
    Colors,
    VerboseRunHooks,
    create_hooks,
)


def print_phase(
    title: str, subtitle: str = "", phase_num: Optional[int] = None, width: int = 60
) -> None:
    """
    Print a simple, consistent phase header.

    Args:
        title: Phase title
        subtitle: Optional one-line subtitle
        phase_num: Optional phase number (for layered phases)
        width: Output width
    """
    line = "=" * int(width)
    print(f"\n{line}")
    if phase_num is None:
        print(f">>> {title} <<<")
    else:
        print(f"  Phase {phase_num}: {title}")
    if subtitle:
        print(f"    {subtitle}")
    print(line)


__all__ = [
    "Colors",
    "VerboseRunHooks",
    "create_hooks",
    "print_phase",
]
