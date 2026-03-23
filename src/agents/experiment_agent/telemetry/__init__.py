"""
Console hooks and lightweight telemetry helpers.
"""

from typing import Optional

from src.agents.experiment_agent.telemetry.hooks import (
    Colors,
    OHColors,
    VerboseRunHooks,
    create_hooks,
)


def print_phase(
    title: str, subtitle: str = "", phase_num: Optional[int] = None, width: int = 60
) -> None:
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
    "OHColors",
    "VerboseRunHooks",
    "create_hooks",
    "print_phase",
]
