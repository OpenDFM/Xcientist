"""Rich progress helpers shared across Idea Agent workflows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Optional, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


T = TypeVar("T")

_PROGRESS_CONSOLE = Console(stderr=False)


def create_progress(*, transient: bool = False) -> Progress:
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=_PROGRESS_CONSOLE,
        transient=transient,
    )


def iter_with_progress(
    iterable: Iterable[T],
    *,
    description: str,
    total: Optional[int] = None,
    transient: bool = False,
) -> Iterator[T]:
    resolved_total = total
    if resolved_total is None:
        try:
            resolved_total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            resolved_total = None

    with create_progress(transient=transient) as progress:
        task_id = progress.add_task(description, total=resolved_total)
        for item in iterable:
            yield item
            progress.advance(task_id)
