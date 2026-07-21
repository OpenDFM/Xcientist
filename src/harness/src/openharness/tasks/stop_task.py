"""Stop task helper."""

from __future__ import annotations

from openharness.tasks.manager import get_task_manager
from openharness.tasks.types import TaskRecord


async def stop_task(task_id: str) -> TaskRecord:
    """Stop a running task via the default task manager."""
    return await get_task_manager().stop_task(task_id)
