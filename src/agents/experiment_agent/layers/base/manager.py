"""
Base Manager - Common base class for Manager agents.

Provides:
- DAG-based task scheduling
- Parallel worker execution
- Task status tracking
- Resume/checkpoint support
- State management (shared between Code and Science layers)

Used by both Code Manager and Experiment Manager.
"""

import os
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Set, List, Optional, Any, TypeVar, Generic

from src.agents.experiment_agent.layers.base.agent import BaseAgent
from src.agents.experiment_agent.layers.base.state import (
    StateManager,
    GlobalPhase,
    StepStatus,
    TaskStatus as StateTaskStatus,
)
from src.agents.experiment_agent.shared.utils.dag import DAGScheduler, TaskStatus
from src.agents.experiment_agent.shared.logger.hooks import create_hooks
from src.agents.experiment_agent.shared.utils.config import MAX_SCIENCE_ITERATIONS


logger = logging.getLogger(__name__)

T = TypeVar("T")  # Task type
R = TypeVar("R")  # Result type


class TaskWrapper(Generic[T]):
    """
    Wrapper for tracking task execution state.
    """

    def __init__(self, task: T, priority: int = 0, max_attempts: Optional[int] = None):
        self.task = task
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.attempts = 0
        self.max_attempts = (
            max_attempts if max_attempts is not None else MAX_SCIENCE_ITERATIONS
        )
        self.last_error: Optional[str] = None
        self.result: Optional[Any] = None


class BaseManager(BaseAgent, ABC, Generic[T, R]):
    """
    Abstract base class for Manager agents.

    Implements DAG-based parallel task scheduling with:
    - Wave-based execution (tasks at same dependency level run in parallel)
    - Worker pooling with semaphore
    - Automatic retry on failure
    - Progress tracking and logging
    """

    def __init__(
        self,
        agent_type: str,
        model: str,
        worker_model: str,
        max_parallel_workers: int = 5,
        verbose: bool = True,
    ):
        """
        Initialize the base manager.

        Args:
            agent_type: Type identifier for logging
            model: Model name for manager
            worker_model: Model name for workers
            max_parallel_workers: Maximum concurrent workers
            verbose: Enable verbose output
        """
        super().__init__(
            agent_type=agent_type,
            model=model,
            verbose=verbose,
        )

        self.worker_model = worker_model
        self.max_parallel_workers = max_parallel_workers

        # Task tracking
        self.tasks: Dict[str, TaskWrapper[T]] = {}
        self.scheduler: Optional[DAGScheduler] = None

        # State management (initialized by subclass or init_state_manager)
        self.state_manager: Optional[StateManager] = None

    def init_state_manager(
        self, workspace_root: str, namespace: Optional[str] = None
    ) -> None:
        """Initialize the StateManager for this manager."""
        if self.state_manager is None:
            self.state_manager = StateManager(workspace_root, namespace=namespace)

    def try_resume_state(self) -> bool:
        """
        Try to load and resume from saved state.

        Returns:
            True if state was loaded successfully, False otherwise.
        """
        if self.state_manager is None:
            return False
        return self.state_manager.load()

    def init_state_if_needed(
        self,
        task_ids: List[str],
        initial_data: Dict[str, Any] = None,
        blueprint_id: str = None,
    ) -> None:
        """
        Initialize state if not already initialized.

        Args:
            task_ids: List of task IDs to track
            initial_data: Optional initial data (blueprint/plan)
            blueprint_id: Optional blueprint hash ID
        """
        if self.state_manager is None:
            return

        if self.state_manager.current_state is None:
            self.state_manager.init_state(
                initial_data=initial_data,
                blueprint_id=blueprint_id,
                task_ids=task_ids,
            )
        else:
            # Legacy migration: if a step exists but tasks list is missing/incomplete, fill it in.
            self.state_manager.ensure_tasks(task_ids)

    def mark_task_completed(
        self,
        task_id: str,
        attempts: int,
    ) -> None:
        """Mark a task as completed and save state."""
        if self.state_manager is None:
            return

        self.state_manager.update_task(
            task_id,
            status=StateTaskStatus.COMPLETED,
            attempts=attempts,
            last_error=None,
        )

    def mark_task_failed(
        self,
        task_id: str,
        attempts: int,
        last_error: str = None,
    ) -> None:
        """Mark a task as failed and save state."""
        if self.state_manager is None:
            return

        self.state_manager.update_task(
            task_id,
            status=StateTaskStatus.FAILED,
            attempts=attempts,
            last_error=last_error,
        )

    def complete_execution(self, status: StepStatus = StepStatus.COMPLETED) -> None:
        """Mark the step as completed/failed (separate from phase)."""
        if self.state_manager is None:
            return

        self.state_manager.set_status(status)

    def get_completed_task_ids_from_state(self) -> Set[str]:
        """Get set of completed task IDs from state manager."""
        if self.state_manager is None:
            return set()
        return set(self.state_manager.get_completed_task_ids())

    @abstractmethod
    def _get_task_id(self, task: T) -> str:
        """
        Get the unique identifier for a task.

        Args:
            task: The task object

        Returns:
            Task ID string
        """
        pass

    @abstractmethod
    def _get_task_dependencies(self, task: T) -> List[str]:
        """
        Get the dependencies for a task.

        Args:
            task: The task object

        Returns:
            List of dependency task IDs
        """
        pass

    @abstractmethod
    async def _process_task(self, wrapper: TaskWrapper[T], **kwargs) -> bool:
        """
        Process a single task.

        Args:
            wrapper: The task wrapper
            **kwargs: Additional context

        Returns:
            True if task succeeded, False otherwise
        """
        pass

    @abstractmethod
    def _create_worker(self) -> Any:
        """
        Create a new worker agent.

        Returns:
            Worker agent instance
        """
        pass

    def _build_dependency_graph(self, tasks: List[T]) -> Dict[str, Set[str]]:
        """
        Build a dependency graph from tasks.

        Args:
            tasks: List of task objects

        Returns:
            Dictionary mapping task_id -> set of dependencies
        """
        all_ids = {self._get_task_id(task) for task in tasks}
        graph: Dict[str, Set[str]] = {}

        for task in tasks:
            task_id = self._get_task_id(task)
            deps = set(self._get_task_dependencies(task)) & all_ids
            graph[task_id] = deps

        return graph

    async def execute_tasks(
        self,
        tasks: List[T],
        resume: bool = False,
        completed_ids: Optional[Set[str]] = None,
        **kwargs,
    ) -> Dict[str, R]:
        """
        Execute tasks using DAG-based scheduling.

        Args:
            tasks: List of tasks to execute
            resume: If True, skip already completed tasks
            completed_ids: Set of already completed task IDs (for resume)
            **kwargs: Additional context passed to _process_task

        Returns:
            Dictionary mapping task_id -> result
        """
        self._log_info(f"Starting DAG execution for {len(tasks)} tasks...")
        self._log_info(f"Max parallel workers: {self.max_parallel_workers}")

        # Build dependency graph
        dep_graph = self._build_dependency_graph(tasks)
        self.scheduler = DAGScheduler(dep_graph)

        stats = self.scheduler.get_graph_stats()
        self._log_info(
            f"DAG validated: {stats['total_nodes']} nodes, {stats['total_edges']} edges"
        )
        self._log_info(f"Tasks with no dependencies: {stats['no_deps_count']}")

        # Initialize task wrappers
        for i, task in enumerate(tasks):
            task_id = self._get_task_id(task)
            wrapper = TaskWrapper(task, priority=i)

            # Mark as completed if resuming
            if resume and completed_ids and task_id in completed_ids:
                wrapper.status = TaskStatus.COMPLETED

            self.tasks[task_id] = wrapper

        # Track execution
        # Only seed "completed" from previous runs when resume=True.
        completed: Set[str] = (
            completed_ids.copy() if (resume and completed_ids) else set()
        )
        in_progress: Set[str] = set()
        semaphore = asyncio.Semaphore(self.max_parallel_workers)
        wave_number = 0
        wave_attempt = 0
        forced_wave: Optional[List[str]] = None
        forced_wave_all: Optional[Set[str]] = None

        # Main execution loop
        while True:
            # Find ready tasks.
            # Wave barrier semantics:
            # - If a wave has any retryable failures, we MUST retry those tasks first
            #   and MUST NOT enter the next wave until all tasks from the current wave
            #   are completed successfully.
            if forced_wave is not None:
                ready = [
                    tid
                    for tid in forced_wave
                    if tid not in completed and tid not in in_progress
                ]
            else:
                ready = self.scheduler.get_ready_tasks(completed, in_progress)

            if not ready and not in_progress:
                break

            if ready:
                is_retry_wave = forced_wave is not None
                if not is_retry_wave:
                    wave_number += 1
                    wave_attempt = 1
                    forced_wave_all = set(ready)
                else:
                    wave_attempt += 1
                print(f"\n{'='*60}")
                print(
                    f"{self.agent_type}: Wave {wave_number} (Attempt {wave_attempt}) - {len(ready)} tasks ready"
                )
                print(f"  Ready: {', '.join(ready)}")
                print(f"{'='*60}")

                # Mark as in progress
                for task_id in ready:
                    self.tasks[task_id].status = TaskStatus.IN_PROGRESS
                    in_progress.add(task_id)

                # Execute in parallel
                async def run_task(task_id: str) -> tuple:
                    async with semaphore:
                        wrapper = self.tasks[task_id]
                        success = await self._process_task(wrapper, **kwargs)
                        return (task_id, success)

                results = await asyncio.gather(*(run_task(tid) for tid in ready))

                # Process results
                wave_completed = []
                wave_failed = []
                retryable_failed: List[str] = []
                permanent_failed: List[str] = []
                for task_id, success in results:
                    in_progress.discard(task_id)
                    if success:
                        completed.add(task_id)
                        self.tasks[task_id].status = TaskStatus.COMPLETED
                        wave_completed.append(task_id)
                    else:
                        wrapper = self.tasks[task_id]
                        wave_failed.append(task_id)

                        # If task still has attempts left, retry it in the SAME wave.
                        # Otherwise, mark as permanently failed and stop progressing waves.
                        if wrapper.attempts < wrapper.max_attempts:
                            wrapper.status = TaskStatus.PENDING
                            retryable_failed.append(task_id)
                        else:
                            wrapper.status = TaskStatus.FAILED
                            permanent_failed.append(task_id)

                # Show DAG progress once per wave
                self._print_dag_progress(
                    wave_number,
                    wave_completed=wave_completed,
                    wave_failed=wave_failed,
                )

                # Hard barrier: do not enter next wave until ALL tasks in this wave succeed.
                # If any task is permanently failed, stop execution here.
                if permanent_failed:
                    self._log_warning(
                        f"Wave {wave_number} has permanently failed tasks; stopping before next wave: {permanent_failed}"
                    )
                    break

                if forced_wave_all is not None:
                    # Keep forcing retries until the wave's full task set is completed.
                    remaining_in_wave = sorted(list(forced_wave_all - completed))
                    if remaining_in_wave:
                        forced_wave = remaining_in_wave
                    else:
                        forced_wave = None
                        forced_wave_all = None
                        wave_attempt = 0
                else:
                    # Should not happen, but stay safe.
                    forced_wave = None

            elif in_progress:
                await asyncio.sleep(0.1)

        # Report results
        success_count = sum(
            1 for w in self.tasks.values() if w.status == TaskStatus.COMPLETED
        )
        fail_count = len(self.tasks) - success_count

        # Get list of failed tasks with reasons
        failed_details = []
        if fail_count > 0:
            for tid, w in self.tasks.items():
                if w.status == TaskStatus.FAILED:
                    failed_details.append(
                        f"  - {tid}: {w.last_error or 'Unknown error'}"
                    )

        print(f"\n{'='*60}")
        print(f"{self.agent_type}: Execution complete")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {fail_count}")
        if failed_details:
            print("  Failure Reasons:")
            print("\n".join(failed_details))
        print(f"  Total waves: {wave_number}")
        print(f"{'='*60}")

        # Collect results
        return {
            task_id: wrapper.result
            for task_id, wrapper in self.tasks.items()
            if wrapper.result is not None
        }

    def get_task_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all tasks."""
        return {
            task_id: {
                "status": wrapper.status.value,
                "attempts": wrapper.attempts,
                "last_error": wrapper.last_error,
            }
            for task_id, wrapper in self.tasks.items()
        }

    def get_failed_tasks(self) -> List[str]:
        """Get list of failed task IDs."""
        return [
            task_id
            for task_id, wrapper in self.tasks.items()
            if wrapper.status == TaskStatus.FAILED
        ]

    def get_completed_tasks(self) -> List[str]:
        """Get list of completed task IDs."""
        return [
            task_id
            for task_id, wrapper in self.tasks.items()
            if wrapper.status == TaskStatus.COMPLETED
        ]

    def _print_dag_progress(
        self,
        wave_number: int,
        wave_completed: Optional[List[str]] = None,
        wave_failed: Optional[List[str]] = None,
    ):
        """
        Print a visual DAG progress summary after each wave.

        Args:
            wave_number: Current wave number
            wave_completed: List of task IDs completed in this wave
            wave_failed: List of task IDs failed in this wave
        """
        if not self.tasks:
            return

        wave_completed = wave_completed or []
        wave_failed = wave_failed or []

        # Count by status
        pending = sum(1 for w in self.tasks.values() if w.status == TaskStatus.PENDING)
        in_progress = sum(
            1 for w in self.tasks.values() if w.status == TaskStatus.IN_PROGRESS
        )
        completed = sum(
            1 for w in self.tasks.values() if w.status == TaskStatus.COMPLETED
        )
        failed = sum(1 for w in self.tasks.values() if w.status == TaskStatus.FAILED)

        total = len(self.tasks)
        progress_pct = (completed / total * 100) if total > 0 else 0

        # Build progress bar
        bar_width = 30
        filled = int(bar_width * completed / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        # Status symbols
        symbols = {
            TaskStatus.PENDING: "○",
            TaskStatus.IN_PROGRESS: "◐",
            TaskStatus.COMPLETED: "●",
            TaskStatus.FAILED: "✗",
        }

        # Print progress header
        print(f"\n┌{'─'*58}┐")
        print(
            f"│ 📊 DAG Progress │ Wave {wave_number:2d} │ {progress_pct:5.1f}% Complete{' '*12}│"
        )
        print(f"├{'─'*58}┤")
        print(f"│ [{bar}] {completed:3d}/{total:3d}{' '*13}│")
        print(
            f"│ ● Done: {completed:3d}  ◐ Run: {in_progress:3d}  ○ Wait: {pending:3d}  ✗ Fail: {failed:3d}{' '*8}│"
        )

        # Show task status grid (up to 50 tasks)
        if total <= 50:
            print(f"├{'─'*58}┤")
            task_line = "│ "
            for task_id, wrapper in self.tasks.items():
                sym = symbols.get(wrapper.status, "?")
                if task_id in wave_completed:
                    task_line += f"\033[92m{sym}\033[0m"
                elif task_id in wave_failed:
                    task_line += f"\033[91m{sym}\033[0m"
                elif wrapper.status == TaskStatus.FAILED:
                    task_line += f"\033[91m{sym}\033[0m"
                elif wrapper.status == TaskStatus.IN_PROGRESS:
                    task_line += f"\033[93m{sym}\033[0m"
                elif wrapper.status == TaskStatus.COMPLETED:
                    task_line += f"\033[92m{sym}\033[0m"
                else:
                    task_line += sym
            task_line += " " * max(0, 56 - total) + "│"
            print(task_line)

        print(f"└{'─'*58}┘")

        # Show wave summary
        if wave_completed:
            names = [t.split("/")[-1] for t in wave_completed[:5]]
            more = f" (+{len(wave_completed)-5})" if len(wave_completed) > 5 else ""
            print(f"  ✅ Wave {wave_number}: {', '.join(names)}{more}")
        if wave_failed:
            names = [t.split("/")[-1] for t in wave_failed[:3]]
            print(f"  ❌ Failed: {', '.join(names)}")
