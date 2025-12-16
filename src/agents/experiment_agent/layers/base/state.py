"""
Unified State Management for Agent Workflows.

This module implements a persistent state machine that supports:
1. Breakpoint resume (crash recovery)
2. Step-based tracking (one step per Blueprint)
3. Unified interface for Code and Science agents

Design:
- Each step file corresponds to one Blueprint execution
- Task completion updates the current step file in-place
- Resume by loading the latest step file
"""

import json
import os
import glob
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class GlobalPhase(str, Enum):
    """Universal phases for any autonomous agent workflow."""

    # Common Phases
    INIT = "INIT"
    PLANNING = "PLANNING"

    # Execution Phases (Loop)
    EXECUTION = "EXECUTION"
    VERIFICATION = "VERIFICATION"
    REFINEMENT = "REFINEMENT"


class StepStatus(str, Enum):
    """Completion status of a step (separate from phase)."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskStatus(str, Enum):
    """Status of individual tasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class TaskState:
    """State of a single task for checkpoint/resume."""

    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "attempts": self.attempts,
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TaskState":
        return cls(
            status=TaskStatus(data.get("status", "pending")),
            attempts=data.get("attempts", 0),
            last_error=data.get("last_error"),
        )


@dataclass
class ExecutionState:
    """State for a single Blueprint execution (one step file)."""

    step_index: int
    phase: GlobalPhase
    status: StepStatus
    timestamp: str
    # Logical namespace/workflow identifier, e.g. "code" or "science".
    # This enables a unified steps directory with a global step_index while still
    # allowing each layer to resume from its latest step deterministically.
    namespace: Optional[str] = None
    blueprint_id: Optional[str] = None  # Reference to blueprint file (hash)
    tasks: Dict[str, TaskState] = field(default_factory=dict)  # Task states
    meta: Dict[str, Any] = field(default_factory=dict)  # Extra metadata

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "step_index": self.step_index,
            "phase": self.phase.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "namespace": self.namespace,
            "blueprint_id": self.blueprint_id,
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ExecutionState":
        tasks = {}
        for task_id, task_data in data.get("tasks", {}).items():
            tasks[task_id] = TaskState.from_dict(task_data)

        return cls(
            step_index=data["step_index"],
            phase=GlobalPhase(data["phase"]),
            status=StepStatus(data["status"]),
            timestamp=data["timestamp"],
            namespace=data.get("namespace"),
            blueprint_id=data.get("blueprint_id"),
            tasks=tasks,
            meta=data.get("meta", {}),
        )


class StateManager:
    """
    Manages the persistence of agent state.

    Directory Structure:
    workspaces/{project}/cached/
      ├── blueprints/        # Blueprint files (saved once)
      │   └── {hash}.json
      └── execution/
          └── steps/         # Step files (one per Blueprint, unified across layers)
              ├── step_0000.json
              └── ...

    Each step file contains:
    - step_index: Index of this step
    - phase: Current phase (INIT, EXECUTION, COMPLETED, etc.)
    - timestamp: Last update time
    - namespace: Which workflow wrote this step (e.g., "code" or "science")
    - blueprint_id: Hash of the blueprint
    - tasks: Dict of task_id -> TaskState
    - meta: Extra metadata
    """

    def __init__(self, workspace_root: str, namespace: Optional[str] = None):
        """
        Args:
            workspace_root: Experiment workspace root (contains cached/)
            namespace: Optional execution namespace to avoid collisions between workflows.
                       Examples: "code", "science". If None, uses legacy path cached/execution/steps.
        """
        self.workspace_root = workspace_root
        self.namespace = (namespace or "").strip().lower() or None
        self.blueprints_dir = os.path.join(workspace_root, "cached", "blueprints")

        # Unified execution steps directory across all layers.
        self.execution_dir = os.path.join(workspace_root, "cached", "execution")
        self.legacy_steps_dir = os.path.join(self.execution_dir, "steps")
        # Always use the unified steps directory for new writes.
        self.steps_dir = self.legacy_steps_dir

        self._ensure_dirs()
        self.current_state: Optional[ExecutionState] = None
        self.current_data: Dict[str, Any] = {}  # The actual Blueprint (cached)

    def _ensure_dirs(self):
        os.makedirs(self.blueprints_dir, exist_ok=True)
        os.makedirs(self.steps_dir, exist_ok=True)

    def _iter_all_step_files(self) -> List[str]:
        """
        List all step files across both the unified directory and any legacy
        namespaced subdirectories (for backward compatibility).
        """
        step_files: List[str] = []

        # Unified directory (new canonical location)
        step_files.extend(glob.glob(os.path.join(self.legacy_steps_dir, "step_*.json")))

        # Legacy namespaced directories (older runs)
        try:
            ns_root = os.path.join(self.execution_dir)
            for name in os.listdir(ns_root):
                if name in ["steps"]:
                    continue
                ns_steps = os.path.join(ns_root, name, "steps")
                if os.path.isdir(ns_steps):
                    step_files.extend(glob.glob(os.path.join(ns_steps, "step_*.json")))
        except Exception:
            pass

        # De-duplicate and sort by numeric step_index (not by path string).
        unique = list(set(step_files))

        def _extract_step_index(path: str) -> int:
            try:
                base = os.path.basename(path)
                return int(base.replace("step_", "").replace(".json", ""))
            except Exception:
                return -1

        unique.sort(key=lambda p: (_extract_step_index(p), p))
        return unique

    def _get_latest_step_file(
        self, namespace_filter: Optional[str] = None
    ) -> Optional[str]:
        """
        Find the latest step file by step_index, optionally filtering by namespace.

        - If namespace_filter is None: return the latest step overall.
        - If namespace_filter is set: return the latest step whose JSON has a matching
          `namespace` field OR (for migration) `meta["namespace"]`.
        """
        ns = (namespace_filter or "").strip().lower() or None
        step_files = self._iter_all_step_files()
        if not step_files:
            return None

        if ns is None:
            return step_files[-1]

        # Scan from newest to oldest and pick the first matching namespace.
        for path in reversed(step_files):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                file_ns = (data.get("namespace") or "").strip().lower() or None
                if file_ns is None:
                    meta = data.get("meta", {})
                    if isinstance(meta, dict):
                        file_ns = (meta.get("namespace") or "").strip().lower() or None
                if file_ns == ns:
                    return path
            except Exception:
                continue
        return None

    def _get_next_step_index(self) -> int:
        """
        Get the next global step index based on existing step files.

        IMPORTANT: This is global across layers (code/science) to keep a single
        monotonic step id sequence in the unified steps directory.
        """
        latest = self._get_latest_step_file(namespace_filter=None)
        if not latest:
            return 0
        basename = os.path.basename(latest)
        try:
            index = int(basename.replace("step_", "").replace(".json", ""))
            return index + 1
        except ValueError:
            return 0

    def get_next_step_index(self) -> int:
        """
        Public API: get the next step index.

        Callers should not rely on the private method _get_next_step_index().
        """
        return self._get_next_step_index()

    @staticmethod
    def format_step4(step_index: int) -> str:
        """Format step index as 4-digit string, e.g. 3 -> \"0003\"."""
        return f"{int(step_index):04d}"

    def init_state(
        self,
        initial_data: Dict[str, Any] = None,
        blueprint_id: str = None,
        task_ids: List[str] = None,
        step_index: Optional[int] = None,
    ) -> None:
        """Initialize a new step for a new Blueprint."""
        # Initialize tasks from task_ids
        tasks = {}
        if task_ids:
            for task_id in task_ids:
                tasks[task_id] = TaskState(status=TaskStatus.PENDING)

        if step_index is None:
            step_index = self._get_next_step_index()
        else:
            step_index = int(step_index)

        self.current_state = ExecutionState(
            step_index=step_index,
            phase=GlobalPhase.INIT,
            status=StepStatus.RUNNING,
            timestamp=self._get_timestamp(),
            namespace=self.namespace,
            blueprint_id=blueprint_id,
            tasks=tasks,
        )
        self.current_data = initial_data or {}
        self._save_current_step()

        logger.info(f"Initialized new step {step_index} with {len(tasks)} tasks")

    def load(self) -> bool:
        """
        Load the latest step from disk.
        Returns True if step exists and was loaded, False if new project.
        """
        latest_step = self._get_latest_step_file(namespace_filter=self.namespace)
        if not latest_step:
            return False

        try:
            # Load the latest step file
            with open(latest_step, "r", encoding="utf-8") as f:
                state_dict = json.load(f)
            self.current_state = ExecutionState.from_dict(state_dict)
            # Migration: older runs may store namespace only in meta. Promote it.
            if self.current_state.namespace is None:
                meta = self.current_state.meta or {}
                if isinstance(meta, dict):
                    promoted = (meta.get("namespace") or "").strip().lower() or None
                    if promoted:
                        self.current_state.namespace = promoted

            # Migration: older runs could leave phase at INIT even after tasks progressed.
            # If any task is non-pending, promote phase to EXECUTION and persist.
            if self.current_state.phase == GlobalPhase.INIT and any(
                t.status != TaskStatus.PENDING
                for t in self.current_state.tasks.values()
            ):
                self.current_state.phase = GlobalPhase.EXECUTION
                self.current_state.timestamp = self._get_timestamp()
                self._save_current_step()

            # Ensure canonical dirs exist for subsequent saves.
            os.makedirs(self.steps_dir, exist_ok=True)

            # Load Blueprint from blueprints directory (if exists)
            if self.current_state.blueprint_id:
                blueprint_path = os.path.join(
                    self.blueprints_dir, f"{self.current_state.blueprint_id}.json"
                )
                if os.path.exists(blueprint_path):
                    with open(blueprint_path, "r", encoding="utf-8") as f:
                        bp_data = json.load(f)
                        self.current_data = bp_data.get("blueprint", bp_data)

            logger.info(
                f"Resumed from step {self.current_state.step_index} "
                f"({self.current_state.phase}), "
                f"Tasks: {len(self.current_state.tasks)} tracked"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False

    def update_task(
        self,
        task_id: str,
        status: TaskStatus = None,
        attempts: int = None,
        last_error: str = None,
        save: bool = True,
    ) -> None:
        """
        Update the state of a specific task and save to current step file.

        Args:
            task_id: ID of the task to update
            status: New status
            attempts: Number of attempts
            last_error: Last error message
            save: Whether to save immediately (default True)
        """
        if self.current_state is None:
            return

        if task_id not in self.current_state.tasks:
            self.current_state.tasks[task_id] = TaskState()

        task = self.current_state.tasks[task_id]
        if status is not None:
            task.status = status
        if attempts is not None:
            task.attempts = attempts
        if last_error is not None:
            task.last_error = last_error

        # Auto-promote phase from INIT -> EXECUTION on the first meaningful task update.
        # Keep REFINEMENT/VERIFICATION/etc intact to avoid overwriting higher-level orchestration phases.
        if (
            self.current_state.phase == GlobalPhase.INIT
            and status is not None
            and status != TaskStatus.PENDING
        ):
            self.current_state.phase = GlobalPhase.EXECUTION

        # Update timestamp
        self.current_state.timestamp = self._get_timestamp()

        # Save immediately by default
        if save:
            self._save_current_step()

    def ensure_tasks(self, task_ids: List[str], save: bool = True) -> None:
        """
        Ensure the current state tracks the provided task_ids.

        This is a non-destructive operation: it only adds missing tasks as PENDING.
        Useful for migrating legacy step files that had an empty tasks dict.
        """
        if self.current_state is None:
            return
        if not task_ids:
            return

        for task_id in task_ids:
            if task_id not in self.current_state.tasks:
                self.current_state.tasks[task_id] = TaskState(status=TaskStatus.PENDING)

        self.current_state.timestamp = self._get_timestamp()
        if save:
            self._save_current_step()

    def set_phase(self, phase: GlobalPhase, meta: Dict[str, Any] = None) -> None:
        """Update the phase of current step and save."""
        if self.current_state is None:
            return

        self.current_state.phase = phase
        self.current_state.timestamp = self._get_timestamp()
        if meta is not None:
            self.current_state.meta = meta

        self._save_current_step()

    def set_status(self, status: StepStatus, meta: Dict[str, Any] = None) -> None:
        """Update the status of current step and save (does NOT change phase)."""
        if self.current_state is None:
            return

        self.current_state.status = status
        self.current_state.timestamp = self._get_timestamp()
        if meta is not None:
            self.current_state.meta = meta

        self._save_current_step()

    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get the state of a specific task."""
        if self.current_state is None:
            return None
        return self.current_state.tasks.get(task_id)

    def get_completed_task_ids(self) -> List[str]:
        """Get list of completed task IDs."""
        if self.current_state is None:
            return []
        return [
            task_id
            for task_id, task in self.current_state.tasks.items()
            if task.status == TaskStatus.COMPLETED
        ]

    def get_pending_task_ids(self) -> List[str]:
        """Get list of pending task IDs."""
        if self.current_state is None:
            return []
        return [
            task_id
            for task_id, task in self.current_state.tasks.items()
            if task.status == TaskStatus.PENDING
        ]

    def _save_current_step(self) -> None:
        """Save current state to the step file (in-place update)."""
        if self.current_state is None:
            return

        step_path = self._get_step_path(self.current_state.step_index)

        snapshot = self.current_state.to_dict()

        # Atomic write
        tmp_path = step_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        os.rename(tmp_path, step_path)

        logger.debug(f"Updated step {self.current_state.step_index}")

    def _get_step_path(self, index: int) -> str:
        return os.path.join(self.steps_dir, f"step_{index:04d}.json")

    def _get_timestamp(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S")

    @property
    def phase(self) -> GlobalPhase:
        return self.current_state.phase if self.current_state else GlobalPhase.INIT

    @property
    def status(self) -> StepStatus:
        return self.current_state.status if self.current_state else StepStatus.RUNNING

    @property
    def step(self) -> int:
        return self.current_state.step_index if self.current_state else 0

    # Legacy compatibility - save_step now just updates phase
    def save_step(
        self,
        phase: GlobalPhase,
        action: str = "",
        active_task_id: str = None,
        meta: Dict[str, Any] = None,
        data: Dict[str, Any] = None,
    ) -> None:
        """Legacy method - now just updates phase of current step."""
        self.set_phase(phase, meta)
