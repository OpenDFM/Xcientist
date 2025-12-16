"""
DAG Scheduler - Unified Dependency Graph Management

Provides:
- Dependency graph construction
- Cycle detection (DAG validation)
- Topological sorting
- Ready task detection for parallel execution

Used by both Code Layer and Science Layer managers.
"""

import threading
from typing import Dict, Set, List, TypeVar, Generic
from collections import defaultdict
from enum import Enum


class TaskStatus(str, Enum):
    """Unified task status enum."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


T = TypeVar("T")


class DAGScheduler(Generic[T]):
    """
    Generic DAG-based task scheduler.

    Thread-safe implementation for parallel task scheduling.

    Type parameter T is the task identifier type (usually str).
    """

    def __init__(self, dependency_graph: Dict[T, Set[T]]):
        """
        Initialize the DAG scheduler.

        Args:
            dependency_graph: Dictionary mapping task_id -> set of dependencies
                             (task_ids that must complete before this task)
        """
        self._lock = threading.RLock()
        self.dep_graph: Dict[T, Set[T]] = {
            k: set(v) for k, v in dependency_graph.items()
        }
        self.reverse_graph: Dict[T, Set[T]] = self._build_reverse_graph()
        self._validate_dag()

    def _build_reverse_graph(self) -> Dict[T, Set[T]]:
        """
        Build reverse dependency graph (dependents).

        Returns:
            Dictionary mapping task_id -> set of tasks that depend on it
        """
        reverse: Dict[T, Set[T]] = defaultdict(set)
        for task_id, deps in self.dep_graph.items():
            for dep in deps:
                reverse[dep].add(task_id)
        return dict(reverse)

    def _validate_dag(self) -> bool:
        """
        Validate that the dependency graph is a valid DAG (no cycles).

        Uses DFS-based cycle detection.

        Returns:
            True if valid DAG

        Raises:
            ValueError: If cycle detected
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[T, int] = {node: WHITE for node in self.dep_graph}

        def dfs(node: T, path: List[T]) -> bool:
            color[node] = GRAY
            path.append(node)

            for dep in self.dep_graph.get(node, set()):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    # Found cycle
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    raise ValueError(
                        f"Circular dependency detected: {' -> '.join(str(c) for c in cycle)}"
                    )
                if color[dep] == WHITE:
                    dfs(dep, path)

            path.pop()
            color[node] = BLACK
            return True

        for node in self.dep_graph:
            if color[node] == WHITE:
                dfs(node, [])

        return True

    def get_ready_tasks(self, completed: Set[T], in_progress: Set[T]) -> List[T]:
        """
        Get list of tasks that are ready to execute.

        A task is ready when:
        1. It's not completed or in progress
        2. All its dependencies are completed

        Args:
            completed: Set of completed task IDs
            in_progress: Set of in-progress task IDs

        Returns:
            List of task IDs that are ready to execute
        """
        with self._lock:
            ready = []
            for task_id in self.dep_graph:
                if task_id in completed or task_id in in_progress:
                    continue

                # Check if all dependencies are completed
                deps = self.dep_graph.get(task_id, set())
                if deps <= completed:  # All deps are in completed set
                    ready.append(task_id)

            return ready

    def get_unblocked_tasks(self, newly_completed: T, completed: Set[T]) -> List[T]:
        """
        Get tasks that become unblocked after a task completes.

        Args:
            newly_completed: The task that just completed
            completed: Full set of completed tasks (including newly_completed)

        Returns:
            List of task IDs that are now ready to execute
        """
        with self._lock:
            dependents = self.reverse_graph.get(newly_completed, set())
            newly_ready = []

            for dep in dependents:
                if dep in completed:
                    continue
                # Check if all dependencies of this dependent are now completed
                if self.dep_graph.get(dep, set()) <= completed:
                    newly_ready.append(dep)

            return newly_ready

    def topological_sort(self) -> List[T]:
        """
        Perform topological sort on the dependency graph.

        Returns:
            List of task IDs in topological order (dependencies first)
        """
        with self._lock:
            in_degree: Dict[T, int] = {node: 0 for node in self.dep_graph}

            for node, deps in self.dep_graph.items():
                in_degree[node] = len(deps & set(self.dep_graph.keys()))

            # Start with nodes that have no dependencies
            queue = [node for node, degree in in_degree.items() if degree == 0]
            result: List[T] = []

            while queue:
                node = queue.pop(0)
                result.append(node)

                # Reduce in-degree of dependents
                for dependent in self.reverse_graph.get(node, set()):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

            return result

    def get_no_deps_count(self) -> int:
        """Get count of tasks with no dependencies."""
        return sum(1 for deps in self.dep_graph.values() if len(deps) == 0)

    def get_graph_stats(self) -> Dict[str, int]:
        """Get statistics about the dependency graph."""
        return {
            "total_nodes": len(self.dep_graph),
            "total_edges": sum(len(deps) for deps in self.dep_graph.values()),
            "no_deps_count": self.get_no_deps_count(),
        }

    def to_serializable(self) -> Dict[str, List[str]]:
        """Convert to JSON-serializable format."""
        return {str(k): [str(v) for v in vals] for k, vals in self.dep_graph.items()}

    @classmethod
    def from_serializable(cls, data: Dict[str, List[str]]) -> "DAGScheduler[str]":
        """Create from JSON-serializable format."""
        dep_graph = {k: set(v) for k, v in data.items()}
        return cls(dep_graph)


def build_dependency_graph_from_items(
    items: List[T],
    get_id: callable,
    get_deps: callable,
) -> Dict[str, Set[str]]:
    """
    Build a dependency graph from a list of items.

    Args:
        items: List of items (e.g., FileSpec, ExperimentTask)
        get_id: Function to extract ID from item
        get_deps: Function to extract dependencies from item

    Returns:
        Dictionary mapping item_id -> set of dependencies
    """
    all_ids = {get_id(item) for item in items}
    graph: Dict[str, Set[str]] = {}

    for item in items:
        item_id = get_id(item)
        deps = set(get_deps(item)) & all_ids  # Only include deps in this set
        graph[item_id] = deps

    return graph
