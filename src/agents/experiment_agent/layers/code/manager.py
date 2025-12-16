"""
Code Manager Agent - Task Orchestration

Based on the paper "Towards a Science of Scaling Agent Systems":
- Implements "Validation Bottleneck" pattern
- Provides "Context Slicing" to minimize information overload
- Parallelizes independent file implementations
- Rejects non-conforming code to maintain system integrity
"""

import os
import logging
from typing import List, Dict, Optional, Set

from src.agents.experiment_agent.layers.base.manager import BaseManager, TaskWrapper
from src.agents.experiment_agent.layers.base.state import GlobalPhase, StepStatus, TaskStatus as StateTaskStatus
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint, FileSpec
from src.agents.experiment_agent.layers.code.schemas.fix_blueprint import FixBlueprint, FixTaskSpec
from src.agents.experiment_agent.layers.code.worker import CodeWorkerAgent
from src.agents.experiment_agent.shared.utils.dag import TaskStatus
from src.agents.experiment_agent.shared.exceptions import exit_on_rate_limit
from src.agents.experiment_agent.shared.utils.config import (
    CODE_MANAGER_MODEL,
    CODE_WORKER_MODEL,
)
from src.agents.experiment_agent.shared.tools.core import (
    SecurityContext,
    validate_code_against_spec,
    run_linter,
    extract_interface_stub,
)


logger = logging.getLogger(__name__)


class CodeManagerAgent(BaseManager[FileSpec, str]):
    """
    Technical Manager / Orchestrator Agent.

    Implements the "Centralized Coordination" pattern:
    - Single point of control for all file implementations
    - Validation bottleneck prevents error propagation
    - Context isolation prevents information overload
    """

    def __init__(
        self,
        project_root: str,
        idea_md_path: str = "",
        model: str = CODE_MANAGER_MODEL,
        worker_model: str = CODE_WORKER_MODEL,
        max_parallel_workers: int = 5,
        reference_repos: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        super().__init__(
            agent_type="CodeManager",
            model=model,
            worker_model=worker_model,
            max_parallel_workers=max_parallel_workers,
            verbose=verbose,
        )

        self.project_root = os.path.abspath(project_root)
        self.reference_repos = reference_repos or []
        self.idea_md_path = str(idea_md_path or "")

        # State Management (use base class method)
        # project_root is workspace_root/project, so we go up one level
        workspace_root = os.path.dirname(self.project_root)
        self.init_state_manager(workspace_root, namespace="code")

        # Set security context
        SecurityContext.set_roots(project_root=self.project_root)

        # Track completed files for stub context
        self.completed_files: Dict[str, str] = {}  # path -> stub

        # Blueprint reference
        self.blueprint: Optional[Blueprint] = None

    def add_reference_repo(self, repo_path: str):
        """Add a reference repository for workers to consult."""
        self.reference_repos.append(repo_path)

    def _get_task_id(self, task: FileSpec) -> str:
        return task.file_path

    def _get_task_dependencies(self, task: FileSpec) -> List[str]:
        """
        Get dependencies for a task.

        TEST-FIRST PATTERN: If this file has a corresponding test_file,
        the test file is added as a dependency, ensuring tests are
        generated BEFORE implementation.
        """
        deps = list(task.dependencies)

        # Add test file as dependency (test-first pattern)
        if task.test_file and not task.is_test:
            if task.test_file not in deps:
                deps.append(task.test_file)

        return deps

    def _create_worker(self) -> CodeWorkerAgent:
        """Create a new worker agent."""
        return CodeWorkerAgent(
            model=self.worker_model,
            reference_repos=self.reference_repos,
            verbose=self.verbose,
        )

    async def execute_blueprint(
        self,
        blueprint: Blueprint,
        blueprint_id: str = None,
        resume: bool = False,
    ) -> Dict[str, str]:
        """
        Execute the blueprint using DAG-based scheduling with state persistence.

        Args:
            blueprint: The project blueprint
            blueprint_id: Hash/ID of the blueprint file (for reference)
            resume: If True, try to resume from last state
        """
        # 1. Try to resume state (use base class method)
        if resume and self.try_resume_state():
            if self.state_manager.current_data:
                try:
                    blueprint = Blueprint(**self.state_manager.current_data)
                    self._log_info(
                        f"Resumed execution from Step {self.state_manager.step} "
                        f"({self.state_manager.phase})"
                    )
                except Exception as e:
                    logger.warning(f"Failed to restore blueprint from state: {e}")

        if (
            resume
            and blueprint_id
            and self.state_manager
            and self.state_manager.current_state
            and self.state_manager.current_state.blueprint_id
            and self.state_manager.current_state.blueprint_id != blueprint_id
        ):
            self._log_warning(
                "Resume state blueprint_id mismatch; starting a new step "
                f"(state={self.state_manager.current_state.blueprint_id}, arg={blueprint_id})"
            )
            self.state_manager.current_state = None
            self.state_manager.current_data = {}

        self.blueprint = blueprint

        self._log_info(f"Starting execution for {len(blueprint.files)} files...")
        self._log_info(f"Project root: {self.project_root}")
        self._log_info(f"Entry point: {blueprint.entry_point}")

        # 2. Initialize state if needed (use base class method)
        task_ids = [f.file_path for f in blueprint.files]
        self.init_state_if_needed(
            task_ids=task_ids,
            initial_data=blueprint.model_dump(),
            blueprint_id=blueprint_id,
        )

        # Create directory structure
        self._create_directories(blueprint)

        completed_ids: Optional[Set[str]] = None
        if resume:
            # Resume must be constructed ONLY from cached/execution state (step_*.json),
            # not from "files exist on disk".
            completed_set: Set[str] = self.get_completed_task_ids_from_state()

            # Consistency guard: if state says "completed" but file is missing/empty,
            # re-run it by reverting state to pending.
            if (
                completed_set
                and self.state_manager
                and self.state_manager.current_state
            ):
                bad_completed: List[str] = []
                for task_id in list(completed_set):
                    full_path = os.path.join(self.project_root, task_id)
                    if (not os.path.exists(full_path)) or os.path.getsize(
                        full_path
                    ) <= 0:
                        completed_set.discard(task_id)
                        bad_completed.append(task_id)
                        self.state_manager.update_task(
                            task_id,
                            status=StateTaskStatus.PENDING,
                            attempts=0,
                            last_error="State marked completed, but file is missing/empty on disk; will re-run.",
                        )
                        continue

                    stub_result = extract_interface_stub(full_path)
                    if stub_result["success"]:
                        self.completed_files[task_id] = stub_result["stub"]

                if bad_completed:
                    self._log_info(
                        f"Resume state had {len(bad_completed)} completed tasks with missing/empty files; reverted to pending"
                    )

            if completed_set:
                self._log_info(
                    f"Found {len(completed_set)} completed files (from execution state)"
                )

            completed_ids = completed_set

        # Execute using base class DAG scheduler
        await self.execute_tasks(
            tasks=blueprint.files,
            resume=resume,
            completed_ids=completed_ids,
            blueprint=blueprint,
        )

        # Gate next stage: all tasks must be COMPLETED before Integration/Verification.
        incomplete = [
            (task_id, wrapper.status.value, wrapper.last_error or "")
            for task_id, wrapper in self.tasks.items()
            if wrapper.status != TaskStatus.COMPLETED
        ]
        if incomplete:
            if self.state_manager is not None:
                self.complete_execution(StepStatus.FAILED)
            details = "\n".join(
                [
                    f"- {task_id}: status={status}" + (f", error={err}" if err else "")
                    for task_id, status, err in incomplete
                ]
            )
            raise RuntimeError(
                "Code implementation did not complete all tasks; refusing to enter verification.\n"
                + details
            )

        return {
            task_id: wrapper.status.value for task_id, wrapper in self.tasks.items()
        }

    async def _process_task(self, wrapper: TaskWrapper[FileSpec], **kwargs) -> bool:
        """Process a single file implementation task."""
        file_spec = wrapper.task
        blueprint = kwargs.get("blueprint")

        print(f"\n{'─'*40}")
        print(f"CodeManager: Assigning {file_spec.file_path} to Worker...")
        print(f"  - Classes: {len(file_spec.classes)}")
        print(f"  - Functions: {len(file_spec.functions)}")
        print(f"  - Dependencies: {len(file_spec.dependencies)}")

        # Build stub context
        stub_context = self._get_stub_context(file_spec, blueprint)

        # Create worker
        worker = self._create_worker()

        while wrapper.attempts < wrapper.max_attempts:
            wrapper.attempts += 1
            print(f"\n  [Attempt {wrapper.attempts}/{wrapper.max_attempts}]")

            # Prepare feedback
            feedback = ""
            if wrapper.last_error:
                feedback = f"Previous attempt failed: {wrapper.last_error}"

            # Dispatch to worker
            # Get is_fix_mode from kwargs (default False for normal implementation retries)
            is_fix_mode = kwargs.get("is_fix_mode", False)
            try:
                _ = await worker.run_task(
                    file_spec=file_spec,
                    stub_context=stub_context,
                    project_root=self.project_root,
                    idea_md_path=self.idea_md_path,
                    feedback=feedback,
                    is_fix_mode=is_fix_mode,
                )
            except Exception as e:
                error_str = str(e)
                wrapper.last_error = f"Worker error: {error_str}"
                print(f"  ❌ Worker failed: {e}")

                exit_on_rate_limit(error_str)

                continue

            code = ""
            full_path = os.path.join(self.project_root, file_spec.file_path)
            if not os.path.exists(full_path):
                wrapper.last_error = "Worker did not write the file to disk."
                print(f"  ❌ Missing file on disk: {file_spec.file_path}")
                continue
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()
                print(f"  📖 Read code from disk: {file_spec.file_path}")
            except Exception as e:
                wrapper.last_error = f"Could not read file from disk: {e}"
                print(f"  ❌ Could not read from disk: {e}")
                continue

            is_python = str(file_spec.file_path or "").endswith(".py")
            if is_python:
                validation_result = validate_code_against_spec(code, file_spec)

                if not validation_result["valid"]:
                    wrapper.last_error = "; ".join(validation_result["errors"])
                    print(f"  ❌ Validation failed: {wrapper.last_error}")
                    continue

                lint_result = run_linter(code=code)
                if not lint_result.get("syntax_valid", True):
                    wrapper.last_error = (
                        f"Syntax error: {lint_result.get('syntax_error', 'Unknown')}"
                    )
                    print(f"  ❌ Syntax error")
                    continue
                print("  ✅ Validation passed")
            else:
                # Non-Python files (e.g., yaml/json/txt) should NOT be parsed with ast.
                if not code.strip():
                    wrapper.last_error = "File content is empty."
                    print("  ❌ Empty non-Python file")
                    continue
                print("  ✅ Non-Python file - skipped Python validation")

            # Success - write to disk
            wrapper.result = code
            full_path = os.path.join(self.project_root, file_spec.file_path)
            self._write_file(full_path, code)

            # Store stub for future workers
            if is_python:
                stub_result = extract_interface_stub(full_path)
                if stub_result["success"]:
                    self.completed_files[file_spec.file_path] = stub_result["stub"]

            # UPDATE TASK STATE (use base class method)
            self.mark_task_completed(file_spec.file_path, wrapper.attempts)

            print(f"  ✅ Completed {file_spec.file_path}")
            return True

        # Max attempts reached - mark as failed (use base class method)
        self.mark_task_failed(
            file_spec.file_path,
            wrapper.attempts,
            wrapper.last_error,
        )
        print(
            f"  ❌ FAILED {file_spec.file_path} after {wrapper.max_attempts} attempts"
        )
        return False

    def _get_stub_context(self, file_spec: FileSpec, blueprint: Blueprint) -> str:
        """Generate "Stub Context" for a worker (Context Slicing)."""
        context_parts = []

        for dep_path in file_spec.dependencies:
            # Check completed files first
            if dep_path in self.completed_files:
                context_parts.append(f"# --- Dependency: {dep_path} ---")
                context_parts.append(self.completed_files[dep_path])
                context_parts.append("")
                continue

            # Generate stub from blueprint spec
            dep_spec = next(
                (f for f in blueprint.files if f.file_path == dep_path), None
            )
            if not dep_spec:
                continue

            context_parts.append(f"# --- Dependency: {dep_path} ---")
            context_parts.append(f"# {dep_spec.description}")
            context_parts.append("")

            # Add class stubs
            for cls in dep_spec.classes:
                context_parts.append(f"class {cls.name}:")
                context_parts.append(f'    """{cls.docstring}"""')

                if cls.attributes:
                    for attr_name, attr_type in cls.attributes.items():
                        context_parts.append(f"    {attr_name}: {attr_type}")

                for method in cls.methods:
                    args = method.args if method.args else "self"
                    if "self" not in args:
                        args = "self, " + args
                    ret = f" -> {method.return_type}" if method.return_type else ""
                    context_parts.append(f"    def {method.name}({args}){ret}: ...")

                context_parts.append("")

            # Add function stubs
            for func in dep_spec.functions:
                args = func.args if func.args else ""
                ret = f" -> {func.return_type}" if func.return_type else ""
                context_parts.append(f"def {func.name}({args}){ret}:")
                context_parts.append(f'    """{func.docstring}"""')
                context_parts.append("    ...")
                context_parts.append("")

        return "\n".join(context_parts)

    def _create_directories(self, blueprint: Blueprint):
        """Create all necessary directories."""
        for file_path in blueprint.file_tree:
            full_path = os.path.join(self.project_root, file_path)
            directory = os.path.dirname(full_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                if self.verbose:
                    print(f"  Created directory: {directory}")

    def _write_file(self, path: str, content: str):
        """Write file to disk."""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    async def fix_files(
        self,
        tickets: List[Dict],
        blueprint: Blueprint,
    ) -> Dict[str, str]:
        """Fix files based on integration error tickets."""
        if not tickets:
            return {}

        self.blueprint = blueprint

        print(f"\n{'='*60}")
        print(f"CodeManager: Fixing {len(tickets)} files from integration errors...")
        print(f"{'='*60}")

        # Reset tasks for files that need fixing
        for ticket in tickets:
            file_path = ticket["file_path"]
            file_spec = next(
                (f for f in blueprint.files if f.file_path == file_path), None
            )

            if not file_spec:
                print(f"  ⚠️ No spec found for {file_path}, skipping...")
                continue

            # Create feedback from ticket
            feedback = f"""⚠️ INTEGRATION ERROR - Please fix this issue:

Issue Type: {ticket["issue_type"]}
Error Message: {ticket["message"]}
Suggestion: {ticket.get("suggestion", "Review and fix the code")}

Please carefully review the error and fix the implementation.
"""

            # Create task wrapper
            task = TaskWrapper(file_spec, priority=0)
            task.last_error = feedback
            self.tasks[file_path] = task

        # Execute fixes (with is_fix_mode=True)
        results = await self.execute_tasks(
            tasks=[
                self.tasks[t["file_path"]].task
                for t in tickets
                if t["file_path"] in self.tasks
            ],
            blueprint=blueprint,
            is_fix_mode=True,
        )

        # Mark step status only (phase stays REFINEMENT)
        self.complete_execution(StepStatus.COMPLETED)

        return {
            task_id: wrapper.status.value for task_id, wrapper in self.tasks.items()
        }

    async def fix_blueprint(
        self,
        fix_blueprint: FixBlueprint,
        blueprint: Blueprint,
    ) -> Dict[str, str]:
        """
        Fix files based on a FixBlueprint (file-level tasks + DAG).

        The task granularity is file-level, aligned with implementation tasks.
        """
        if not fix_blueprint.tasks:
            return {}

        self.blueprint = blueprint

        # Reload state to pick up any new step created by entry.py
        if self.state_manager:
            self.state_manager.load()

        # Resume semantics for fix loop:
        # Only skip tasks that execution state marked as completed (NOT based on "file exists").
        completed_ids: Optional[Set[str]] = None
        if self.state_manager and self.state_manager.current_state:
            completed_set: Set[str] = self.get_completed_task_ids_from_state()
            if completed_set:
                completed_ids = completed_set
                self._log_info(
                    f"FixBlueprint resume: skipping {len(completed_ids)} completed tasks from execution state"
                )

        print(f"\n{'='*60}")
        print(
            f"CodeManager: Fixing {len(fix_blueprint.tasks)} files from FixBlueprint..."
        )
        print(f"{'='*60}")

        # Reset tasks for files that need fixing
        tasks_to_run: List[FileSpec] = []
        for task in fix_blueprint.tasks:
            file_path = task.file_path
            file_spec = next(
                (f for f in blueprint.files if f.file_path == file_path), None
            )

            if not file_spec:
                print(f"  ⚠️ No spec found for {file_path}, skipping...")
                continue

            issues_text = "\n".join(
                [
                    f"- {i.issue_type}: {i.message}\n  Suggestion: {i.suggestion}"
                    for i in task.issues
                ]
            )

            feedback = f"""⚠️ FIX TASK - Please fix these issues:

Task: {task.title}
File: {task.file_path}

Issues:
{issues_text}

Acceptance:
- Fix the root cause (do not change tests just to make them pass).
- Keep code complete (no TODO/placeholder).
"""

            wrapper = TaskWrapper(file_spec, priority=0)
            wrapper.last_error = feedback
            self.tasks[file_path] = wrapper
            tasks_to_run.append(file_spec)

        # Execute fixes (with is_fix_mode=True)
        results = await self.execute_tasks(
            tasks=tasks_to_run,
            resume=True if completed_ids is not None else False,
            completed_ids=completed_ids,
            blueprint=blueprint,
            is_fix_mode=True,
        )

        # Mark step status only (phase stays REFINEMENT)
        self.complete_execution(StepStatus.COMPLETED)

        return {
            task_id: wrapper.status.value for task_id, wrapper in self.tasks.items()
        }

    def _build_system_prompt(self, **kwargs) -> str:
        return ""  # Manager doesn't use LLM directly

    def _build_user_prompt(self, **kwargs) -> str:
        return ""  # Manager doesn't use LLM directly
