"""
Code Integrator Agent - Integration & Verification

Based on the paper "Towards a Science of Scaling Agent Systems":
- Returns to single-agent mode for global state verification
- Has full view of the codebase for integration testing
- Can generate and run integration tests
- Reports bugs back to the Manager for fixing
"""

import os
import re
import ast
import subprocess
import logging
from typing import Dict, List, Optional

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.layers.code.schemas.fix_blueprint import FixBlueprint, FixIssueSpec, FixTaskSpec
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint
from src.agents.experiment_agent.layers.code.schemas.integration import (
    StackFrame,
    TestFailureInfo,
    IntegrationIssue,
)
from src.agents.experiment_agent.layers.code.traceback_parser import TracebackParser
from src.agents.experiment_agent.shared.tools.core import get_integrator_tools
from src.agents.experiment_agent.shared.utils.config import CODE_INTEGRATOR_MODEL
from src.agents.experiment_agent.shared.tools.parsing import parse_to_model
from src.agents.experiment_agent.shared.utils.prompts import load_prompt_text


logger = logging.getLogger(__name__)


class CodeIntegratorAgent(BaseAgent):
    """
    Integration and Verification Agent.

    Verifies project integrity through:
    - Syntax checking
    - Import validation

    - Entry point testing
    - Unit test execution
    """

    def __init__(
        self,
        project_root: str,
        model: str = CODE_INTEGRATOR_MODEL,
        verbose: bool = True,
    ):
        super().__init__(
            agent_type="CodeIntegrator",
            model=model,
            max_turns=10000,
            verbose=verbose,
        )

        self.project_root = os.path.abspath(project_root)
        self.issues: List[IntegrationIssue] = []
        self.test_failures: List[TestFailureInfo] = []
        self.last_pytest_output: str = ""
        self.last_pytest_returncode: Optional[int] = None

        # Traceback parser for extracting structured failure info
        self.parser = TracebackParser(self.project_root)

    def _get_tools(self) -> List:
        return get_integrator_tools()

    async def verify_project(self, entry_point: str) -> bool:
        """
        Perform comprehensive project verification.

        Args:
            entry_point: Path to the entry point file

        Returns:
            True if verification passes, False otherwise
        """
        self._log_info("Starting project verification...")
        self.issues = []
        self.test_failures = []

        # Phase 1: Syntax Check
        print("Integrator: [1/4] Running syntax checks...")
        syntax_ok = self._check_all_syntax()

        if not syntax_ok:
            print("Integrator: ❌ Syntax errors found. Aborting further checks.")
            self._print_issues()
            return False
        print("Integrator: ✅ Syntax check passed.")

        # Phase 3: Entry Point Check
        print("Integrator: [2/4] Testing entry point...")
        entry_ok = self._test_entry_point(entry_point)

        if not entry_ok:
            print(f"Integrator: ❌ Entry point '{entry_point}' test failed.")
            self._print_issues()
            return False
        print("Integrator: ✅ Entry point test passed.")

        # Phase 4: Compile All
        print("Integrator: [3/4] Compiling all Python files...")
        compile_ok = self._compile_all()

        if not compile_ok:
            print("Integrator: ❌ Compilation errors found.")
            self._print_issues()
            return False
        print("Integrator: ✅ Compilation passed.")

        # Phase 5: Run Unit Tests
        print("Integrator: [4/4] Running unit tests...")
        tests_ok = self._run_unit_tests()

        if not tests_ok:
            print("Integrator: ❌ Unit tests failed.")
        else:
            print("Integrator: ✅ Unit tests passed.")

        # Final report
        self._print_issues()

        error_count = sum(1 for i in self.issues if i.severity == "error")
        warning_count = sum(1 for i in self.issues if i.severity == "warning")

        print(f"\n{'='*60}")
        print("Integrator: Verification Complete.")
        print(f"  - Errors: {error_count}")
        print(f"  - Warnings: {warning_count}")
        print(f"{'='*60}")

        return error_count == 0

    def _check_all_syntax(self) -> bool:
        """Check syntax of all Python files."""
        all_ok = True

        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [
                d
                for d in dirs
                if d not in ["__pycache__", ".git", "venv", ".venv", ".cache"]
            ]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        ast.parse(content)
                    except SyntaxError as e:
                        all_ok = False
                        self.issues.append(
                            IntegrationIssue(
                                severity="error",
                                file_path=rel_path,
                                issue_type="SyntaxError",
                                message=f"Line {e.lineno}: {e.msg}",
                                suggestion="Fix the syntax error",
                            )
                        )
                    except Exception as e:
                        all_ok = False
                        self.issues.append(
                            IntegrationIssue(
                                severity="error",
                                file_path=rel_path,
                                issue_type="ParseError",
                                message=str(e),
                            )
                        )

        return all_ok

    def _collect_project_modules(self) -> set:
        """Collect all module names available in the project."""
        project_modules = set()

        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [
                d
                for d in dirs
                if d not in ["__pycache__", ".git", "venv", ".venv", ".cache"]
            ]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    module_name = rel_path.replace("/", ".").replace("\\", ".")[:-3]
                    project_modules.add(module_name)

                    # Add parent modules
                    parts = module_name.split(".")
                    for i in range(len(parts)):
                        project_modules.add(".".join(parts[: i + 1]))

        return project_modules

    def _is_stdlib_or_external(self, module_name: str) -> bool:
        """Check if module is standard library or known external package."""
        stdlib = {
            "os",
            "sys",
            "re",
            "json",
            "ast",
            "typing",
            "pathlib",
            "collections",
            "functools",
            "itertools",
            "datetime",
            "time",
            "math",
            "random",
            "subprocess",
            "threading",
            "asyncio",
            "abc",
            "dataclasses",
            "enum",
            "copy",
            "io",
            "logging",
            "unittest",
            "argparse",
            "configparser",
            "hashlib",
            "base64",
            "pickle",
            "csv",
            "xml",
            "html",
            "urllib",
            "http",
            "socket",
            "ssl",
            "email",
            "mimetypes",
            "struct",
            "codecs",
            "textwrap",
            "difflib",
            "contextlib",
            "warnings",
            "traceback",
        }

        external = {
            "numpy",
            "torch",
            "tensorflow",
            "pandas",
            "scipy",
            "sklearn",
            "matplotlib",
            "seaborn",
            "requests",
            "flask",
            "django",
            "fastapi",
            "pydantic",
            "pytest",
            "openai",
            "transformers",
            "tqdm",
            "yaml",
            "PIL",
            "cv2",
            "jax",
            "optax",
            "flax",
            "einops",
            "wandb",
            "mlflow",
            "agents",
            "httpx",
            "aiohttp",
            "rich",
            "click",
            "typer",
        }

        first_part = module_name.split(".")[0]
        return first_part in stdlib or first_part in external

    def _test_entry_point(self, entry_point: str) -> bool:
        """Test that the entry point can be imported without errors."""
        entry_path = os.path.join(self.project_root, entry_point)

        if not os.path.exists(entry_path):
            self.issues.append(
                IntegrationIssue(
                    severity="error",
                    file_path=entry_point,
                    issue_type="FileNotFound",
                    message=f"Entry point file not found: {entry_point}",
                )
            )
            return False

        try:
            result = subprocess.run(
                f"python -c \"import ast; ast.parse(open('{entry_path}').read())\"",
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception as e:
            self.issues.append(
                IntegrationIssue(
                    severity="error",
                    file_path=entry_point,
                    issue_type="EntryPointError",
                    message=str(e),
                )
            )
            return False

    def _compile_all(self) -> bool:
        """Run compileall on the project."""
        try:
            result = subprocess.run(
                f"python -m compileall -q {self.project_root}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                self.issues.append(
                    IntegrationIssue(
                        severity="error",
                        file_path="project",
                        issue_type="CompileError",
                        message=result.stderr or "Compilation failed",
                    )
                )
                return False

            return True

        except Exception as e:
            self.issues.append(
                IntegrationIssue(
                    severity="error",
                    file_path="project",
                    issue_type="CompileError",
                    message=str(e),
                )
            )
            return False

    def _run_unit_tests(self) -> bool:
        """Run unit tests using pytest, executing each test file individually."""
        tests_dir = os.path.join(self.project_root, "tests")
        if not os.path.isdir(tests_dir):
            print("Integrator: ⚠️ No 'tests/' directory found. Skipping test phase.")
            return True

        # Collect all test files
        test_files = self._collect_test_files(tests_dir)
        if not test_files:
            print("Integrator: ⚠️ No test files found in 'tests/'. Skipping test phase.")
            return True

        print(
            f"Integrator: Found {len(test_files)} test file(s). Running individually..."
        )

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            self.project_root
            if not existing_pythonpath
            else f"{self.project_root}{os.pathsep}{existing_pythonpath}"
        )

        all_passed = True
        all_outputs = []

        for test_file in test_files:
            rel_path = os.path.relpath(test_file, self.project_root)
            print(f"Integrator: Running {rel_path}...")

            result = self._run_single_test_file(test_file, env)
            all_outputs.append(f"\n{'='*60}\n{rel_path}\n{'='*60}\n{result['output']}")

            if not result["passed"]:
                all_passed = False
                # Use TracebackParser to parse failures
                failures = self.parser.parse_test_output(rel_path, result["output"])
                self.test_failures.extend(failures)

                # Create IntegrationIssue for each failure
                for failure in failures:
                    for impl_file in failure.impl_files:
                        stack_info = ""
                        project_frames = [
                            f for f in failure.call_stack if f.is_project_file
                        ]
                        if project_frames:
                            stack_lines = [
                                f"  {f.file_path}:{f.line_num} in {f.func_name}"
                                for f in project_frames[:5]
                            ]
                            stack_info = "\nCall Stack:\n" + "\n".join(stack_lines)

                        self.issues.append(
                            IntegrationIssue(
                                severity="error",
                                file_path=impl_file,
                                issue_type="TestFailure",
                                message=f"[{failure.test_name}] {failure.error_type}: {failure.root_cause[:200]}",
                                suggestion=f"Test: {failure.test_file}::{failure.test_name}\n"
                                f"Root Cause: {failure.root_cause}"
                                f"{stack_info}",
                            )
                        )

                print(f"Integrator: ❌ {rel_path} FAILED ({len(failures)} failure(s))")
            else:
                print(f"Integrator: ✅ {rel_path} PASSED")

        self.last_pytest_output = "\n".join(all_outputs)
        self.last_pytest_returncode = 0 if all_passed else 1

        if not all_passed:
            print(
                f"\nIntegrator: ❌ {len(self.test_failures)} test failure(s) in total."
            )

        return all_passed

    def _collect_test_files(self, tests_dir: str) -> List[str]:
        """Collect all test files in the tests directory."""
        test_files = []
        for root, dirs, files in os.walk(tests_dir):
            dirs[:] = [d for d in dirs if d not in ["__pycache__", ".pytest_cache"]]
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_files.append(os.path.join(root, file))
        return sorted(test_files)

    def _run_single_test_file(self, test_file: str, env: dict) -> Dict:
        """Run a single test file with pytest."""
        try:
            result = subprocess.run(
                ["pytest", "-v", "--tb=long", "--no-header", test_file],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.project_root,
                env=env,
            )
            return {
                "passed": result.returncode == 0,
                "output": result.stdout + "\n" + result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "output": f"Test file timed out after 120 seconds: {test_file}",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "passed": False,
                "output": f"Error running test: {str(e)}",
                "returncode": -1,
            }

    def _print_issues(self):
        """Print all found issues."""
        if not self.issues:
            return

        print(f"\n{'─'*40}")
        print("Integration Issues:")
        for issue in self.issues:
            icon = "❌" if issue.severity == "error" else "⚠️"
            print(f"  {icon} {issue}")
        print(f"{'─'*40}")

    def get_issues(self) -> List[IntegrationIssue]:
        """Get list of all integration issues found."""
        return self.issues.copy()

    def get_fix_tickets(self) -> List[Dict]:
        """Generate fix tickets for Manager to dispatch to Workers."""
        return [
            {
                "file_path": issue.file_path,
                "issue_type": issue.issue_type,
                "message": issue.message,
                "suggestion": issue.suggestion,
            }
            for issue in self.issues
            if issue.severity == "error"
        ]

    def build_fix_blueprint(
        self,
        entry_point: str,
        blueprint: Optional[Blueprint] = None,
    ) -> FixBlueprint:
        """
        Build a file-level FixBlueprint from current integration issues.

        Notes:
        - Task granularity is file-level (aligned with implementation tasks).
        - Uses detailed test failure information when available.
        - dependency_graph is derived from Blueprint dependencies.
        """
        # Aggregate issues by file
        issues_by_file: Dict[str, List[IntegrationIssue]] = {}
        for issue in self.issues:
            if issue.severity != "error":
                continue
            issues_by_file.setdefault(issue.file_path, []).append(issue)

        # Aggregate detailed test failures by implementation file
        failures_by_impl_file: Dict[str, List[TestFailureInfo]] = {}
        for failure in self.test_failures:
            for impl_file in failure.impl_files:
                failures_by_impl_file.setdefault(impl_file, []).append(failure)

        # Merge all files that need fixing
        all_files = set(issues_by_file.keys()) | set(failures_by_impl_file.keys())

        # Build dependency map from Blueprint
        file_deps: Dict[str, List[str]] = {}
        if blueprint:
            blueprint_deps = {f.file_path: f.dependencies for f in blueprint.files}
            for file_path in all_files:
                deps = blueprint_deps.get(file_path, [])
                relevant_deps = [d for d in deps if d in all_files]
                file_deps[file_path] = relevant_deps

        tasks: List[FixTaskSpec] = []
        dependency_graph: Dict[str, List[str]] = {}

        for file_path in sorted(all_files):
            file_issues = issues_by_file.get(file_path, [])
            file_failures = failures_by_impl_file.get(file_path, [])

            # Build issue specs
            issue_specs = []
            for i in file_issues:
                issue_specs.append(
                    FixIssueSpec(
                        severity=i.severity,
                        issue_type=i.issue_type,
                        message=i.message,
                        suggestion=i.suggestion,
                    )
                )

            # Add detailed info from test failures
            for failure in file_failures:
                stack_info = ""
                project_frames = [f for f in failure.call_stack if f.is_project_file]
                if project_frames:
                    stack_lines = [
                        f"  {f.file_path}:{f.line_num} in {f.func_name}"
                        for f in project_frames[:5]
                    ]
                    stack_info = "\nCall Stack:\n" + "\n".join(stack_lines)

                issue_specs.append(
                    FixIssueSpec(
                        severity="error",
                        issue_type=f"TestFailure:{failure.error_type}",
                        message=f"[{failure.test_name}] {failure.root_cause[:300]}",
                        suggestion=f"Test: {failure.test_file}::{failure.test_name}\n"
                        f"Root Cause: {failure.root_cause}"
                        f"{stack_info}",
                    )
                )

            # Build description
            title = f"Fix {file_path}"
            description_lines = [f"Fix issues in `{file_path}`."]

            if file_issues:
                description_lines.append("\n**Integration Errors:**")
                for i in file_issues:
                    description_lines.append(f"- {i.issue_type}: {i.message[:150]}")

            if file_failures:
                description_lines.append("\n**Test Failures:**")
                for f in file_failures:
                    description_lines.append(
                        f"- `{f.test_file}::{f.test_name}` - {f.error_type}: {f.root_cause[:100]}"
                    )
                    project_frames = [
                        frame for frame in f.call_stack if frame.is_project_file
                    ]
                    if project_frames:
                        description_lines.append("  Call Stack:")
                        for frame in project_frames[:3]:
                            code_snippet = (
                                f" > {frame.code_line.strip()}"
                                if frame.code_line
                                else ""
                            )
                            description_lines.append(
                                f"    - {frame.file_path}:{frame.line_num} in {frame.func_name}{code_snippet}"
                            )

            task_deps = file_deps.get(file_path, [])

            task = FixTaskSpec(
                task_id=file_path,
                file_path=file_path,
                title=title,
                description="\n".join(description_lines),
                issues=issue_specs,
                dependencies=task_deps,
            )
            tasks.append(task)
            dependency_graph[task.task_id] = task_deps

        notes = "Tasks are file-level and derived from Integrator issues."
        if self.test_failures:
            notes += f" {len(self.test_failures)} test failure(s) detected."

        return FixBlueprint(
            entry_point=entry_point,
            trigger="integration_verify",
            tasks=tasks,
            dependency_graph=dependency_graph,
            notes=notes,
        )

    async def generate_fix_blueprint(
        self,
        blueprint: Blueprint,
        entry_point: str,
    ) -> FixBlueprint:
        """
        Use an LLM agent to synthesize a FixBlueprint from verification results.

        Requirements:
        - File-level tasks (align with Architect/Blueprint file granularity).
        - Only reference existing files in the code blueprint.
        - Output must be valid FixBlueprint JSON.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            blueprint=blueprint,
            entry_point=entry_point,
        )

        result = await self._run_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=self._get_tools(),
        )

        output = self._extract_output(result)
        parsed = parse_to_model(
            output=output, model_class=FixBlueprint, partial_ok=False
        )
        if parsed is None:
            return self.build_fix_blueprint(
                entry_point=entry_point, blueprint=blueprint
            )

        # Hard validation: restrict to blueprint files only
        allowed_files = {f.file_path for f in blueprint.files}
        filtered_tasks: List[FixTaskSpec] = []
        for t in parsed.tasks:
            if t.file_path in allowed_files:
                filtered_tasks.append(t)

        parsed.tasks = filtered_tasks
        parsed.entry_point = entry_point
        if not parsed.trigger:
            parsed.trigger = "integration_verify"

        parsed.dependency_graph = {
            t.task_id: list(t.dependencies) for t in parsed.tasks
        }
        return parsed

    def _build_system_prompt(self, **kwargs) -> str:
        """Build the system prompt for the LLM-based fix planner."""
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "code_integrator",
            "system_fix_planner.txt",
        )
        return load_prompt_text(prompt_path)

    def _build_user_prompt(
        self,
        blueprint: Optional[Blueprint] = None,
        entry_point: str = "",
        **kwargs,
    ) -> str:
        """Build the user prompt for the LLM-based fix planner."""
        builder = PromptBuilder()
        builder.add_header("Integration Verification Results")

        # Summary section
        error_issues = [i for i in self.issues if i.severity == "error"]
        warning_issues = [i for i in self.issues if i.severity == "warning"]
        builder.add_key_value("Project Root", self.project_root)
        builder.add_key_value("Entry Point", entry_point)
        builder.add_key_value("Errors", str(len(error_issues)))
        builder.add_key_value("Warnings", str(len(warning_issues)))
        builder.add_key_value("Test Failures", str(len(self.test_failures)))
        builder.add_text("")

        # Test failures grouped by impl file
        if self.test_failures:
            builder.add_header("Test Failures by File", level=2)

            failures_by_file: Dict[str, List[TestFailureInfo]] = {}
            for failure in self.test_failures:
                primary_file = (
                    failure.impl_files[0] if failure.impl_files else failure.test_file
                )
                failures_by_file.setdefault(primary_file, []).append(failure)

            for impl_file, file_failures in sorted(failures_by_file.items()):
                builder.add_text(
                    f"\n### `{impl_file}` ({len(file_failures)} failure(s))"
                )

                for i, failure in enumerate(file_failures[:5], 1):
                    builder.add_text(
                        f"\n**[{i}] {failure.test_file}::{failure.test_name}**"
                    )
                    builder.add_text(
                        f"- Root Cause: `{failure.error_type}`: {failure.root_cause}"
                    )

                    if failure.assertion_details:
                        builder.add_text(f"- Assertion: {failure.assertion_details}")

                    # Show call stack - only project files (not external libs)
                    project_frames = [
                        f for f in failure.call_stack if f.is_project_file
                    ]

                    if project_frames:
                        builder.add_text("- Call Stack:")
                        for j, frame in enumerate(project_frames[:5], 1):
                            code_snippet = (
                                f" > {frame.code_line.strip()}"
                                if frame.code_line
                                else ""
                            )
                            builder.add_text(
                                f"  {j}. `{frame.file_path}:{frame.line_num}` in `{frame.func_name}`{code_snippet}"
                            )
                    elif failure.impl_files:
                        # No call stack but have inferred impl files
                        builder.add_text(
                            f"- Likely Files: {', '.join(f'`{f}`' for f in failure.impl_files[:3])}"
                        )
                    else:
                        # Try to extract project paths from raw traceback
                        project_lines = []
                        if failure.raw_traceback:
                            for line in failure.raw_traceback.split("\n"):
                                if (
                                    ".py:" in line
                                    and "site-packages" not in line
                                    and "anaconda" not in line.lower()
                                ):
                                    clean = line.strip()
                                    if clean and not clean.startswith("E "):
                                        project_lines.append(clean[:100])
                        if project_lines:
                            builder.add_text("- Traceback (project files):")
                            for line in project_lines[:3]:
                                builder.add_text(f"  {line}")
                        else:
                            # Infer from test name
                            test_module = failure.test_file.replace(
                                "tests/test_", ""
                            ).replace(".py", "")
                            builder.add_text(
                                f"- Likely Module: `{test_module}` (inferred from test name)"
                            )

                if len(file_failures) > 5:
                    builder.add_text(
                        f"  ... and {len(file_failures) - 5} more failure(s)"
                    )

        # Non-test integration issues
        non_test_issues = [i for i in self.issues if i.issue_type != "TestFailure"]
        if non_test_issues:
            builder.add_header("Other Integration Issues", level=2)
            for issue in non_test_issues[:30]:
                builder.add_text(
                    f"- [{issue.severity.upper()}] `{issue.file_path}`: {issue.issue_type} - {issue.message[:150]}"
                )

        # Available files section
        builder.add_header("Available Files (from Blueprint)", level=2)
        if blueprint is None:
            builder.add_text("Blueprint not provided.")
            return builder.build()

        builder.add_text("You may ONLY create tasks for these files:")
        file_lines: List[str] = []
        for f in blueprint.files:
            deps = ", ".join(f.dependencies[:5]) if f.dependencies else ""
            if deps:
                file_lines.append(f"- `{f.file_path}` (deps: {deps})")
            else:
                file_lines.append(f"- `{f.file_path}`")
        builder.add_text("\n".join(file_lines[:80]))

        # Dependency subgraph for files that need fixing
        files_to_fix = set()
        for failure in self.test_failures:
            files_to_fix.update(failure.impl_files)
        for issue in self.issues:
            if issue.severity == "error" and issue.file_path not in ["project"]:
                files_to_fix.add(issue.file_path)

        if files_to_fix and blueprint:
            builder.add_header("Dependency Order (for files to fix)", level=2)
            builder.add_text(
                "**IMPORTANT:** If file A depends on file B, and both need fixing, "
                "fix B first (add B to A's dependencies in the FixBlueprint)."
            )
            builder.add_text("")

            blueprint_deps = {f.file_path: f.dependencies for f in blueprint.files}
            dep_lines = []
            for file_path in sorted(files_to_fix):
                all_deps = blueprint_deps.get(file_path, [])
                fix_deps = [d for d in all_deps if d in files_to_fix]
                if fix_deps:
                    dep_lines.append(
                        f"- `{file_path}` depends on: {', '.join(f'`{d}`' for d in fix_deps)}"
                    )
                else:
                    dep_lines.append(f"- `{file_path}` (no dependencies in fix set)")

            builder.add_text("\n".join(dep_lines))

        # Your Task section
        builder.add_separator()
        builder.add_header("Your Task", level=2)
        builder.add_list(
            [
                "Analyze the test failures and identify root causes",
                "Group related failures by the file that needs fixing",
                "Set `dependencies` correctly: if A depends on B and both need fixing, A.dependencies = [B]",
                "Create one FixTaskSpec per file that needs changes",
                "Output the FixBlueprint as a JSON object wrapped in ```json ... ```",
            ],
            ordered=True,
        )

        return builder.build()
