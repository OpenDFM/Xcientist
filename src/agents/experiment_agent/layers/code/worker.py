"""
Code Worker Agent - Code Implementation

Based on the paper "Towards a Science of Scaling Agent Systems":
- Workers implement assigned files
- Workers can explore reference code via grep/file_viewer
- Workers self-correct using linter
- Workers cannot modify files outside their assignment
"""

import os
import logging
from typing import Optional, List

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.layers.code.schemas.blueprint import FileSpec
from src.agents.experiment_agent.shared.tools.core import get_worker_tools
from src.agents.experiment_agent.shared.utils.config import CODE_WORKER_MODEL
from src.agents.experiment_agent.shared.utils.prompts import load_and_render_prompt, load_prompt_text


logger = logging.getLogger(__name__)


class CodeWorkerAgent(BaseAgent):
    """
    Python Engineer Worker Agent.

    Implements single files based on specifications.
    Uses tools for exploration and self-correction.
    """

    def __init__(
        self,
        model: str = CODE_WORKER_MODEL,
        reference_repos: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        super().__init__(
            agent_type="CodeWorker",
            model=model,
            max_turns=10000,
            verbose=verbose,
        )
        self.reference_repos = reference_repos or []

    def add_reference_repo(self, repo_path: str):
        """Add a reference repository."""
        self.reference_repos.append(repo_path)

    def _get_tools(self) -> List:
        return get_worker_tools()

    async def run_task(
        self,
        file_spec: FileSpec,
        stub_context: str,
        project_root: str,
        idea_md_path: str = "",
        feedback: str = "",
        is_fix_mode: bool = False,
    ) -> str:
        """
        Run a coding task: implement a new file or fix an existing one.

        Args:
            file_spec: The file specification from the blueprint
            stub_context: Interface stubs of dependencies
            project_root: Root directory of the project
            feedback: Optional feedback from previous attempts or test failures
            is_fix_mode: If True, enter fix mode to repair existing code.
                        If False (default), implement new code from specification.

        Returns:
            Complete Python code for the file
        """
        idea_file_path = self._resolve_idea_path(
            idea_md_path=idea_md_path, project_root=project_root
        )

        # Read existing content only if this is fix mode
        existing_content = ""
        full_path = os.path.join(project_root, file_spec.file_path)
        if is_fix_mode and os.path.exists(full_path):
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    existing_content = f.read()
            except Exception:
                pass

        # Build prompts based on mode
        if is_fix_mode:
            system_prompt = self._build_fix_system_prompt()
            user_prompt = self._build_fix_user_prompt(
                file_spec=file_spec,
                project_root=project_root,
                idea_file_path=idea_file_path,
                feedback=feedback,
                existing_content=existing_content,
            )
        else:
            system_prompt = self._build_implement_system_prompt()
            user_prompt = self._build_implement_user_prompt(
                file_spec=file_spec,
                stub_context=stub_context,
                project_root=project_root,
                idea_file_path=idea_file_path,
                feedback=feedback,
            )

        # Run agent
        result = await self._run_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=self._get_tools(),
        )

        return result

    def _build_system_prompt(self, **kwargs) -> str:
        """Build the default system prompt (delegates to implement mode)."""
        return self._build_implement_system_prompt()

    def _resolve_idea_path(self, idea_md_path: str, project_root: str) -> str:
        """
        Resolve the research proposal file path (idea.md/idea.json).

        We prefer the explicit path passed from the caller, but fall back to common locations:
        - {project_root}/idea.md
        - {workspace_root}/idea.md (where workspace_root = dirname(project_root))
        and the same for idea.json.
        """
        candidates: List[str] = []
        if idea_md_path:
            candidates.append(str(idea_md_path))

        project_root_abs = os.path.abspath(project_root)
        workspace_root_abs = os.path.dirname(project_root_abs)

        candidates.extend(
            [
                os.path.join(project_root_abs, "idea.md"),
                os.path.join(workspace_root_abs, "idea.md"),
                os.path.join(project_root_abs, "idea.json"),
                os.path.join(workspace_root_abs, "idea.json"),
            ]
        )

        chosen_path = ""
        for p in candidates:
            if p and os.path.exists(p) and os.path.isfile(p):
                chosen_path = p
                break

        if not chosen_path:
            return ""

        return chosen_path

    def _build_tensor_shape_discipline_prompt(self) -> str:
        """
        Build shared, high-signal instructions to prevent and debug PyTorch shape mismatch bugs.

        This is injected into both implement and fix system prompts because tensor shape bugs are
        one of the highest-frequency failure modes for generated code.
        """
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "code_worker",
            "tensor_shape_rules.txt",
        )
        return load_prompt_text(prompt_path)

    def _build_implement_system_prompt(self) -> str:
        """Build the system prompt for implementation mode."""
        reference_repos_str = ""
        if self.reference_repos:
            reference_repos_str = f"""
**REFERENCE REPOSITORIES:**
You have access to the following reference codebases:
{chr(10).join(f'- {repo}' for repo in self.reference_repos)}

Use `bash("grep -rn 'pattern' repo/")` to find code, then `file_viewer` to examine it.
"""
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "code_worker",
            "system_implement.txt",
        )
        return load_and_render_prompt(
            prompt_path=prompt_path,
            variables={
                "reference_repos_str": (
                    reference_repos_str.strip("\n") if reference_repos_str else ""
                ),
                "tensor_shape_rules": self._build_tensor_shape_discipline_prompt().strip(
                    "\n"
                ),
            },
        )

    def _build_fix_system_prompt(self) -> str:
        """Build the system prompt for fix mode."""
        reference_repos_str = ""
        if self.reference_repos:
            reference_repos_str = f"""
**REFERENCE REPOSITORIES:**
You have access to the following reference codebases:
{chr(10).join(f'- {repo}' for repo in self.reference_repos)}

Use `bash("grep -rn 'pattern' repo/")` to find code, then `file_viewer` to examine it.
"""
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "code_worker",
            "system_fix.txt",
        )
        return load_and_render_prompt(
            prompt_path=prompt_path,
            variables={
                "reference_repos_str": (
                    reference_repos_str.strip("\n") if reference_repos_str else ""
                ),
                "tensor_shape_rules": self._build_tensor_shape_discipline_prompt().strip(
                    "\n"
                ),
            },
        )

    def _build_user_prompt(self, **kwargs) -> str:
        """Build user prompt (delegates based on mode)."""
        if kwargs.get("is_fix_mode"):
            return self._build_fix_user_prompt(**kwargs)
        return self._build_implement_user_prompt(**kwargs)

    def _build_implement_user_prompt(
        self,
        file_spec: FileSpec = None,
        stub_context: str = "",
        project_root: str = "",
        idea_file_path: str = "",
        feedback: str = "",
        **kwargs,
    ) -> str:
        """Build the user prompt for implementation mode."""
        builder = PromptBuilder()

        builder.add_header("Implementation Task")
        builder.add_key_value("Target File", f"`{file_spec.file_path}`")
        builder.add_key_value(
            "Full Path", f"`{os.path.join(project_root, file_spec.file_path)}`"
        )
        builder.add_key_value("Purpose", file_spec.description)
        builder.add_text("")

        if idea_file_path:
            builder.add_header("Research Proposal (idea.md/idea.json)", level=2)
            builder.add_key_value("Proposal Path", f"`{idea_file_path}`")
            builder.add_text(
                "You MUST read this proposal FIRST using tools (prefer `file_viewer` with multiple ranges). "
                "Then implement this file consistent with the proposal's Title, Key Innovations, Methodology, "
                "and Expected Outcomes."
            )
            builder.add_text("")
            builder.add_text("**FIRST STEP (MANDATORY):**")
            builder.add_list(
                [
                    f'Read the proposal via `file_viewer("{idea_file_path}", 1, 200)` (and continue as needed)',
                    "Extract: non-negotiable requirements; implied interfaces/outputs; constraints/assumptions",
                ]
            )
            builder.add_text("")

        # Specification
        builder.add_header("Specification", level=2)

        if file_spec.classes:
            builder.add_header("Classes to Implement", level=3)
            for cls in file_spec.classes:
                builder.add_text(f"**`{cls.name}`** - {cls.docstring}")
                if cls.attributes:
                    builder.add_text("Attributes:")
                    for attr, typ in cls.attributes.items():
                        builder.add_text(f"  - `{attr}: {typ}`")
                builder.add_text("Methods:")
                for method in cls.methods:
                    ret = f" -> {method.return_type}" if method.return_type else ""
                    builder.add_text(
                        f"  - `{method.name}({method.args}){ret}` - {method.docstring}"
                    )
                builder.add_text("")

        if file_spec.functions:
            builder.add_header("Functions to Implement", level=3)
            for func in file_spec.functions:
                ret = f" -> {func.return_type}" if func.return_type else ""
                builder.add_text(
                    f"**`{func.name}({func.args}){ret}`** - {func.docstring}"
                )
            builder.add_text("")

        if file_spec.dependencies:
            builder.add_header("Dependencies", level=3)
            builder.add_text("This file imports from:")
            builder.add_list([f"`{dep}`" for dep in file_spec.dependencies])

        # TEST-FIRST: Include test file content if available
        if (
            hasattr(file_spec, "test_file")
            and file_spec.test_file
            and not getattr(file_spec, "is_test", False)
        ):
            test_path = os.path.join(project_root, file_spec.test_file)
            if os.path.exists(test_path):
                try:
                    with open(test_path, "r", encoding="utf-8") as f:
                        test_content = f.read()
                    builder.add_header(
                        "📋 TEST FILE (Your implementation must pass these tests)",
                        level=2,
                    )
                    builder.add_text(
                        "**CRITICAL**: Study these tests carefully! They define the expected behavior, "
                        "input shapes, output shapes, and edge cases your implementation must handle."
                    )
                    builder.add_code(test_content, "python")
                    builder.add_text("")
                    builder.add_text("**EXTRACT FROM TESTS:**")
                    builder.add_list(
                        [
                            "What input shapes are expected?",
                            "What output shapes should be produced?",
                            "What edge cases are tested?",
                            "What exceptions should be raised for invalid inputs?",
                        ]
                    )
                except Exception:
                    pass  # Test file not readable, continue without it

        if stub_context:
            builder.add_header("Dependency Interfaces & Reference Code", level=2)
            builder.add_text(
                "Use these interfaces (already implemented, just import them):"
            )
            builder.add_code(stub_context, "python")

        if feedback:
            builder.add_header("⚠️ Previous Attempt Failed", level=2)
            builder.add_text(feedback)

        builder.add_separator()
        builder.add_header("Your Task", level=2)
        builder.add_list(
            [
                f"Use `write_file` to create `{os.path.join(project_root, file_spec.file_path)}`",
                'Verify syntax with `bash("python -m py_compile ...")`',
                "Fix any issues with `edit_file`",
                'Output "IMPLEMENTATION COMPLETE" when done',
            ],
            ordered=True,
        )

        return builder.build()

    def _build_fix_user_prompt(
        self,
        file_spec: FileSpec = None,
        project_root: str = "",
        idea_file_path: str = "",
        feedback: str = "",
        existing_content: str = "",
        **kwargs,
    ) -> str:
        """Build the user prompt for fix mode."""
        builder = PromptBuilder()

        builder.add_header("🛠️ Fix Task")
        builder.add_key_value("Target File", f"`{file_spec.file_path}`")
        builder.add_key_value(
            "Full Path", f"`{os.path.join(project_root, file_spec.file_path)}`"
        )
        builder.add_key_value("Purpose", file_spec.description)
        builder.add_text("")

        if idea_file_path:
            builder.add_header("Research Proposal (idea.md/idea.json)", level=2)
            builder.add_key_value("Proposal Path", f"`{idea_file_path}`")
            builder.add_text(
                "You MUST read this proposal FIRST using tools and treat it as the top-level constraints while fixing. "
                "Do not change behavior in ways that violate the proposal's requirements."
            )
            builder.add_text("")
            builder.add_text("**FIRST STEP (MANDATORY):**")
            builder.add_list(
                [
                    f'Read the proposal via `file_viewer("{idea_file_path}", 1, 200)` (and continue as needed)',
                    "Summarize the constraints that affect this file before applying edits",
                ]
            )
            builder.add_text("")

        # Error feedback section
        builder.add_header("Error Feedback", level=2)
        builder.add_text("The following errors were detected during testing:")
        builder.add_text("")
        builder.add_text(feedback)
        builder.add_text("")

        # Current file content
        if existing_content:
            builder.add_header("Current File Content", level=2)
            builder.add_text("Here is the current implementation that needs fixing:")
            builder.add_code(existing_content, "python")

        # Test file content if available
        if (
            hasattr(file_spec, "test_file")
            and file_spec.test_file
            and not getattr(file_spec, "is_test", False)
        ):
            test_path = os.path.join(project_root, file_spec.test_file)
            if os.path.exists(test_path):
                try:
                    with open(test_path, "r", encoding="utf-8") as f:
                        test_content = f.read()
                    builder.add_header("Test File (for reference)", level=2)
                    builder.add_text(
                        "Review these tests to understand expected behavior:"
                    )
                    builder.add_code(test_content, "python")
                except Exception:
                    pass

        # Fix instructions
        builder.add_separator()
        builder.add_header("Your Task", level=2)
        builder.add_list(
            [
                "Analyze the error feedback carefully",
                "Identify the root cause of the failure",
                "Use `edit_file` to make targeted fixes (preferred)",
                "Only use `write_file` if major rewrite is needed",
                'Verify syntax with `bash("python -m py_compile ...")`',
                'Output "FIX COMPLETE" when done',
            ],
            ordered=True,
        )

        builder.add_text("")
        builder.add_text("**REMEMBER:** Make minimal changes. Fix only what's broken.")

        return builder.build()

    def _extract_code_from_result(
        self, result, file_spec: FileSpec, project_root: str
    ) -> str:
        """Extract the generated code from agent result."""
        file_path = os.path.join(project_root, file_spec.file_path)

        # Try to read the file that should have been written
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        # Try to extract from output
        output = self._extract_output(result)
        if "```python" in output:
            code = output.split("```python", 1)[1]
            if "```" in code:
                code = code.split("```", 1)[0]
            return code.strip()

        return f"# Error: Failed to generate code for {file_spec.file_path}\n# Please check the agent logs"
