"""
Code Worker Agent - Code Implementation

Based on the paper "Towards a Science of Scaling Agent Systems":
- Workers implement assigned files
- Workers can explore reference code via grep/file_viewer
- Workers self-correct using linter
- Workers cannot modify files outside their assignment
"""

import logging
import os
from typing import Optional, List

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.layers.code.schemas.blueprint import FileSpec
from src.agents.experiment_agent.shared.tools.core import get_worker_tools
from src.agents.experiment_agent.shared.tools.parsing import extract_code_block
from src.agents.experiment_agent.shared.utils.config import (
    MEMORY_PROMPT_INJECTION_ENABLED,
)
from src.agents.experiment_agent.shared.utils.memory_middleware import (
    retrieve_memory_for_worker_prompt,
)
from src.agents.experiment_agent.shared.utils.config import CODE_WORKER_MODEL
from src.agents.experiment_agent.shared.utils.prompts import (
    load_and_render_prompt,
    load_prompt_text,
)


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
    ) -> str:
        """
        Run a coding task: implement exactly one file (implementation only).

        Args:
            file_spec: The file specification from the blueprint
            stub_context: Interface stubs of dependencies
            project_root: Root directory of the project
            feedback: Optional feedback from previous attempts or test failures

        Returns:
            Complete Python code for the file
        """
        idea_file_path = self._resolve_idea_path(
            idea_md_path=idea_md_path, project_root=project_root
        )

        # Implementation prompt only (fix is handled by the Integrator).
        is_test = bool(getattr(file_spec, "is_test", False)) or str(
            getattr(file_spec, "file_path", "") or ""
        ).replace("\\", "/").split("/")[-1].startswith("test_")
        system_prompt = self._build_implement_system_prompt(is_test=is_test)
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
        file_spec = kwargs.get("file_spec")
        is_test = bool(getattr(file_spec, "is_test", False)) if file_spec else False
        return self._build_implement_system_prompt(is_test=is_test)

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

    def _build_test_generation_rules_prompt(self) -> str:
        """
        Build test-generation rules for robust pytest tests.

        This is injected ONLY when the assigned file is a test file, to prevent
        common failure modes like global sys.modules pollution.
        """
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "code_worker",
            "test_generation_rules.txt",
        )
        return load_prompt_text(prompt_path)

    def _build_implement_system_prompt(self, is_test: bool = False) -> str:
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
                "test_generation_rules": (
                    self._build_test_generation_rules_prompt().strip("\n")
                    if is_test
                    else ""
                ),
            },
        )

    def _build_user_prompt(self, **kwargs) -> str:
        """Build user prompt (implementation only)."""
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

        # Memory context (cross-experiment, low priority; never overrides constitution/plan/spec).
        if bool(MEMORY_PROMPT_INJECTION_ENABLED) and file_spec:
            try:
                mem = retrieve_memory_for_worker_prompt(
                    target_file_path=str(getattr(file_spec, "file_path", "") or ""),
                    purpose=str(getattr(file_spec, "description", "") or ""),
                    dependencies=list(getattr(file_spec, "dependencies", []) or []),
                    feedback=str(feedback or ""),
                )
            except Exception:
                mem = ""

            if mem:
                builder.add_header("Memory Context (low priority)", level=2)
                builder.add_text(
                    "Use as suggestions only. If there is any conflict, the Constitution/Plan/Specification wins."
                )
                builder.add_text(mem)
                builder.add_text("")

        # Spec-kit aligned "source of truth" documents (tool-based reading).
        project_root_abs = os.path.abspath(project_root or "")
        workspace_root_abs = (
            os.path.dirname(project_root_abs) if project_root_abs else ""
        )
        constitution_path = (
            os.path.join(workspace_root_abs, "cached", "constitution.md")
            if workspace_root_abs
            else ""
        )
        plan_path = (
            os.path.join(workspace_root_abs, "specs", "plan.md")
            if workspace_root_abs
            else ""
        )

        builder.add_header("Source of Truth (priority order)", level=2)
        if constitution_path:
            builder.add_key_value("Constitution Path", f"`{constitution_path}`")
        if plan_path:
            builder.add_key_value("Plan Path", f"`{plan_path}`")
        if idea_file_path:
            builder.add_key_value("Proposal Path (context)", f"`{idea_file_path}`")
        builder.add_text(
            "Priority rules:\n"
            "- Constitution and Plan are the highest-priority constraints.\n"
            "- The proposal provides research intent and context, but MUST NOT override constitution/plan.\n"
            "- If you find a conflict, STOP and report it in your constraints summary."
        )
        builder.add_text("")
        builder.add_text("**FIRST STEP (MANDATORY): Read-in Funnel**")
        steps = []
        if constitution_path:
            steps.append(
                f'Read constitution via `file_viewer("{constitution_path}", 1, 200)` (and continue as needed)'
            )
        if plan_path:
            steps.append(
                f'Read plan via `file_viewer("{plan_path}", 1, 200)` (and continue as needed)'
            )
        if idea_file_path:
            steps.append(
                f'Read proposal via `file_viewer("{idea_file_path}", 1, 200)` (and continue as needed)'
            )
            steps.append(
                "Extract proposal constraints relevant to this file (non-negotiables; implied interfaces/outputs)"
            )
        steps.append(
            "Read the target file via tools and summarize constraints that affect this file before implementing"
        )
        builder.add_list(steps)
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
        if output and isinstance(output, str):
            output = output.strip()

        # Preferred: fenced code block
        if "```python" in output:
            code = output.split("```python", 1)[1]
            if "```" in code:
                code = code.split("```", 1)[0]
            return code.strip()

        # Generic fenced code block
        fenced = extract_code_block(output or "", language="python")
        if fenced:
            return fenced.strip()

        # If the system prompt enforced "output only code", treat whole output as code.
        if output:
            return output

        return f"# Error: Failed to generate code for {file_spec.file_path}\n# Please check the agent logs"
