"""
Code Integrator Agent - Integration & Verification

Based on the paper "Towards a Science of Scaling Agent Systems":
- Returns to single-agent mode for global state verification
- Has full view of the codebase for integration testing
- Can generate and run integration tests
- Reports bugs back to the Manager for fixing
"""

import os
import logging
from typing import Dict, List, Optional

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.layers.base.state import (
    GlobalPhase,
    StateManager,
    StepStatus,
)
from src.agents.experiment_agent.shared.tools.core import (
    SecurityContext,
    get_integrator_tools,
    get_worker_tools,
)
from src.agents.experiment_agent.shared.utils.config import CODE_INTEGRATOR_MODEL
from src.agents.experiment_agent.shared.utils.prompts import load_prompt_text


logger = logging.getLogger(__name__)


class CodeIntegratorAgent(BaseAgent):
    """
    Integration and Verification Agent.

    Minimal verification policy:
    - Run ALL tests under `tests/` using pytest.
    - If tests fail, the integrator edits code and re-runs the full test suite until green.
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
        self.workspace_root = os.path.dirname(self.project_root)
        self.state_manager = StateManager(self.workspace_root, namespace="code")
        SecurityContext.set_roots(
            project_root=self.project_root, workspace_root=self.workspace_root
        )

    def _get_tools(self) -> List:
        return get_integrator_tools()

    def _build_system_prompt(self, **kwargs) -> str:
        _ = kwargs
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "code_integrator",
            "system_fix_executor.txt",
        )
        return load_prompt_text(prompt_path)

    def _build_user_prompt(self, **kwargs) -> str:
        entry_point = str(kwargs.get("entry_point", "") or "")
        extra_context = str(kwargs.get("extra_context", "") or "")
        return self._build_fix_executor_user_prompt(
            entry_point=entry_point, extra_context=extra_context
        )

    async def fix_until_tests_pass(
        self,
        entry_point: str,
        tickets: Optional[List[Dict]] = None,
    ) -> bool:
        """
        Single-agent fix loop (fully internalized):
        - The LLM agent is instructed to run ALL tests under `tests/` (pytest) itself.
        - It must iterate: run tests -> inspect failures -> edit code -> rerun tests,
          until all tests pass.

        Returns True if the agent reports success, else False.
        """
        extra_context = self._format_optimization_tickets(tickets or [])
        stage = "OPTIMIZATION" if (tickets or []) else "INTEGRATION_FIX"
        self._checkpoint_start(
            stage=stage, entry_point=entry_point, tickets=tickets or []
        )
        result = await self._run_fix_executor(
            entry_point=entry_point, extra_context=extra_context
        )
        output = self._extract_output(result)
        ok = "ALL TESTS PASSED" in (output or "")
        self._checkpoint_end(ok=ok)
        return ok

    def _checkpoint_start(
        self, stage: str, entry_point: str, tickets: List[Dict]
    ) -> None:
        """
        Breakpoint/resume support:
        - Mark the latest code step as REFINEMENT/RUNNING with meta that indicates we are in integrator fix.
        - This allows `--resume` to skip implementation and jump back into this stage after a crash.
        """
        try:
            if self.state_manager.load() and self.state_manager.current_state:
                meta = dict(self.state_manager.current_state.meta or {})
                meta.update(
                    {
                        "stage": str(stage or "INTEGRATION_FIX"),
                        "entry_point": str(entry_point or ""),
                        "tickets_count": int(len(tickets or [])),
                    }
                )
                self.state_manager.set_phase(GlobalPhase.REFINEMENT, meta=meta)
                self.state_manager.set_status(StepStatus.RUNNING, meta=meta)
        except Exception:
            pass

    def _checkpoint_end(self, ok: bool) -> None:
        try:
            if self.state_manager.load() and self.state_manager.current_state:
                meta = dict(self.state_manager.current_state.meta or {})
                meta["stage_done"] = True

                if ok:
                    self.state_manager.set_phase(GlobalPhase.VERIFICATION, meta=meta)
                self.state_manager.set_status(
                    StepStatus.COMPLETED if ok else StepStatus.FAILED, meta=meta
                )
        except Exception:
            pass

    def _format_optimization_tickets(self, tickets: List[Dict]) -> str:
        lines: List[str] = []
        for i, t in enumerate(tickets or [], 1):
            if not isinstance(t, dict):
                continue
            file_path = str(t.get("file_path", "") or "")
            issue_type = str(t.get("issue_type", "") or "")
            msg = str(t.get("message", "") or "")
            suggestion = str(t.get("suggestion", "") or "")
            lines.append(f"[{i}] file={file_path} type={issue_type}")
            if msg:
                lines.append(f"  message: {msg}")
            if suggestion:
                lines.append(f"  suggestion: {suggestion}")
        return "\n".join(lines).strip()

    async def _run_fix_executor(self, entry_point: str, extra_context: str = ""):
        """
        Run an LLM-powered fix executor that can edit project files directly,
        and is responsible for looping until `pytest -q` passes.
        """
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "code_integrator",
            "system_fix_executor.txt",
        )
        system_prompt = load_prompt_text(prompt_path)
        user_prompt = self._build_fix_executor_user_prompt(
            entry_point=entry_point, extra_context=extra_context
        )

        return await self._run_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=get_worker_tools(),
        )

    def _build_fix_executor_user_prompt(
        self, entry_point: str, extra_context: str = ""
    ) -> str:
        builder = PromptBuilder()
        builder.add_header("Integration Fix Executor")
        builder.add_key_value("Project Root", self.project_root)
        builder.add_key_value("Workspace Root", self.workspace_root)
        builder.add_key_value("Entry Point", str(entry_point or ""))
        builder.add_text("")

        # Spec-coding "Source of Truth" documents (read-in funnel like CodeWorker).
        constitution_path = os.path.join(
            self.workspace_root, "cached", "constitution.md"
        )
        plan_path = os.path.join(self.workspace_root, "specs", "plan.md")
        idea_md = os.path.join(self.workspace_root, "idea.md")
        idea_json = os.path.join(self.workspace_root, "idea.json")

        builder.add_header("Source of Truth (read-in funnel; MUST read first)", level=2)
        builder.add_text(
            "Before running tests or editing code, read these in order (if they exist):"
        )
        builder.add_list(
            [
                f"`{constitution_path}`",
                f"`{plan_path}`",
                f"`{idea_md}` (optional context)",
                f"`{idea_json}` (optional context)",
            ],
            ordered=True,
        )
        builder.add_text(
            "Hard rule: your FIRST tool calls must open the Constitution and Plan via `file_viewer` (when present)."
        )
        builder.add_text("")

        if extra_context:
            builder.add_header("Additional Context (tickets / intent)", level=2)
            builder.add_text(str(extra_context))
            builder.add_text("")

        builder.add_separator()
        builder.add_header("Your Task", level=2)
        builder.add_list(
            [
                "Run ALL tests under `tests/` by executing `pytest -q` from the PROJECT ROOT.",
                "You MUST set `working_dir` to the provided Project Root when calling `bash(...)` (do not run from workspace root).",
                "If tests fail, inspect the pytest output, identify the root cause, and fix the code by editing existing project files.",
                "Repeat until `pytest -q` passes.",
                "Make the smallest set of changes needed; do NOT change tests just to make them pass.",
                "ABSOLUTE RULE: Do NOT modify any file under `tests/`.",
                "Prefer `edit_file` for small diffs; use `write_file` only if necessary.",
                "Do NOT create new files unless absolutely necessary.",
            ],
            ordered=True,
        )
        builder.add_text(
            'When ALL tests pass, output the final line exactly: "ALL TESTS PASSED".'
        )
        return builder.build()
