import os
from typing import List

from src.agents.paper_agent.prompts import render_template
from src.agents.paper_agent.tools.core import get_architect_tools
from src.agents.paper_agent.utils.agent_base import BaseAgent


class PaperArchitectAgent(BaseAgent):
    def __init__(
        self, model: str = "gpt-5.2", max_turns: int = 999, verbose: bool = True
    ):
        super().__init__(
            agent_type="PaperArchitect",
            model=model,
            max_turns=max_turns,
            verbose=verbose,
        )

    def _build_system_prompt(self, **kwargs) -> str:
        idea_path = str(kwargs.get("idea_path", "") or "")
        project_dir = str(kwargs.get("project_dir", "") or "")
        paper_dir = str(kwargs.get("paper_dir", "") or "")
        specs_dir = str(kwargs.get("specs_dir", "") or "")
        spec_template = str(kwargs.get("spec_template", "") or "")
        plan_template = str(kwargs.get("plan_template", "") or "")
        experiment_id = str(kwargs.get("experiment_id", "") or "")
        experiment_workspace_dir = str(kwargs.get("experiment_workspace_dir", "") or "")
        experiment_specs_dir = str(kwargs.get("experiment_specs_dir", "") or "")
        experiment_spec_files = kwargs.get("experiment_spec_files", None) or []
        experiment_result_files = kwargs.get("experiment_result_files", None) or []

        exp_lines = []
        if experiment_id or experiment_workspace_dir:
            exp_lines.append("Experiment workspace (read-only):")
            if experiment_id:
                exp_lines.append(f"- experiment_id: {experiment_id}")
            if experiment_workspace_dir:
                exp_lines.append(f"- workspace_dir: {experiment_workspace_dir}")
            if experiment_specs_dir:
                exp_lines.append(f"- specs_dir: {experiment_specs_dir}")
            if experiment_spec_files:
                exp_lines.append("- spec files (read):")
                for p in experiment_spec_files:
                    exp_lines.append(f"  - {p}")
            if experiment_result_files:
                exp_lines.append("- result files (read):")
                for p in experiment_result_files:
                    exp_lines.append(f"  - {p}")
            exp_lines.append("")
        exp_block = "\n".join(exp_lines)

        spec_out = (
            os.path.abspath(os.path.join(specs_dir, "spec.md")) if specs_dir else ""
        )
        plan_out = (
            os.path.abspath(os.path.join(specs_dir, "plan.md")) if specs_dir else ""
        )
        constitution_out = (
            os.path.abspath(os.path.join(specs_dir, "constitution.md"))
            if specs_dir
            else ""
        )

        return render_template(
            "architect_system.j2",
            idea_path=idea_path,
            project_dir=project_dir,
            paper_dir=paper_dir,
            specs_dir=specs_dir,
            spec_out=spec_out,
            plan_out=plan_out,
            constitution_out=constitution_out,
            spec_template=spec_template,
            plan_template=plan_template,
            experiment_id=experiment_id,
            experiment_workspace_dir=experiment_workspace_dir,
            experiment_specs_dir=experiment_specs_dir,
            experiment_spec_files=experiment_spec_files,
            experiment_result_files=experiment_result_files,
        )

    def _build_user_prompt(self, **kwargs) -> str:
        specs_dir = str(kwargs.get("specs_dir", "") or "")
        idea_path = str(kwargs.get("idea_path", "") or "")
        project_dir = str(kwargs.get("project_dir", "") or "")
        paper_dir = str(kwargs.get("paper_dir", "") or "")
        experiment_workspace_dir = str(kwargs.get("experiment_workspace_dir", "") or "")

        spec_out = (
            os.path.abspath(os.path.join(specs_dir, "spec.md")) if specs_dir else ""
        )
        plan_out = (
            os.path.abspath(os.path.join(specs_dir, "plan.md")) if specs_dir else ""
        )

        return render_template(
            "architect_user.j2",
            specs_dir=specs_dir,
            spec_out=spec_out,
            plan_out=plan_out,
            idea_path=idea_path,
            project_dir=project_dir,
            paper_dir=paper_dir,
            experiment_workspace_dir=experiment_workspace_dir,
        )

    def _get_tools(self) -> List:
        return get_architect_tools()
