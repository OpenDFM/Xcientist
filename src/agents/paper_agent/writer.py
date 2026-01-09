import os

from typing import List

from src.agents.paper_agent.prompts import render_template
from src.agents.paper_agent.tools.checks import check_citations
from src.agents.paper_agent.tools.core import get_writer_tools
from src.agents.paper_agent.tools.subagents import (
    analyze_results,
    research_literature,
    review_paper,
    visualize_request,
)
from src.agents.paper_agent.utils.agent_base import BaseAgent


class PaperWriterAgent(BaseAgent):
    def __init__(
        self, model: str = "gpt-5.2", max_turns: int = 999, verbose: bool = True
    ):
        super().__init__(
            agent_type="PaperWriter", model=model, max_turns=max_turns, verbose=verbose
        )

    def _build_system_prompt(self, **kwargs) -> str:
        paper_dir = str(kwargs.get("paper_dir", "") or "")
        specs_dir = str(kwargs.get("specs_dir", "") or "")
        idea_path = str(kwargs.get("idea_path", "") or "")
        project_dir = str(kwargs.get("project_dir", "") or "")
        artifact_dir = str(kwargs.get("artifact_dir", "") or "")

        spec_out = (
            os.path.abspath(os.path.join(specs_dir, "spec.md")) if specs_dir else ""
        )
        plan_out = (
            os.path.abspath(os.path.join(specs_dir, "plan.md")) if specs_dir else ""
        )

        return render_template(
            "writer_system.j2",
            paper_dir=paper_dir,
            specs_dir=specs_dir,
            idea_path=idea_path,
            project_dir=project_dir,
            artifact_dir=artifact_dir,
            spec_out=spec_out,
            plan_out=plan_out,
        )

    def _build_user_prompt(self, **kwargs) -> str:
        return render_template("writer_user.j2")

    def _get_tools(self) -> List:
        return get_writer_tools() + [
            analyze_results,
            visualize_request,
            research_literature,
            review_paper,
            check_citations,
        ]
