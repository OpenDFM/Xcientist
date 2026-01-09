from typing import List

from src.agents.paper_agent.prompts import render_template
from src.agents.paper_agent.tools.checks import check_citations
from src.agents.paper_agent.tools.compile import compile_and_vlm_review
from src.agents.paper_agent.tools.core import bash, file_viewer, write_file
from src.agents.paper_agent.utils.agent_base import BaseAgent


class PaperReviewerAgent(BaseAgent):
    def __init__(
        self, model: str = "gpt-5.2", max_turns: int = 80, verbose: bool = True
    ):
        super().__init__(
            agent_type="PaperReviewer",
            model=model,
            max_turns=max_turns,
            verbose=verbose,
        )

    def _build_system_prompt(self, **kwargs) -> str:
        paper_dir = str(kwargs.get("paper_dir", "") or "")
        artifact_dir = str(kwargs.get("artifact_dir", "") or "")
        output_path = str(kwargs.get("output_path", "") or "")

        return render_template(
            "reviewer_system.j2",
            paper_dir=paper_dir,
            artifact_dir=artifact_dir,
            output_path=output_path,
        )

    def _build_user_prompt(self, **kwargs) -> str:
        return render_template("reviewer_user.j2")

    def _get_tools(self) -> List:
        return [bash, file_viewer, write_file, compile_and_vlm_review, check_citations]
