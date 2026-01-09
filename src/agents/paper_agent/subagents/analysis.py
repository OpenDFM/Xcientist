from typing import List

from src.agents.paper_agent.prompts import render_template
from src.agents.paper_agent.tools.core import bash, edit_file, file_viewer, write_file
from src.agents.paper_agent.utils.agent_base import BaseAgent


class PaperAnalysisAgent(BaseAgent):
    def __init__(
        self, model: str = "gpt-5.2", max_turns: int = 200, verbose: bool = True
    ):
        super().__init__(
            agent_type="PaperAnalysis",
            model=model,
            max_turns=max_turns,
            verbose=verbose,
        )

    def _build_system_prompt(self, **kwargs) -> str:
        project_dir = str(kwargs.get("project_dir", "") or "")
        artifact_dir = str(kwargs.get("artifact_dir", "") or "")
        output_path = str(kwargs.get("output_path", "") or "")
        return render_template(
            "analysis_system.j2",
            project_dir=project_dir,
            artifact_dir=artifact_dir,
            output_path=output_path,
        )

    def _build_user_prompt(self, **kwargs) -> str:
        return render_template("analysis_user.j2")

    def _get_tools(self) -> List:
        return [bash, file_viewer, write_file, edit_file]
