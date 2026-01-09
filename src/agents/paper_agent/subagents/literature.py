from typing import List

from src.agents.paper_agent.prompts import render_template
from src.agents.paper_agent.tools.core import bash, edit_file, file_viewer, write_file
from src.agents.paper_agent.tools.mineru import mineru_parse_pdf
from src.agents.paper_agent.tools.semantic_scholar import (
    semantic_scholar_download_openaccess_pdf,
    semantic_scholar_paper,
    semantic_scholar_search,
)
from src.agents.paper_agent.utils.agent_base import BaseAgent


class PaperLiteratureAgent(BaseAgent):
    def __init__(
        self, model: str = "gpt-5.2", max_turns: int = 200, verbose: bool = True
    ):
        super().__init__(
            agent_type="PaperLiterature",
            model=model,
            max_turns=max_turns,
            verbose=verbose,
        )

    def _build_system_prompt(self, **kwargs) -> str:
        artifact_dir = str(kwargs.get("artifact_dir", "") or "")
        request = str(kwargs.get("request", "") or "")
        output_path = str(kwargs.get("output_path", "") or "")
        return render_template(
            "literature_system.j2",
            artifact_dir=artifact_dir,
            request=request,
            output_path=output_path,
        )

    def _build_user_prompt(self, **kwargs) -> str:
        return render_template("literature_user.j2")

    def _get_tools(self) -> List:
        return [
            bash,
            file_viewer,
            write_file,
            edit_file,
            semantic_scholar_search,
            semantic_scholar_paper,
            semantic_scholar_download_openaccess_pdf,
            mineru_parse_pdf,
        ]
