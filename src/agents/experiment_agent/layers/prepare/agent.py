import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.shared.tools.core import (
    SecurityContext,
    get_prepare_tools,
)
from src.agents.experiment_agent.shared.utils.config import (
    PREPARE_AGENT_MODEL,
    ensure_experiment_dirs,
)
from src.agents.experiment_agent.shared.utils.prompts import load_and_render_prompt


@dataclass
class PrepareReport:
    experiment_id: str
    workspace_dir: str
    project_dir: str
    repos_dir: str
    dataset_dir: str
    idea_md_path: str


class PrepareAgent(BaseAgent):
    """
    LLM tool-calling agent that initializes an experiment workspace.

    It is intentionally not rule-based: it receives goals + inputs, then uses tools
    (bash/file_viewer/write_file/edit_file) to perform filesystem/network actions.
    """

    def __init__(self, model: str = PREPARE_AGENT_MODEL, verbose: bool = True):
        super().__init__(
            agent_type="PrepareAgent",
            model=model,
            max_turns=2000,
            verbose=verbose,
        )

    def _get_tools(self) -> List:
        return get_prepare_tools()

    def _build_system_prompt(self, **kwargs) -> str:
        _ = kwargs
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "system.txt")
        return load_and_render_prompt(prompt_path, variables={})

    def _build_user_prompt(self, **kwargs) -> str:
        pb = PromptBuilder()

        pb.add_header("Prepare Workspace Task", level=1)
        pb.add_key_value("experiment_id", str(kwargs.get("experiment_id") or ""))
        pb.add_key_value("idea_json_path", str(kwargs.get("idea_json_path") or ""))
        pb.add_key_value("workspace_dir", str(kwargs.get("workspace_dir") or ""))
        pb.add_key_value("project_dir", str(kwargs.get("project_dir") or ""))
        pb.add_key_value("repos_dir", str(kwargs.get("repos_dir") or ""))
        pb.add_key_value("dataset_dir", str(kwargs.get("dataset_dir") or ""))
        pb.add_text("")

        pb.add_header("Inputs (parsed from idea.json)", level=2)
        summary = kwargs.get("idea_summary") or {}
        try:
            pb.add_code(
                json.dumps(summary, ensure_ascii=False, indent=2), language="json"
            )
        except Exception:
            pb.add_text("(failed to dump idea_summary)")

        pb.add_header("Flags", level=2)
        pb.add_list(
            [
                f"force: {bool(kwargs.get('force'))}",
                f"skip_repos: {bool(kwargs.get('skip_repos'))}",
                f"skip_datasets: {bool(kwargs.get('skip_datasets'))}",
                f"clone_depth: {int(kwargs.get('clone_depth') or 1)}",
            ],
            ordered=False,
        )

        pb.add_header("Required Outputs", level=2)
        pb.add_list(
            [
                "Write <workspace_dir>/idea.md (English, UTF-8) with required sections.",
                "Create/ensure repos/ and dataset_candidate/ directories exist.",
                "**Create Python venv at <project_dir>/venv** for isolated dependency management.",
                "If not skipped: clone repos into repos/ (best-effort).",
                "If not skipped: download datasets from HuggingFace only (best-effort; skip missing).",
            ],
            ordered=False,
        )

        return pb.build()

    async def prepare_workspace(
        self,
        experiment_id: str,
        force: bool = False,
        clone_depth: int = 1,
        skip_repos: bool = False,
        skip_datasets: bool = False,
    ) -> PrepareReport:
        if not experiment_id:
            raise ValueError("experiment_id is required")

        paths = ensure_experiment_dirs(experiment_id)
        workspace_dir = str(paths.get("workspace_dir") or "")
        project_dir = str(paths.get("project_dir") or "")
        repos_dir = str(paths.get("repos_dir") or os.path.join(workspace_dir, "repos"))
        dataset_dir = str(
            paths.get("dataset_dir") or os.path.join(workspace_dir, "dataset_candidate")
        )
        idea_json_path = os.path.join(workspace_dir, "idea.json")
        idea_md_path = os.path.join(workspace_dir, "idea.md")

        SecurityContext.set_roots(
            project_root=os.path.abspath(project_dir),
            workspace_root=os.path.abspath(workspace_dir),
        )

        with open(idea_json_path, "r", encoding="utf-8") as f:
            raw_json_text = f.read()
        try:
            data = json.loads(raw_json_text)
        except Exception:
            data = {"_raw_text": raw_json_text}

        summary: Dict[str, Any] = {
            "idea": data.get("idea"),
            "experiment": data.get("experiment"),
            "repos": data.get("repos"),
            "datasets": data.get("datasets"),
            "_meta": data.get("_meta"),
        }

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            experiment_id=experiment_id,
            idea_json_path=os.path.abspath(idea_json_path),
            workspace_dir=os.path.abspath(workspace_dir),
            project_dir=os.path.abspath(project_dir),
            repos_dir=os.path.abspath(repos_dir),
            dataset_dir=os.path.abspath(dataset_dir),
            idea_summary=summary,
            force=bool(force),
            clone_depth=int(clone_depth),
            skip_repos=bool(skip_repos),
            skip_datasets=bool(skip_datasets),
        )

        await self._run_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=self._get_tools(),
            project_root=os.path.abspath(project_dir),
            purpose="prepare_workspace",
        )

        return PrepareReport(
            experiment_id=experiment_id,
            workspace_dir=os.path.abspath(workspace_dir),
            project_dir=os.path.abspath(project_dir),
            repos_dir=os.path.abspath(repos_dir),
            dataset_dir=os.path.abspath(dataset_dir),
            idea_md_path=os.path.abspath(idea_md_path),
        )
