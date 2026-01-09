import glob
import os
from dataclasses import dataclass
from typing import List


@dataclass
class ExperimentWorkspaceContext:
    experiment_id: str
    workspace_dir: str
    idea_md: str
    specs_dir: str
    project_dir: str
    spec_files: List[str]
    result_files: List[str]


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p or "")))


def resolve_experiment_workspace_context(
    experiment_id: str, experiment_workspaces_root: str
) -> ExperimentWorkspaceContext:
    exp_id = str(experiment_id or "").strip()
    if not exp_id:
        raise ValueError("experiment_id is required")

    root = _abs(experiment_workspaces_root)
    workspace_dir = _abs(os.path.join(root, exp_id))
    if not os.path.isdir(workspace_dir):
        raise FileNotFoundError(f"experiment workspace not found: {workspace_dir}")

    idea_md = _abs(os.path.join(workspace_dir, "idea.md"))
    specs_dir = _abs(os.path.join(workspace_dir, "specs"))
    project_dir = _abs(os.path.join(workspace_dir, "project"))

    spec_files: List[str] = []
    if os.path.isfile(idea_md):
        spec_files.append(idea_md)
    if os.path.isdir(specs_dir):
        for root_dir, _, files in os.walk(specs_dir):
            for fn in files:
                p = _abs(os.path.join(root_dir, fn))
                if os.path.isfile(p):
                    spec_files.append(p)

    result_files: List[str] = []
    for pat in [
        os.path.join(
            project_dir, "result", "science", "iter_v*", "result_summary.json"
        ),
        os.path.join(project_dir, "result", "science", "iter_v*", "report.md"),
        os.path.join(project_dir, "result", "science", "iter_v*", "feedback.md"),
        os.path.join(project_dir, "result", "code", "iter_v*", "report.md"),
        os.path.join(project_dir, "result", "code", "iter_v*", "feedback.md"),
    ]:
        result_files.extend([_abs(p) for p in glob.glob(pat)])

    return ExperimentWorkspaceContext(
        experiment_id=exp_id,
        workspace_dir=workspace_dir,
        idea_md=idea_md,
        specs_dir=specs_dir,
        project_dir=project_dir,
        spec_files=sorted(set(spec_files)),
        result_files=sorted(set(result_files)),
    )
