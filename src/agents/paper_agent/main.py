import argparse
import asyncio
import os
import sys

# Add project root to sys.path (avoid shadowing third-party `agents` package)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT and (_PROJECT_ROOT not in sys.path):
    sys.path.insert(0, _PROJECT_ROOT)

from src.agents.paper_agent.entry import run_paper_cycle
from src.agents.paper_agent.utils.config import (
    BASE_WORKSPACES_DIR as PAPER_WORKSPACES_DIR,
    EXPERIMENT_WORKSPACES_DIR,
    LATEX_TEMPLATE_DIR,
    get_model_config,
    print_config,
)
from src.agents.paper_agent.utils.experiment_workspace import (
    resolve_experiment_workspace_context,
)


def _abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path or "")))


def get_args():
    parser = argparse.ArgumentParser(
        description="paper_agent - write a paper for an experiment workspace"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Experiment ID (experiment_agent workspace name)",
    )
    parser.add_argument(
        "--template-dir",
        type=str,
        default="",
        help="LaTeX template directory to copy into the paper workspace as 'paper/' (overrides PAPER_AGENT_LATEX_TEMPLATE_DIR)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing paper workspace for this experiment",
    )
    return parser.parse_args()


async def main(args) -> int:
    experiment_id = str(args.experiment or "").strip()
    if not experiment_id:
        raise ValueError("--experiment is required")

    print_config(experiment_id=experiment_id)
    if str(getattr(args, "template_dir", "") or "").strip():
        print(f"  latex_template_dir (cli): {_abs(args.template_dir)}")

    exp_ctx = resolve_experiment_workspace_context(
        experiment_id=experiment_id,
        experiment_workspaces_root=_abs(EXPERIMENT_WORKSPACES_DIR),
    )

    paper_workspaces_root = _abs(PAPER_WORKSPACES_DIR)
    run_name = experiment_id
    output_dir = paper_workspaces_root

    # Resume policy: if workspace exists and not resuming, force user to be explicit.
    run_dir = os.path.join(output_dir, run_name)
    state_path = os.path.join(run_dir, "state", "paper_state.json")
    if os.path.exists(state_path) and (not bool(args.resume)):
        raise ValueError(
            f"paper workspace already exists for experiment '{experiment_id}'. Use --resume to continue: {run_dir}"
        )

    result = await run_paper_cycle(
        run_name=str(run_name),
        template_dir=(
            _abs(args.template_dir)
            if str(getattr(args, "template_dir", "") or "").strip()
            else (_abs(LATEX_TEMPLATE_DIR) if LATEX_TEMPLATE_DIR else "")
        ),
        idea_md=_abs(exp_ctx.idea_md),
        project_dir=_abs(exp_ctx.project_dir),
        output_dir=_abs(output_dir),
        model=str(get_model_config().get("default") or "gpt-5.2"),
        models=get_model_config(),
        compile_first=False,
        run_writer=True,
        run_architect=True,
        final_compile_with_vlm=False,
        experiment_id=str(exp_ctx.experiment_id),
        experiment_workspace_dir=str(exp_ctx.workspace_dir),
        experiment_idea_md=str(exp_ctx.idea_md),
        experiment_specs_dir=str(exp_ctx.specs_dir),
        experiment_spec_files=list(exp_ctx.spec_files),
        experiment_result_files=list(exp_ctx.result_files),
        verbose=True,
        resume=bool(args.resume),
    )
    paths = result.get("paths") or {}
    print("\n✓ paper_agent run initialized")
    print(f"  run_dir: {paths.get('run_dir')}")
    print(f"  paper_dir: {paths.get('paper_dir')}")
    print(f"  artifact_dir: {paths.get('artifact_dir')}")
    print(f"  state: {paths.get('state_path')}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(get_args())))
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)
