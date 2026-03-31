"""Prepare phase worker agents."""

from __future__ import annotations

from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from src.agents.experiment_agent.agents.base.subagents import create_phase_subagent


PREPARE_REPO_WORKER = "prepare_repo_worker"
PREPARE_ENV_WORKER = "prepare_env_worker"
PREPARE_DATASET_WORKER = "prepare_dataset_worker"
PREPARE_MODEL_WORKER = "prepare_model_worker"
PREPARE_SYNTHESIS_WORKER = "prepare_synthesis_worker"
PREPARE_WORKER = PREPARE_REPO_WORKER


def _base_prepare_worker_prompt(stage_label: str, stage_focus: str) -> str:
    return f"""You are the prepare phase worker for the `{stage_label}` stage.

Your job is to execute exactly one prepare-stage contract. The planner defines the stage order and file contracts. You own only the current stage's local research and operations.

Core rules:
1. Read the full stage contract before taking action.
2. If validator feedback exists, treat it as the highest-priority retry brief.
3. Perform real local work; do not return only plans or recommendations.
4. Write the exact worker report path declared in the stage contract.
5. Back every claim with concrete evidence: commands, files, paths, entrypoints, or downloaded artifacts.
6. Keep `project/` self-contained. `repos/` are reference-only and must never become runtime dependencies.
7. Do not modify files outside the contract's allowed write roots.

Stage focus:
{stage_focus}

Hard prepare rules:
- Datasets must land under `dataset_candidate/` to count as prepared.
- Local models must be available from the prepared surface under `model_candidate/`, either as workspace-local downloads or as read-only shared assets under `model_candidate/model_share/`.
- API-only models may be recorded, but they are not local model downloads.
- Never use editable installs, local-path installs, import-path injection, or copied repo code to satisfy project requirements.
- Never claim the whole prepare phase is complete; report only the current stage outcome.
- Stage reports must separate researched candidates, selected targets, downloaded assets, reused assets, and skipped candidates.
- Stage reports must not stop at `validated existing resources`. For repo, dataset, and model stages, explicitly record current-run research evidence, selection rationale, and whether each selected target was downloaded, refreshed, or reused.
"""


def create_prepare_repo_worker_agent(llm):
    return create_phase_subagent(
        llm,
        role=PREPARE_REPO_WORKER,
        tool_names=[
            TerminalTool.name,
            FileEditorTool.name,
            TaskTrackerTool.name,
            "web_search",
            "github_search",
        ],
        system_prompt=_base_prepare_worker_prompt(
            "repos",
            "- Research benchmark repositories, official codebases, and exact runnable entrypoints.\n"
            "- Use web search and GitHub search as needed, then clone or refresh the repositories required for the formal experiment surface.\n"
            "- Prefer acquiring the validated relevant set of repositories instead of merely verifying what already exists.\n"
            "- Record exact repository URLs, local checkout paths, revisions, benchmark entrypoints, support files, and whether each repo was downloaded this run or reused.",
        ),
    )


def create_prepare_env_worker_agent(llm):
    return create_phase_subagent(
        llm,
        role=PREPARE_ENV_WORKER,
        tool_names=[
            TerminalTool.name,
            FileEditorTool.name,
            TaskTrackerTool.name,
            "web_search",
            "github_search",
        ],
        system_prompt=_base_prepare_worker_prompt(
            "env",
            "- Research and create the runnable environment under `project/venv`.\n"
            "- Use web search and GitHub search to verify dependency installation methods, official docs, and benchmark-specific setup requirements.\n"
            "- Record exact commands, imports, and environment variables required to run later phases.",
        ),
    )


def create_prepare_dataset_worker_agent(llm):
    return create_phase_subagent(
        llm,
        role=PREPARE_DATASET_WORKER,
        tool_names=[
            TerminalTool.name,
            FileEditorTool.name,
            TaskTrackerTool.name,
            "web_search",
            "hf_hub_search",
            "hf_hub_download",
            "modelscope_search",
            "modelscope_download",
        ],
        system_prompt=_base_prepare_worker_prompt(
            "dataset",
            "- Research and stage the formal experiment datasets.\n"
            "- Prefer registry-backed dataset discovery through HuggingFace or ModelScope tools, using web search only for confirmation.\n"
            "- After selecting the validated relevant datasets, download as many required prepared datasets as possible into `dataset_candidate/`, reusing only when a matching prepared copy already exists.\n"
            "- Record exact file paths, versions, whether each dataset was downloaded or reused, and any remaining acquisition blockers.",
        ),
    )


def create_prepare_model_worker_agent(llm):
    return create_phase_subagent(
        llm,
        role=PREPARE_MODEL_WORKER,
        tool_names=[
            TerminalTool.name,
            FileEditorTool.name,
            TaskTrackerTool.name,
            "web_search",
            "hf_hub_search",
            "hf_hub_download",
            "modelscope_search",
            "modelscope_download",
        ],
        system_prompt=_base_prepare_worker_prompt(
            "model",
            "- Research and stage only the local models that the formal experiment requires to exist on disk.\n"
            "- Use HuggingFace or ModelScope search and download tools. Reuse `model_candidate/` local contents or `model_candidate/model_share/` shared contents only when they satisfy the declared revision and required files.\n"
            "- If a required local model is not already available, download it into a dedicated subdirectory under `model_candidate/` instead of relying on runtime auto-download.\n"
            "- Record API-only models separately from local downloaded models, and explicitly distinguish downloaded, reused local, and reused shared model sources.",
        ),
    )


def create_prepare_synthesis_worker_agent(llm):
    return create_phase_subagent(
        llm,
        role=PREPARE_SYNTHESIS_WORKER,
        tool_names=[
            TerminalTool.name,
            FileEditorTool.name,
            TaskTrackerTool.name,
            "web_search",
        ],
        system_prompt=_base_prepare_worker_prompt(
            "synthesis",
            "- Produce `prepare_idea.md`, `prepare_target_inventory.json`, and handoff notes using only validator-backed outputs from earlier stages.\n"
            "- Do not perform primary discovery in this stage. Only verify and summarize previously validated facts.\n"
            "- Copy the exact ordered component list from `idea.json.components` without renaming, merging, or reordering.",
        ),
    )


__all__ = [
    "PREPARE_WORKER",
    "PREPARE_REPO_WORKER",
    "PREPARE_ENV_WORKER",
    "PREPARE_DATASET_WORKER",
    "PREPARE_MODEL_WORKER",
    "PREPARE_SYNTHESIS_WORKER",
    "create_prepare_repo_worker_agent",
    "create_prepare_env_worker_agent",
    "create_prepare_dataset_worker_agent",
    "create_prepare_model_worker_agent",
    "create_prepare_synthesis_worker_agent",
]
