"""Prepare phase worker prompt helpers."""

from __future__ import annotations


PREPARE_REPO_WORKER = "prepare_repo_worker"
PREPARE_ENV_WORKER = "prepare_env_worker"
PREPARE_DATASET_WORKER = "prepare_dataset_worker"
PREPARE_MODEL_WORKER = "prepare_model_worker"
PREPARE_SYNTHESIS_WORKER = "prepare_synthesis_worker"
PREPARE_WORKER = PREPARE_REPO_WORKER


def prepare_worker_prompt(stage_label: str) -> str:
    return f"""You are the prepare phase worker for the `{stage_label}` stage.
Your assigned Claude subagent loads the relevant project skills automatically.

Execute exactly one prepare-stage contract.

Rules:
- Read the full stage contract before taking action.
- If validator feedback exists, treat it as the highest-priority retry brief.
- Perform real local work; do not return only plans or recommendations.
- Keep `project/` runtime self-contained. `repos/` may be used as reference or selective copy sources, but they must never become runtime dependencies.
- Datasets must land under `dataset_candidate/` to count as prepared.
- Local models must be available from `model_candidate/` or `model_candidate/model_share/`.
- Never claim the whole prepare phase is complete; report only the current stage outcome.
"""


def create_prepare_repo_worker_agent(llm):
    _ = llm
    return {"role": PREPARE_REPO_WORKER, "system_prompt": prepare_worker_prompt("repos")}


def create_prepare_env_worker_agent(llm):
    _ = llm
    return {"role": PREPARE_ENV_WORKER, "system_prompt": prepare_worker_prompt("env")}


def create_prepare_dataset_worker_agent(llm):
    _ = llm
    return {"role": PREPARE_DATASET_WORKER, "system_prompt": prepare_worker_prompt("dataset")}


def create_prepare_model_worker_agent(llm):
    _ = llm
    return {"role": PREPARE_MODEL_WORKER, "system_prompt": prepare_worker_prompt("model")}


def create_prepare_synthesis_worker_agent(llm):
    _ = llm
    return {"role": PREPARE_SYNTHESIS_WORKER, "system_prompt": prepare_worker_prompt("synthesis")}


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
