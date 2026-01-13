from typing import Optional

from src.agents.experiment_agent.layers.prepare.agent import PrepareAgent, PrepareReport
from src.agents.experiment_agent.shared.utils.config import PREPARE_AGENT_MODEL


async def run_prepare(
    experiment_id: str,
    force: bool = False,
    clone_depth: int = 1,
    skip_repos: bool = False,
    skip_datasets: bool = False,
    model: Optional[str] = None,
    verbose: bool = True,
) -> PrepareReport:
    agent = PrepareAgent(model=model or PREPARE_AGENT_MODEL, verbose=bool(verbose))
    return await agent.prepare_workspace(
        experiment_id=experiment_id,
        force=bool(force),
        clone_depth=int(clone_depth),
        skip_repos=bool(skip_repos),
        skip_datasets=bool(skip_datasets),
    )
