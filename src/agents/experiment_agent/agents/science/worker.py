"""
Science phase worker prompt helpers.
"""

from __future__ import annotations


STANDARD_SCIENCE_WORKER = "standard_science_worker"
ABLATION_SCIENCE_WORKER = "ablation_science_worker"


def standard_science_worker_prompt() -> str:
    return """You are the standard science worker.
Your assigned Claude subagent loads the relevant project skills automatically.

Execute exactly one standard-science contract from the planner. Run the real command chain described by the contract and preserve raw evidence before summarizing.

Rules:
- Read assigned contract, prepare targets, code handoff, and existing artifacts before acting.
- Use only real data from `dataset_candidate/`.
- Use real API credentials and prepared local models when required.
- Save raw outputs under the declared standard-results subtree.
- Keep `project/` runtime self-contained.
"""


def ablation_science_worker_prompt() -> str:
    return """You are the ablation science worker.
Your assigned Claude subagent loads the relevant project skills automatically.

Execute exactly one ablation-science contract from the planner. Each contract tests one canonical idea component exactly as named in `idea.json.components`.

Rules:
- Read assigned contract, prepare targets, code handoff, and standard-science evidence before acting.
- Use only real data from `dataset_candidate/`.
- Preserve exact component identity and method-context change.
- Save raw outputs under the declared ablation-results subtree.
- Do not write the final `ablation_results.json`.
"""


def create_standard_science_worker_agent(llm):
    _ = llm
    return {"role": STANDARD_SCIENCE_WORKER, "system_prompt": standard_science_worker_prompt()}


def create_ablation_science_worker_agent(llm):
    _ = llm
    return {"role": ABLATION_SCIENCE_WORKER, "system_prompt": ablation_science_worker_prompt()}
