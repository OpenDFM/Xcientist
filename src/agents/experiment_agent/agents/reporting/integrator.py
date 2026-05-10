from __future__ import annotations


EXPERIMENT_ABLATION_REPORT_INTEGRATOR = "experiment_ablation_report_integrator"


def ablation_report_integrator_prompt() -> str:
    return """You are the ablation report integrator.
Your assigned Claude subagent loads the relevant project skills automatically.

Your only job is to read idea and ablation evidence, then compose the final candidate `ablation_results.json` payload.

Rules:
- Read only `idea.json`, `agent_reports/`, and `results/ablation/` as evidence.
- Treat `idea.json.components` as the only canonical source for component names, order, and `method_context`.
- Do not invent unsupported component results.
- Prefer phase-level and step-level validator-backed JSON before raw logs.
- Return structured JSON only; the runtime will validate and write the final canonical artifact.
"""


def create_ablation_report_integrator_agent(llm):
    _ = llm
    return {
        "role": EXPERIMENT_ABLATION_REPORT_INTEGRATOR,
        "system_prompt": ablation_report_integrator_prompt(),
    }


def register_ablation_report_integrator() -> None:
    return None
