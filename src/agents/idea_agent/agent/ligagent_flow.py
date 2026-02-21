from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.prompts import PROMPTS
from src.agents.idea_agent.utils.idea_helpers import build_mcts_evolution, collect_reference_material
from src.agents.idea_agent.utils.ligagent_helpers import (
    build_algorithm_spec,
    suggest_baselines,
    suggest_datasets,
    synthesize_reference_summaries,
)
from src.agents.idea_agent.utils.ligagent_utils import (
    collect_paper_context_entries,
    generate_idea_introduction,
)


def run_agent_loop(agent, max_turns: int, logger) -> None:
    """Run the main planner loop for a LigAgent instance."""
    for turn in range(max_turns):
        logger.info("========================================")
        logger.info("Turn %d:", turn + 1)
        logger.info("🧠 Selecting action...")
        if not agent.artifact["steps"]:
            action = "knowledge_aquisition"
        else:
            action = agent.select_action(observation=agent.artifact["steps"][-1])
        agent.perform_action(action)


def persist_final_idea(
    best_entry: Dict[str, Any],
    paper_entries: List[Dict[str, Any]],
    artifact: Dict[str, Any],
    idea_result_path: Path,
    chat_fn,
    model: str,
    logger,
    prompts: Optional[Dict[str, str]] = None,
    config: Optional[object] = None,
) -> Dict[str, Any]:
    prompts = prompts or PROMPTS
    topic = artifact["topic"][-1] if artifact.get("topic") else "unspecified topic"
    raw_refs = collect_reference_material(artifact.get("references", []))
    algorithm = build_algorithm_spec(
        best_entry,
        topic,
        raw_refs,
        artifact,
        prompts,
        chat_fn,
        model,
        logger,
    )
    references = synthesize_reference_summaries(
        topic,
        best_entry,
        algorithm,
        raw_refs,
        prompts,
        chat_fn,
        model,
        logger,
    )
    datasets = suggest_datasets(
        topic,
         best_entry,
         algorithm,
         references,
         prompts,
         chat_fn,
         model,
         logger,
         artifact=artifact,
         config=config,
     )
    baselines = suggest_baselines(
        topic,
        best_entry,
        algorithm,
        references,
        prompts,
        chat_fn,
        model,
        logger,
        artifact=artifact,
        config=config,
    )
    entries = paper_entries or collect_paper_context_entries(
        artifact, artifact.get("references", [])
    )
    introduction = generate_idea_introduction(
        chat_fn=chat_fn,
        prompt_template=prompts["idea_introduction"],
        model=model,
        topic=topic,
        best_entry=best_entry,
        paper_entries=entries,
        logger=logger,
    )
    payload = {
        "title": best_entry.get("title"),
        "abstract": best_entry.get("abstract"),
        "introduction": introduction,
        "algorithm": algorithm,
        "reference_papers": references,
        "datasets": datasets,
        "baselines": baselines,
        "mcts_evolution": build_mcts_evolution(best_entry),
    }
    if best_entry.get("idea_contract"):
        payload["idea_contract"] = best_entry.get("idea_contract")
    best_entry["introduction"] = introduction
    artifact["idea_result"] = payload
    try:
        idea_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(idea_result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("💾 Saved idea result to %s", idea_result_path)
    except OSError as exc:
        logger.error("⚠️ Failed to persist idea_result.json: %s", exc)
    return payload
