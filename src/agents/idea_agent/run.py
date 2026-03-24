import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from src.agents.idea_agent.agent.artifacts import artifact_set
from src.agents.idea_agent.agent.ligagent import LigAgent
from src.agents.idea_agent.utils.core.logger import get_logger, init_logger
from src.agents.idea_agent.utils.core.config_loader import get_config_value, load_idea_agent_config
from src.agents.idea_agent.utils.core.run_inputs import clean_optional_text, load_topic, resolve_run_inputs
from src.agents.idea_agent.utils.workflow.ligagent_flow import run_agent_loop

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
IDEA_AGENT_ROOT = Path(__file__).resolve().parent


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    slug = slug.strip("-")
    return slug or "topic"


def _run_topic(
    topic: str,
    output_root: str,
    run_id: str,
    include_console: bool,
    rag_config: str,
    resolved_inputs: Dict[str, object],
) -> str:
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    os.environ["IDEA_AGENT_TASK_TOPIC"] = topic
    print(f"[{topic}] 🏃 Starting run in {run_dir}...")

    init_logger(
        log_dir=str(run_dir / "logs"),
        filename="ligagent.log",
        include_console=include_console,
        include_timestamp=False,
        force_reinit=True,
    )
    logger = get_logger()
    logger.info("========================================")
    logger.info("💡 The research topic is %s", topic)

    config = load_idea_agent_config()
    agent = LigAgent(run_dir=run_dir, rag_config=rag_config, config=config)
    agent.bootstrap_topic(topic)

    if clean_optional_text(str(resolved_inputs.get("input_text") or "")):
        artifact_set(agent.artifact, "input_text", clean_optional_text(str(resolved_inputs["input_text"])))
    artifact_set(agent.artifact, "topic_source", str(resolved_inputs.get("topic_source") or ""))
    if clean_optional_text(str(resolved_inputs.get("mature_idea") or "")):
        artifact_set(agent.artifact, "mature_idea", clean_optional_text(str(resolved_inputs["mature_idea"])))
    artifact_set(agent.artifact, "mature_idea_source", str(resolved_inputs.get("mature_idea_source") or ""))
    if clean_optional_text(str(resolved_inputs.get("refinement_scope") or "")):
        artifact_set(
            agent.artifact,
            "refinement_scope",
            clean_optional_text(str(resolved_inputs["refinement_scope"])),
        )
    artifact_set(
        agent.artifact,
        "refinement_scope_source",
        str(resolved_inputs.get("refinement_scope_source") or ""),
    )

    try:
        run_agent_loop(agent, logger)
    except (Exception, KeyboardInterrupt):
        logger.info("Artifact snapshot at failure: %s", getattr(agent, "artifact", {}))
        tb = traceback.format_exc()
        logger.error("Traceback:\n%s", tb)
        raise RuntimeError(f"Worker failed for topic '{topic}': {tb}") from None

    logger.info("✅ Finished topic '%s'. Results in %s", topic, run_dir)
    return str(run_dir)


def _apply_env_config(config: Optional[object]) -> None:
    if config is None:
        return
    env_map = {
        "OPENAI_API_KEY": "run.openai_api_key",
        "OPENAI_BASE_URL": "run.openai_base_url",
        "S2_API_KEY": "run.s2_api_key",
        "S2_API_TIMEOUT": "run.s2_api_timeout",
    }
    for env_var, key in env_map.items():
        if env_var not in os.environ:
            value = get_config_value(config, key, None)
            if value is not None:
                os.environ[env_var] = str(value)

def main() -> None:
    config = load_idea_agent_config()
    _apply_env_config(config)
    resolved_inputs = resolve_run_inputs(config, default_output_root=str(DEFAULT_OUTPUT_ROOT))
    topic = load_topic(str(resolved_inputs["topic"]))
    output_root = Path(str(resolved_inputs["output_root"])).expanduser()
    if not output_root.is_absolute():
        output_root = IDEA_AGENT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rag_config = str(resolved_inputs["rag_config"])
    include_console = bool(resolved_inputs["console_logs"])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    unique = uuid4().hex[:8]
    run_id = f"{_slugify(topic)}-{timestamp}-{unique}"
    try:
        result_dir = _run_topic(
            topic,
            str(output_root),
            run_id,
            include_console,
            rag_config,
            resolved_inputs,
        )
        print(f"[{topic}] ✅ completed -> {result_dir}")
    except Exception as exc:
        print(f"[{topic}] ❌ failed: {exc}")


if __name__ == "__main__":
    main()
