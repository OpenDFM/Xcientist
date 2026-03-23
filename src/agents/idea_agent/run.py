import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.agents.idea_agent.agent.ligagent import LigAgent
from src.agents.idea_agent.utils.core.config_loader import get_config_value, load_idea_agent_config
from src.agents.idea_agent.utils.workflow.ligagent_flow import run_agent_loop

from agent import get_logger, init_logger

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
IDEA_AGENT_ROOT = Path(__file__).resolve().parent


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    slug = slug.strip("-")
    return slug or "topic"


def _load_topic(topic: Optional[str]) -> str:
    value = topic.strip() if isinstance(topic, str) else ""
    if not value:
        raise ValueError("run.topic must be a non-empty string.")
    return value


def _run_topic(
    topic: str,
    output_root: str,
    run_id: str,
    include_console: bool,
    rag_config: str,
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

    try:
        run_agent_loop(agent, logger)
    except (Exception, KeyboardInterrupt):
        logger.info("Artifact snapshot at failure: %s", getattr(agent, "artifact", {}))
        tb = traceback.format_exc()
        logger.error("Traceback:\n%s", tb)
        raise RuntimeError(f"Worker failed for topic '{topic}': {tb}") from None

    logger.info("✅ Finished topic '%s'. Results in %s", topic, run_dir)
    return str(run_dir)


def _load_run_defaults(config: Optional[object]) -> dict:
    return {
        "topic": get_config_value(config, "run.topic", ""),
        "output_root": get_config_value(config, "run.output_root", str(DEFAULT_OUTPUT_ROOT)),
        "console_logs": get_config_value(config, "run.console_logs", False),
        "rag_config": get_config_value(
            config,
            "run.rag_config",
            "src/agents/survey_agent/config/outcomeRAG.yaml",
        ),
    }


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
    defaults = _load_run_defaults(config)
    topic = _load_topic(defaults["topic"])
    output_root = Path(defaults["output_root"]).expanduser()
    if not output_root.is_absolute():
        output_root = IDEA_AGENT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rag_config = defaults["rag_config"]
    include_console = defaults["console_logs"]

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
        )
        print(f"[{topic}] ✅ completed -> {result_dir}")
    except Exception as exc:
        print(f"[{topic}] ❌ failed: {exc}")


if __name__ == "__main__":
    main()
