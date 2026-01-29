import json
import os
import re
import sys
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.agents.idea_agent.agent.ligagent import LigAgent
from src.agents.idea_agent.agent.ligagent_flow import run_agent_loop
from src.agents.idea_agent.utils.config_loader import load_idea_agent_config, get_config_value

from agent import init_logger, get_logger

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
IDEA_AGENT_ROOT = Path(__file__).resolve().parent


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    slug = slug.strip("-")
    return slug or "topic"


def _load_topics(topics: Sequence[str], topics_file: Optional[str]) -> List[str]:
    resolved: List[str] = [topic for topic in topics if topic]
    if topics_file:
        path = Path(topics_file)
        if not path.exists():
            raise FileNotFoundError(f"Topics file does not exist: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if text:
            if path.suffix.lower() == ".json":
                data = json.loads(text)
                if isinstance(data, str):
                    resolved.append(data.strip())
                elif isinstance(data, Iterable):
                    resolved.extend(str(item).strip() for item in data if str(item).strip())
                else:
                    raise ValueError("JSON topics file must contain a string or list of strings.")
            else:
                resolved.extend(line.strip() for line in text.splitlines() if line.strip())
    sanitized = [topic.strip() for topic in resolved if topic and topic.strip()]
    if not sanitized:
        raise ValueError("No valid topics found from inputs! Please check your configuration.")
    return sanitized


def _expand_single_topic_for_parallelism(topics: Sequence[str], parallelism: int) -> List[tuple[str, Optional[int]]]:
    cleaned = [t for t in topics if t]
    if len(cleaned) == 1 and parallelism > 1:
        return [(cleaned[0], i) for i in range(parallelism)]
    return [(t, None) for t in cleaned]


def _run_topic(
    topic: str,
    max_turn: int,
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
    logger.info("🤖 Hello, I am LigAgent!")
    logger.info("💡 The research topic is %s", topic)

    config = load_idea_agent_config()
    agent = LigAgent(run_dir=run_dir, rag_config=rag_config, config=config)
    agent.bootstrap_topic(topic)

    try:
        run_agent_loop(agent, max_turn, logger)
    except (Exception, KeyboardInterrupt):
        logger.info(agent.memory)
        tb = traceback.format_exc()
        logger.error("Traceback:\n%s", tb)
        # Ensure the exception is picklable for ProcessPoolExecutor.
        raise RuntimeError(f"Worker failed for topic '{topic}': {tb}") from None

    logger.info("✅ Finished topic '%s'. Results in %s", topic, run_dir)
    return str(run_dir)


def _load_run_defaults(config: Optional[object]) -> dict:
    return {
        "topics": get_config_value(config, "run.topics", []),
        "topics_file": get_config_value(config, "run.topics_file", None),
        "max_turns": get_config_value(config, "run.max_turns", None),
        "parallelism": get_config_value(config, "run.parallelism", None),
        "output_root": get_config_value(config, "run.output_root", str(DEFAULT_OUTPUT_ROOT)),
        "console_logs": get_config_value(config, "run.console_logs", False),
        "rag_config": get_config_value(config, "run.rag_config", "src/agents/survey_agent/config/outcomeRAG.yaml"),
    }


def _apply_env_config(config: Optional[object]) -> None:
    if config is None:
        return
    env_map = {
        "OPENAI_API_KEY": "run.openai_api_key",
        "OPENAI_BASE_URL": "run.openai_base_url",
        "S2_API_KEY": "run.s2_api_key",
        "S2_API_TIMEOUT": "run.s2_api_timeout",
        "SERPER_API_KEY": "run.serper_api_key",
        "MINERU_MODEL_SOURCE": "run.mineru_model_source",
    }
    for env_var, key in env_map.items():
        value = get_config_value(config, key, None)
        if value is None:
            continue
        os.environ[env_var] = str(value)


def main() -> None:
    config = load_idea_agent_config()
    _apply_env_config(config)
    defaults = _load_run_defaults(config)
    topics = _load_topics(defaults["topics"], defaults["topics_file"])
    max_turns = defaults["max_turns"] if defaults["max_turns"] is not None else 4
    output_root = Path(defaults["output_root"]).expanduser()
    if not output_root.is_absolute():
        output_root = IDEA_AGENT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    requested_parallelism = defaults["parallelism"] or len(topics)
    requested_parallelism = max(1, requested_parallelism)
    expanded = _expand_single_topic_for_parallelism(topics, requested_parallelism)
    parallelism = max(1, min(requested_parallelism, len(expanded)))

    rag_config = defaults["rag_config"]
    include_console = defaults["console_logs"]

    futures = {}
    with ProcessPoolExecutor(max_workers=parallelism) as executor:
        for topic, replica_index in expanded:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            unique = uuid4().hex[:8]
            run_id = f"{_slugify(topic)}-{timestamp}-{unique}"
            if replica_index is not None:
                run_id = f"{run_id}-r{replica_index + 1:02d}"
            future = executor.submit(
                _run_topic,
                topic,
                max_turns,
                str(output_root),
                run_id,
                include_console,
                rag_config,
            )
            futures[future] = (topic, output_root / run_id)

        for future in as_completed(futures):
            topic, run_dir = futures[future]
            try:
                result_dir = future.result()
                print(f"[{topic}] ✅ completed -> {result_dir}")
            except Exception as exc:
                print(f"[{topic}] ❌ failed: {exc}")


if __name__ == "__main__":
    main()
