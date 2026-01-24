import argparse
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.ligagent import LigAgent
from agent import init_logger, get_logger


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "runs"


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
        sanitized = ["Reinforcement Learning for LLM Reasoning"]
    return sanitized


def _expand_single_topic_for_parallelism(topics: Sequence[str], parallelism: int) -> List[tuple[str, Optional[int]]]:
    """Expand topics into (topic, replica_index) pairs.

    If there is only one topic but the user requests `parallelism > 1`, replicate
    that topic so we can utilize multiple workers.
    """

    cleaned = [t for t in topics if t]
    if len(cleaned) == 1 and parallelism > 1:
        return [(cleaned[0], i) for i in range(parallelism)]
    return [(t, None) for t in cleaned]


def _run_topic(topic: str, max_turn: int, output_root: str, run_id: str, include_console: bool, rag_config: str) -> str:
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

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

    agent = LigAgent(run_dir=run_dir, rag_config=rag_config)
    agent.bootstrap_topic(topic)

    try:
        for turn in range(max_turn):
            logger.info("========================================")
            logger.info("Turn %d:", turn + 1)
            logger.info("🧠 Selecting action...")
            if not agent.memory["steps"]:
                action = "knowledge_aquisition"
            else:
                action = agent.select_action(observation=agent.memory["steps"][-1])
            agent.perform_action(action)
    except (Exception, KeyboardInterrupt):
        logger.info(agent.memory)
        logger.error("Traceback:\n%s", traceback.format_exc())
        raise

    logger.info("✅ Finished topic '%s'. Results in %s", topic, run_dir)
    return str(run_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LigAgent for one or more topics in parallel.")
    parser.add_argument(
        "--topics",
        nargs="*",
        default=[],
        help="Topic strings to explore (can be repeated).",
    )
    parser.add_argument(
        "--topics-file",
        type=str,
        default=None,
        help="Optional path to a text/JSON file containing topics (one per line or JSON list).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Maximum planner turns per topic.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=None,
        help="Maximum number of concurrent topics. Defaults to len(topics).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where per-topic run folders (logs/results) are stored.",
    )
    parser.add_argument(
        "--console-logs",
        action="store_true",
        help="If set, also log to the console from each worker process.",
    )
    parser.add_argument(
        "--rag-config",
        type=str,
        default="config/idea_agent/rag_config.yaml",
        help="Path to RAG configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topics = _load_topics(args.topics, args.topics_file)
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    requested_parallelism = args.parallelism or len(topics)
    requested_parallelism = max(1, requested_parallelism)
    expanded = _expand_single_topic_for_parallelism(topics, requested_parallelism)
    parallelism = max(1, min(requested_parallelism, len(expanded)))

    rag_config = args.rag_config

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
                args.max_turns,
                str(output_root),
                run_id,
                args.console_logs,
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
