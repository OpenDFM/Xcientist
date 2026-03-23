import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
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
    # 首先尝试从全局配置读取
    global_run_cfg = {}
    try:
        from src.config import load_config, get_idea_config
        load_config()
        idea_cfg = get_idea_config()
        if hasattr(idea_cfg, 'run'):
            global_run_cfg = idea_cfg.run
    except Exception:
        pass

    def _get_run_value(key: str, default: Any) -> Any:
        # 优先从全局配置获取
        if hasattr(global_run_cfg, key):
            return getattr(global_run_cfg, key)
        # 其次从局部配置获取
        return get_config_value(config, f"run.{key}", default)

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
    # 首先尝试从全局配置读取
    try:
        from src.config import load_config, get_idea_config
        global_config = load_config()
        idea_cfg = get_idea_config()
        if hasattr(idea_cfg, 'agent') and hasattr(idea_cfg.agent, 'model'):
            os.environ["IDEA_AGENT_MODEL"] = str(idea_cfg.agent.model)
        if hasattr(idea_cfg, 'api'):
            api_cfg = idea_cfg.api
            if hasattr(api_cfg, 'openai_api_key'):
                os.environ["OPENAI_API_KEY"] = str(api_cfg.openai_api_key)
            if hasattr(api_cfg, 'openai_base_url'):
                os.environ["OPENAI_BASE_URL"] = str(api_cfg.openai_base_url)
            if hasattr(api_cfg, 's2_api_key'):
                os.environ["S2_API_KEY"] = str(api_cfg.s2_api_key)
            if hasattr(api_cfg, 'serper_api_key'):
                os.environ["SERPER_API_KEY"] = str(api_cfg.serper_api_key)
        if hasattr(idea_cfg, 'mcts'):
            mcts_cfg = idea_cfg.mcts
            # 基础 MCTS 配置
            mcts_fields = [
                'max_iterations', 'max_depth', 'branching_factor',
                'exploration_constant', 'generation_model', 'evaluation_model',
                'generation_temperature', 'evaluation_temperature',
                'generation_max_tokens', 'evaluation_max_tokens',
                'component_novelty_retrieval_top_k', 'min_confidence_for_memory',
                'pareto_top_k', 'skill_prior_memory_path', 'skill_prior_success_threshold'
            ]
            for field in mcts_fields:
                if hasattr(mcts_cfg, field):
                    env_key = f"IDEA_MCTS_{field.upper()}"
                    os.environ[env_key] = str(getattr(mcts_cfg, field))
            # Weights 配置
            if hasattr(mcts_cfg, 'weights'):
                weights_cfg = mcts_cfg.weights
                weight_fields = ['alignment', 'complexity', 'novelty', 'impact', 'feasibility']
                for field in weight_fields:
                    if hasattr(weights_cfg, field):
                        os.environ[f"IDEA_MCTS_WEIGHT_{field.upper()}"] = str(getattr(weights_cfg, field))
    except Exception as e:
        print(f"Warning: Failed to load global config: {e}")

    if config is None:
        return
    env_map = {
        "OPENAI_API_KEY": "run.openai_api_key",
        "OPENAI_BASE_URL": "run.openai_base_url",
        "S2_API_KEY": "run.s2_api_key",
        "S2_API_TIMEOUT": "run.s2_api_timeout",
        "MINERU_MODEL_SOURCE": "run.mineru_model_source",
    }
    for env_var, key in env_map.items():
        # 只在环境变量未设置时才从局部配置读取
        if env_var not in os.environ:
            value = get_config_value(config, key, None)
            if value is not None:
                os.environ[env_var] = str(value)

    # 设置默认模型（如果环境变量未设置）
    if "IDEA_AGENT_MODEL" not in os.environ:
        model = get_config_value(config, "agent.model", "gpt-4.1")
        os.environ["IDEA_AGENT_MODEL"] = str(model)


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
