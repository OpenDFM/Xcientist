import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from src.agents.idea_agent.agent.artifacts import artifact_set
from src.agents.idea_agent.agent.base import AgentBase
from src.agents.idea_agent.agent.ligagent import LigAgent
from src.agents.idea_agent.agent.prompts.input_interpreter import INPUT_INTERPRETER_PROMPT
from src.agents.idea_agent.utils.core.logger import get_logger, init_logger
from src.agents.idea_agent.utils.core.config_loader import get_config_value, load_idea_agent_config
from src.agents.idea_agent.utils.workflow.ligagent_utils import parse_json_response
from src.agents.idea_agent.utils.workflow.ligagent_flow import run_agent_loop

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

    if _clean_optional_text(str(resolved_inputs.get("input_text") or "")):
        artifact_set(agent.artifact, "input_text", _clean_optional_text(str(resolved_inputs["input_text"])))
    artifact_set(agent.artifact, "topic_source", str(resolved_inputs.get("topic_source") or ""))
    if _clean_optional_text(str(resolved_inputs.get("mature_idea") or "")):
        artifact_set(agent.artifact, "mature_idea", _clean_optional_text(str(resolved_inputs["mature_idea"])))
    artifact_set(agent.artifact, "mature_idea_source", str(resolved_inputs.get("mature_idea_source") or ""))
    if _clean_optional_text(str(resolved_inputs.get("refinement_scope") or "")):
        artifact_set(
            agent.artifact,
            "refinement_scope",
            _clean_optional_text(str(resolved_inputs["refinement_scope"])),
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


def _load_run_defaults(config: Optional[object]) -> dict:
    return {
        "input": get_config_value(config, "run.input", ""),
        "topic": get_config_value(config, "run.topic", ""),
        "mature_idea": get_config_value(config, "run.mature_idea", ""),
        "refinement_scope": get_config_value(config, "run.refinement_scope", ""),
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


def _clean_optional_text(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_interpreter_source(value: Optional[str], *, allow_empty: bool = False) -> str:
    cleaned = _clean_optional_text(value).lower()
    allowed = {"explicit", "inferred"}
    if allow_empty:
        allowed.add("empty")
    return cleaned if cleaned in allowed else ("empty" if allow_empty else "inferred")


def _fallback_topic_from_input(input_text: str) -> str:
    candidate = re.split(r"(?<=[.!?])\s+", input_text.strip(), maxsplit=1)[0].strip()
    words = candidate.split()
    if len(words) > 12:
        candidate = " ".join(words[:12])
    return candidate or "unspecified topic"


def _interpret_input_text(input_text: str, config: Optional[object]) -> Dict[str, object]:
    normalized_input = _clean_optional_text(input_text)
    if not normalized_input:
        return {
            "topic": "",
            "topic_source": "empty",
            "mature_idea": "",
            "mature_idea_source": "empty",
            "refinement_scope": "",
            "refinement_scope_source": "empty",
            "needs_grounding": True,
        }

    model = (
        _clean_optional_text(get_config_value(config, "run.input_interpreter_model", ""))
        or _clean_optional_text(get_config_value(config, "agent.model", ""))
        or "gpt-5-mini"
    )
    prompt = INPUT_INTERPRETER_PROMPT.format(input_text=normalized_input)
    agent = AgentBase()
    try:
        response = agent.chat(prompt, model=model, temperature=0.1, max_output_tokens=4096)
        payload = parse_json_response(response)
    except Exception:
        return {
            "topic": _fallback_topic_from_input(normalized_input),
            "topic_source": "inferred",
            "mature_idea": "",
            "mature_idea_source": "empty",
            "refinement_scope": "",
            "refinement_scope_source": "empty",
            "needs_grounding": True,
        }

    topic = _clean_optional_text(payload.get("topic") if isinstance(payload, dict) else "")
    if not topic:
        topic = _fallback_topic_from_input(normalized_input)
    return {
        "topic": topic,
        "topic_source": _normalize_interpreter_source(
            payload.get("topic_source") if isinstance(payload, dict) else None
        ),
        "mature_idea": _clean_optional_text(payload.get("mature_idea") if isinstance(payload, dict) else ""),
        "mature_idea_source": _normalize_interpreter_source(
            payload.get("mature_idea_source") if isinstance(payload, dict) else None,
            allow_empty=True,
        ),
        "refinement_scope": _clean_optional_text(
            payload.get("refinement_scope") if isinstance(payload, dict) else ""
        ),
        "refinement_scope_source": _normalize_interpreter_source(
            payload.get("refinement_scope_source") if isinstance(payload, dict) else None,
            allow_empty=True,
        ),
        "needs_grounding": bool(payload.get("needs_grounding")) if isinstance(payload, dict) else True,
    }


def _resolve_run_inputs(config: Optional[object]) -> Dict[str, object]:
    defaults = _load_run_defaults(config)
    input_text = _clean_optional_text(defaults["input"])
    explicit_topic = _clean_optional_text(defaults["topic"])
    explicit_mature_idea = _clean_optional_text(defaults["mature_idea"])
    explicit_refinement_scope = _clean_optional_text(defaults["refinement_scope"])
    interpreted = (
        _interpret_input_text(input_text, config)
        if input_text and (not explicit_topic or not explicit_mature_idea or not explicit_refinement_scope)
        else {
            "topic": "",
            "topic_source": "empty",
            "mature_idea": "",
            "mature_idea_source": "empty",
            "refinement_scope": "",
            "refinement_scope_source": "empty",
            "needs_grounding": True,
        }
    )

    topic = explicit_topic or _clean_optional_text(str(interpreted.get("topic") or ""))
    if not topic:
        raise ValueError("run.topic must be non-empty, or run.input must contain an interpretable topic.")

    mature_idea = explicit_mature_idea or _clean_optional_text(str(interpreted.get("mature_idea") or ""))
    refinement_scope = explicit_refinement_scope or _clean_optional_text(
        str(interpreted.get("refinement_scope") or "")
    )
    interpreted_topic_source = str(interpreted.get("topic_source") or "inferred")
    interpreted_mature_source = str(interpreted.get("mature_idea_source") or "empty")
    interpreted_scope_source = str(interpreted.get("refinement_scope_source") or "empty")
    return {
        **defaults,
        "topic": topic,
        "mature_idea": mature_idea,
        "refinement_scope": refinement_scope,
        "input_text": input_text,
        "topic_source": (
            "config_explicit"
            if explicit_topic
            else f"input_{interpreted_topic_source}"
        ),
        "mature_idea_source": (
            "config_explicit"
            if explicit_mature_idea
            else (
                "empty"
                if interpreted_mature_source == "empty"
                else f"input_{interpreted_mature_source}"
            )
        ),
        "refinement_scope_source": (
            "config_explicit"
            if explicit_refinement_scope
            else (
                "empty"
                if interpreted_scope_source == "empty"
                else f"input_{interpreted_scope_source}"
            )
        ),
        "needs_grounding": bool(interpreted.get("needs_grounding", True)),
    }


def main() -> None:
    config = load_idea_agent_config()
    _apply_env_config(config)
    resolved_inputs = _resolve_run_inputs(config)
    topic = _load_topic(str(resolved_inputs["topic"]))
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
