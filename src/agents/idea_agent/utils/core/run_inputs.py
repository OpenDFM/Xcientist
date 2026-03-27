import os
import re
from typing import Dict, Optional

from src.agents.idea_agent.agent.base import AgentBase
from src.agents.idea_agent.agent.prompts.input_interpreter import INPUT_INTERPRETER_PROMPT
from src.agents.idea_agent.utils.core.config_loader import get_config_value
from src.agents.idea_agent.utils.core.response_parsing import parse_json_response


def clean_optional_text(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def load_topic(topic: Optional[str]) -> str:
    value = topic.strip() if isinstance(topic, str) else ""
    if not value:
        raise ValueError("run.topic must be a non-empty string.")
    return value


def load_run_defaults(config: Optional[object], *, default_output_root: str) -> dict:
    ablation_results_path = clean_optional_text(os.getenv("IDEA_AGENT_ABLATION_RESULTS_PATH"))
    if not ablation_results_path:
        ablation_results_path = clean_optional_text(
            str(get_config_value(config, "run.ablation_results_path", "") or "")
        )
    return {
        "input": get_config_value(config, "run.input", ""),
        "topic": get_config_value(config, "run.topic", ""),
        "mature_idea": get_config_value(config, "run.mature_idea", ""),
        "refinement_scope": get_config_value(config, "run.refinement_scope", ""),
        "ablation_results_path": ablation_results_path,
        "output_root": get_config_value(config, "run.output_root", default_output_root),
        "console_logs": get_config_value(config, "run.console_logs", False),
        "rag_config": get_config_value(
            config,
            "run.rag_config",
            "src/agents/survey_agent/config/outcomeRAG.yaml",
        ),
    }


def _normalize_interpreter_source(value: Optional[str], *, allow_empty: bool = False) -> str:
    cleaned = clean_optional_text(value).lower()
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
    normalized_input = clean_optional_text(input_text)
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
        clean_optional_text(get_config_value(config, "run.input_interpreter_model", ""))
        or clean_optional_text(get_config_value(config, "agent.model", ""))
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

    topic = clean_optional_text(payload.get("topic") if isinstance(payload, dict) else "")
    if not topic:
        topic = _fallback_topic_from_input(normalized_input)
    return {
        "topic": topic,
        "topic_source": _normalize_interpreter_source(
            payload.get("topic_source") if isinstance(payload, dict) else None
        ),
        "mature_idea": clean_optional_text(payload.get("mature_idea") if isinstance(payload, dict) else ""),
        "mature_idea_source": _normalize_interpreter_source(
            payload.get("mature_idea_source") if isinstance(payload, dict) else None,
            allow_empty=True,
        ),
        "refinement_scope": clean_optional_text(
            payload.get("refinement_scope") if isinstance(payload, dict) else ""
        ),
        "refinement_scope_source": _normalize_interpreter_source(
            payload.get("refinement_scope_source") if isinstance(payload, dict) else None,
            allow_empty=True,
        ),
        "needs_grounding": bool(payload.get("needs_grounding")) if isinstance(payload, dict) else True,
    }


def resolve_run_inputs(config: Optional[object], *, default_output_root: str) -> Dict[str, object]:
    defaults = load_run_defaults(config, default_output_root=default_output_root)
    input_text = clean_optional_text(defaults["input"])
    explicit_topic = clean_optional_text(defaults["topic"])
    explicit_mature_idea = clean_optional_text(defaults["mature_idea"])
    explicit_refinement_scope = clean_optional_text(defaults["refinement_scope"])
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

    topic = explicit_topic or clean_optional_text(str(interpreted.get("topic") or ""))
    if not topic:
        raise ValueError("run.topic must be non-empty, or run.input must contain an interpretable topic.")

    mature_idea = explicit_mature_idea or clean_optional_text(str(interpreted.get("mature_idea") or ""))
    refinement_scope = explicit_refinement_scope or clean_optional_text(
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
        "ablation_results_path": clean_optional_text(str(defaults.get("ablation_results_path") or "")),
        "input_text": input_text,
        "topic_source": "config_explicit" if explicit_topic else f"input_{interpreted_topic_source}",
        "mature_idea_source": (
            "config_explicit"
            if explicit_mature_idea
            else ("empty" if interpreted_mature_source == "empty" else f"input_{interpreted_mature_source}")
        ),
        "refinement_scope_source": (
            "config_explicit"
            if explicit_refinement_scope
            else ("empty" if interpreted_scope_source == "empty" else f"input_{interpreted_scope_source}")
        ),
        "needs_grounding": bool(interpreted.get("needs_grounding", True)),
    }
