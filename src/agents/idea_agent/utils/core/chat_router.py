from __future__ import annotations

from typing import Any, Dict, Mapping


HIGH_REASONING_STAGES = frozenset(
    {
        "advanced_analysis",
        "algorithm_alignment",
        "algorithm_structuring",
        "experiment_findings_extraction",
        "idea_fusion",
        "mcts_expand",
        "re_analysis_replan",
    }
)


def _is_gpt5_family(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("gpt-5")


def _is_gemini_3_pro_family(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith(("gemini-3-pro", "gemini-3.1-pro"))


def _is_claude_opus_4_6_family(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("claude-opus-4-6")


def _is_claude_sonnet_4_6_family(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("claude-sonnet-4-6")


def _is_deepseek_v3_2_family(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("deepseek-v3.2")


def _is_kimi_k2_5_family(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("kimi-k2.5")


def _is_glm_5_family(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith("glm-5")


def prepare_ligagent_chat_request(
    *,
    model: str,
    stage: str,
    kwargs: Mapping[str, Any],
) -> tuple[str, Dict[str, Any]]:
    resolved_model = str(model or "").strip()
    if not resolved_model:
        raise ValueError("LigAgent chat requires a non-empty model name.")

    request_kwargs = dict(kwargs)

    if _is_gpt5_family(resolved_model):
        request_kwargs["temperature"] = 1.0
        request_kwargs["reasoning"] = {
            "effort": "high" if stage in HIGH_REASONING_STAGES else "low"
        }
        return resolved_model, request_kwargs

    if _is_gemini_3_pro_family(resolved_model):
        return resolved_model, request_kwargs

    if _is_claude_opus_4_6_family(resolved_model):
        return resolved_model, request_kwargs

    if _is_claude_sonnet_4_6_family(resolved_model):
        return resolved_model, request_kwargs

    if _is_deepseek_v3_2_family(resolved_model):
        return resolved_model, request_kwargs

    if _is_kimi_k2_5_family(resolved_model):
        return resolved_model, request_kwargs

    if _is_glm_5_family(resolved_model):
        return resolved_model, request_kwargs

    raise ValueError(f"Unsupported LigAgent chat model: {resolved_model}")
