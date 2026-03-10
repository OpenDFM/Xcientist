from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlparse

_SUPPORTED_API_STYLES = {"auto", "responses", "chat_completions"}


def normalize_api_style(api_style: Optional[str]) -> str:
    style = str(api_style or "auto").strip().lower()
    if style not in _SUPPORTED_API_STYLES:
        supported = ", ".join(sorted(_SUPPORTED_API_STYLES))
        raise ValueError(f"Unsupported OPENAI_API_STYLE '{style}'. Expected one of: {supported}.")
    return style


def resolve_chat_transport(base_url: Optional[str], api_style: Optional[str] = "auto") -> str:
    style = normalize_api_style(api_style)
    if style != "auto":
        return style

    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        return "responses"

    parsed = urlparse(normalized)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    if host.endswith("api.openai.com"):
        return "responses"
    if "coding.dashscope.aliyuncs.com" in host:
        return "chat_completions"
    if "dashscope.aliyuncs.com" in host and "compatible-mode" in path:
        return "chat_completions"
    return "responses"


def ensure_default_max_output_tokens(
    kwargs: Mapping[str, Any],
    default_max_output_tokens: int,
) -> Dict[str, Any]:
    normalized = dict(kwargs)
    has_token_limit = any(
        key in normalized
        for key in ("max_output_tokens", "max_tokens", "max_completion_tokens")
    )
    if not has_token_limit:
        normalized["max_output_tokens"] = default_max_output_tokens
    return normalized


def normalize_responses_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(kwargs)
    if "max_output_tokens" not in normalized:
        if "max_completion_tokens" in normalized:
            normalized["max_output_tokens"] = normalized.pop("max_completion_tokens")
        elif "max_tokens" in normalized:
            normalized["max_output_tokens"] = normalized.pop("max_tokens")
    normalized.pop("max_completion_tokens", None)
    normalized.pop("max_tokens", None)
    return normalized


def normalize_chat_completions_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(kwargs)
    normalized.pop("reasoning", None)
    if "max_tokens" not in normalized and "max_completion_tokens" not in normalized:
        max_output_tokens = normalized.pop("max_output_tokens", None)
        if max_output_tokens is not None:
            normalized["max_tokens"] = max_output_tokens
    else:
        normalized.pop("max_output_tokens", None)
    return normalized


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    fragments = []
    for part in content:
        if isinstance(part, str):
            if part.strip():
                fragments.append(part.strip())
            continue
        if isinstance(part, Mapping):
            part_type = str(part.get("type", "")).lower()
            text = part.get("text")
            if text and part_type in {"text", "output_text", "input_text"}:
                fragments.append(str(text).strip())
            continue
        text = getattr(part, "text", None)
        if text:
            fragments.append(str(text).strip())

    return "\n".join(fragment for fragment in fragments if fragment).strip()


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        text = extract_text_content(content)
        if text:
            return text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if content is None and isinstance(item, Mapping):
                content = item.get("content")
            text = extract_text_content(content)
            if text:
                return text

    raise ValueError("LLM response did not include any text content.")
