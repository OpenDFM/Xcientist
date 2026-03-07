import json

from typing import Any, Dict, List

from src.agents.idea_agent.utils.prompting.prompt_views import format_analysis_prompt_view


def parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Strip potential code fences and capture the first JSON object/array.
    Fallbacks to incremental decoding if the model prepends commentary.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty response")
    if text.startswith("```"):
        fence_end = text.find("\n")
        if fence_end != -1:
            text = text[fence_end + 1 :]
        if text.endswith("```"):
            text = text[: -3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch in "{[":
                try:
                    parsed, _ = decoder.raw_decode(text[idx:])
                    return parsed
                except json.JSONDecodeError:
                    continue
    raise ValueError(f"Unable to parse JSON from response: {text[:200]}")


def format_analysis_blob(analysis: List[Any]) -> str:
    return format_analysis_prompt_view(analysis)


def clip_text(value: Any, limit: int = 800) -> str:
    text = "" if value is None else str(value).strip()
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "..."
