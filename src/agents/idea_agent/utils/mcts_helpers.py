import json
from typing import Any, Dict, List


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
    if not analysis:
        return "No prior analysis."
    latest = analysis[-1]
    if isinstance(latest, dict):
        try:
            return json.dumps(latest, ensure_ascii=False, indent=2)
        except Exception:
            return str(latest)
    return str(latest)


def format_edit_operators(edit_operators: List[Any]) -> str:
    lines = []
    for op in edit_operators:
        lines.append(
            f"- {op.name}: {op.description} | targets {', '.join(op.defects)} | guardrails: {', '.join(op.guardrails)}"
        )
    return "\n".join(lines)

def clip_text(value: Any, limit: int = 800) -> str:
    text = "" if value is None else str(value).strip()
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + "..."
