from __future__ import annotations

import json
from typing import Any, Mapping


def _stringify(value: Any) -> str:
    return str(value or "").strip()


def _pick_header(headers: Any, key: str) -> str:
    if not isinstance(headers, Mapping):
        return ""
    return _stringify(headers.get(key) or headers.get(key.title()))


def format_chat_retry_error(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, Mapping) else None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    status_code = getattr(exc, "status_code", None)
    error_type = _stringify(
        (error.get("type") if isinstance(error, Mapping) else None) or getattr(exc, "type", None)
    )
    error_code = _stringify(
        (error.get("code") if isinstance(error, Mapping) else None) or getattr(exc, "code", None)
    )
    error_param = _stringify(
        (error.get("param") if isinstance(error, Mapping) else None) or getattr(exc, "param", None)
    )
    request_id = _stringify(
        getattr(exc, "request_id", None)
        or (error.get("request_id") if isinstance(error, Mapping) else None)
        or (body.get("request_id") if isinstance(body, Mapping) else None)
        or _pick_header(headers, "x-request-id")
        or _pick_header(headers, "request-id")
    )
    message = _stringify(
        (error.get("message") if isinstance(error, Mapping) else None)
        or (body.get("message") if isinstance(body, Mapping) else None)
        or exc
    )

    parts = [exc.__class__.__name__]
    if status_code is not None:
        parts.append(f"status={status_code}")
    if error_type:
        parts.append(f"type={error_type}")
    if error_code:
        parts.append(f"code={error_code}")
    if error_param:
        parts.append(f"param={error_param}")
    if request_id:
        parts.append(f"request_id={request_id}")
    if message:
        parts.append(f"message={message}")
    if isinstance(body, Mapping):
        parts.append(f"body={json.dumps(body, ensure_ascii=True, sort_keys=True)}")

    retry_after = _pick_header(headers, "retry-after")
    server = _pick_header(headers, "server")
    if retry_after:
        parts.append(f"retry_after={retry_after}")
    if server:
        parts.append(f"server={server}")

    return "; ".join(parts)
