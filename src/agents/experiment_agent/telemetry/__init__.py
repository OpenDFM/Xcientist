"""Console hooks and lightweight telemetry helpers."""

import os
import re
from typing import Any, Mapping, Optional

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class Colors:
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    MAGENTA = "\033[95m"


def _safe_width(width: int) -> int:
    return max(48, int(width or 72))


def _visible_len(value: object) -> int:
    return len(_ANSI_RE.sub("", str(value)))


def _color(text: str, color: str = "") -> str:
    if not color or not _supports_color():
        return text
    return f"{color}{text}{Colors.ENDC}"


def _supports_color() -> bool:
    return os.environ.get("NO_COLOR") is None


def _mask_key(key: str, value: Any) -> str:
    text = "" if value is None else str(value)
    if not text:
        return "NOT SET"
    lowered = key.lower()
    normalized_key = lowered.replace("-", "_").replace(" ", "_")
    compact_key = re.sub(r"[^a-z0-9]", "", lowered)
    sensitive_names = (
        "api_key",
        "access_token",
        "auth_token",
        "bearer_token",
        "secret",
        "password",
    )
    compact_sensitive_names = (
        "apikey",
        "accesstoken",
        "authtoken",
        "bearertoken",
        "githubaitoken",
        "secret",
        "password",
    )
    if any(token in normalized_key for token in sensitive_names) or any(
        token in compact_key for token in compact_sensitive_names
    ):
        suffix = text[-4:] if len(text) >= 4 else text
        return f"**********...{suffix}"
    return text


def _fit(text: Any, width: int) -> str:
    raw = str(text)
    if _visible_len(raw) <= width:
        return raw
    if width <= 3:
        return raw[:width]
    return raw[: width - 3] + "..."


def print_phase(
    title: str, subtitle: str = "", phase_num: Optional[int] = None, width: int = 72
) -> None:
    width = _safe_width(width)
    line = "─" * (width - 2)
    label = f"Phase {phase_num}: {title}" if phase_num is not None else title
    print(f"\n{_color('┌' + line + '┐', Colors.OKBLUE)}")
    print_box_line(_color(label, Colors.BOLD), width=width)
    if subtitle:
        print_box_line(subtitle, width=width)
    print(f"{_color('└' + line + '┘', Colors.OKBLUE)}")


def print_box_line(text: Any, *, width: int = 72, indent: int = 1) -> None:
    width = _safe_width(width)
    pad_width = width - 4 - indent
    content = " " * indent + _fit(text, pad_width)
    padding = " " * max(0, pad_width - _visible_len(content) + indent)
    print(f"{_color('│', Colors.OKBLUE)} {content}{padding} {_color('│', Colors.OKBLUE)}")


def print_kv_table(
    title: str,
    rows: Mapping[str, Any],
    *,
    width: int = 80,
    mask_sensitive: bool = True,
) -> None:
    width = _safe_width(width)
    print_phase(title, width=width)
    key_width = min(max((_visible_len(key) for key in rows), default=8), 24)
    for key, value in rows.items():
        rendered = _mask_key(key, value) if mask_sensitive else str(value)
        label = str(key).replace("_", " ").title()
        print(f"  {_color(label.ljust(key_width), Colors.OKCYAN)} : {rendered}")


def print_status(
    label: str,
    status: str,
    detail: str = "",
    *,
    color: str = Colors.OKGREEN,
    label_width: int = 22,
) -> None:
    label_width = max(8, int(label_width or 22))
    label_text = _fit(label, label_width).ljust(label_width)
    suffix = f"  {detail}" if detail else ""
    print(f"  {_color(label_text, Colors.OKCYAN)} {_color(status, color)}{suffix}", flush=True)


def print_activity(
    component: str,
    event: str,
    detail: str = "",
    *,
    color: str = Colors.OKGREEN,
) -> None:
    """Print one compact progress line for long-running nested agent activity."""
    print_status(component, event, detail, color=color, label_width=18)


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rem}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


__all__ = [
    "Colors",
    "format_seconds",
    "print_box_line",
    "print_activity",
    "print_kv_table",
    "print_phase",
    "print_status",
]
