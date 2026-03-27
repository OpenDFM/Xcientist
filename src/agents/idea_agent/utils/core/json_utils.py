"""Shared JSON formatting and file helpers for Idea Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def compact_json(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    )


def read_json_file(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: Any) -> None:
    Path(path).write_text(pretty_json(payload) + "\n", encoding="utf-8")
