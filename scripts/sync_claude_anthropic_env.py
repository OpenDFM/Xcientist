#!/usr/bin/env python3
"""Sync Anthropic entries from a project .env file into Claude Code settings."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Dict


def _fallback_dotenv_values(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values


def _dotenv_values(env_path: Path) -> Dict[str, str]:
    try:
        from dotenv import dotenv_values
    except Exception:
        return _fallback_dotenv_values(env_path)

    parsed = dotenv_values(env_path)
    return {str(key): str(value or "") for key, value in parsed.items() if key}


def _load_settings(settings_path: Path) -> Dict[str, object]:
    if not settings_path.exists():
        return {}
    with settings_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Claude settings must be a JSON object: {settings_path}")
    return payload


def _claude_settings_env(env_values: Dict[str, str]) -> Dict[str, str]:
    settings_env: Dict[str, str] = {}
    for key, value in env_values.items():
        target_key = "ANTHROPIC_AUTH_TOKEN" if key == "ANTHROPIC_API_KEY" else key
        settings_env[target_key] = value
    return settings_env


def _atomic_write_json(settings_path: Path, payload: Dict[str, object]) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = None
    if settings_path.exists():
        existing_mode = stat.S_IMODE(settings_path.stat().st_mode)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{settings_path.name}.",
        suffix=".tmp",
        dir=str(settings_path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.chmod(tmp_path, existing_mode if existing_mode is not None else 0o600)
        os.replace(tmp_path, settings_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def sync_anthropic_env(env_path: Path, settings_path: Path) -> int:
    if not env_path.exists():
        print(f"[claude-settings] skipped: env file not found: {env_path}")
        return 0

    env_values = _claude_settings_env({
        key: value
        for key, value in _dotenv_values(env_path).items()
        if key.startswith("ANTHROPIC_")
    })
    if not env_values:
        print(f"[claude-settings] skipped: no ANTHROPIC_* entries in {env_path}")
        return 0

    settings = _load_settings(settings_path)
    existing_env = settings.get("env")
    if not isinstance(existing_env, dict):
        existing_env = {}

    updated_env = {str(key): str(value) for key, value in existing_env.items()}
    updated_env.pop("ANTHROPIC_API_KEY", None)
    synced = 0
    removed = 0
    for key, value in sorted(env_values.items()):
        if value:
            updated_env[key] = value
            synced += 1
        elif key in updated_env:
            updated_env.pop(key, None)
            removed += 1

    settings["env"] = updated_env
    _atomic_write_json(settings_path, settings)

    suffix = f", removed {removed}" if removed else ""
    print(
        f"[claude-settings] synced {synced} ANTHROPIC_* entr"
        f"{'y' if synced == 1 else 'ies'} to {settings_path}{suffix}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync ANTHROPIC_* values from .env into Claude Code settings.json"
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the project .env file",
    )
    parser.add_argument(
        "--settings",
        default="~/.claude/settings.json",
        help="Path to Claude Code settings.json",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file).expanduser().resolve()
    settings_path = Path(args.settings).expanduser()
    return sync_anthropic_env(env_path, settings_path)


if __name__ == "__main__":
    raise SystemExit(main())
