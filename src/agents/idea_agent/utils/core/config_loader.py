"""Configuration loading helpers for Idea Agent runtime defaults and overrides."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "config" / "default.yaml"
)


def _resolve_config_path(config_path: Optional[str]) -> Path:
    if config_path:
        path = Path(config_path)
    else:
        env_path = os.getenv("IDEA_AGENT_CONFIG")
        path = Path(env_path) if env_path else DEFAULT_CONFIG_PATH
    return path.expanduser().resolve()


def _load_defaults(defaults: list[Any], base_dir: Path) -> Any:
    merged = OmegaConf.create()
    for entry in defaults or []:
        if isinstance(entry, str):
            if entry in ("_self_", "self"):
                continue
            if "/" in entry:
                group, name = entry.split("/", 1)
                candidate = base_dir / group / f"{name}.yaml"
            else:
                candidate = base_dir / f"{entry}.yaml"
        elif isinstance(entry, Mapping):
            candidate = None
            for group, name in entry.items():
                if group in ("_self_", "self"):
                    continue
                candidate = base_dir / group / f"{name}.yaml"
                break
        else:
            candidate = None

        if candidate and candidate.exists():
            merged = OmegaConf.merge(merged, OmegaConf.load(candidate))
    return merged


def load_project_config(config_path: Optional[str] = None) -> Any:
    path = _resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Idea agent config not found at {path}")
    config = OmegaConf.load(path)
    defaults = config.get("defaults") if isinstance(config, dict) or hasattr(config, "get") else None
    if not defaults:
        return config
    merged = _load_defaults(defaults, path.parent)
    without_defaults = OmegaConf.create({k: v for k, v in config.items() if k != "defaults"})
    return OmegaConf.merge(merged, without_defaults)


def load_idea_agent_config(config_path: Optional[str] = None) -> Any:
    config = load_project_config(config_path)
    idea_config = config.get("idea") if hasattr(config, "get") else None
    if idea_config is not None:
        return idea_config
    return config


def get_config_value(config: Optional[Any], key: str, default: Any) -> Any:
    if config is None:
        return default
    try:
        value = OmegaConf.select(config, key)
    except Exception:
        value = None
    return default if value is None else value


def get_workspace_root(config: Optional[Any] = None) -> Path:
    workspace_root = get_config_value(config, "workspace.root", None)
    if workspace_root is None:
        project_config = load_project_config()
        workspace_root = get_config_value(project_config, "workspace.root", None)
    return Path(str(workspace_root)).expanduser().resolve()


def resolve_workspace_path(path_value: Any, config: Optional[Any] = None) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return raw
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((get_workspace_root(config) / candidate).resolve())
