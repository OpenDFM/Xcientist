"""Configuration loading helpers for Idea Agent runtime defaults and overrides."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "config" / "default.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[5]
_REPO_ROOT_TOKEN = "__REPO_ROOT__"
_WORKSPACE_TOKEN = "__WORKSPACE__"


def _resolve_config_path(config_path: Optional[str]) -> Path:
    if config_path:
        path = Path(config_path)
    else:
        env_path = os.getenv("IDEA_AGENT_CONFIG")
        path = Path(env_path) if env_path else DEFAULT_CONFIG_PATH
    return path.expanduser().resolve()


def _preprocess_yaml_content(content: str) -> str:
    env_pattern = re.compile(r"\$\{env:([^}]+)\}")
    content = env_pattern.sub(lambda match: os.environ.get(match.group(1), ""), content)

    oc_env_pattern = re.compile(r"\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}")
    content = oc_env_pattern.sub(
        lambda match: os.environ.get(match.group(1), match.group(2) or ""),
        content,
    )

    content = content.replace("${repo_root}", _REPO_ROOT_TOKEN)
    content = content.replace("${workspace}", _WORKSPACE_TOKEN)
    return content


def _load_yaml_config(path: Path) -> Any:
    raw_content = _preprocess_yaml_content(path.read_text(encoding="utf-8"))
    return OmegaConf.create(raw_content)


def _recursive_apply(value: Any, transform) -> Any:
    if OmegaConf.is_dict(value):
        result = OmegaConf.create()
        for key in value:
            result[key] = _recursive_apply(value[key], transform)
        return result
    if OmegaConf.is_list(value):
        return [_recursive_apply(item, transform) for item in value]
    return transform(value)


def _resolve_custom_placeholders(config: Any) -> Any:
    repo_root = str(REPO_ROOT.resolve())
    workspace_root = OmegaConf.select(config, "workspace.root")
    if workspace_root is None:
        workspace = str((REPO_ROOT / "workspace").resolve())
    else:
        workspace = str(workspace_root).replace(_REPO_ROOT_TOKEN, repo_root)

    def _transform(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(_REPO_ROOT_TOKEN, repo_root).replace(_WORKSPACE_TOKEN, workspace)
        return value

    return _recursive_apply(config, _transform)


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
            merged = OmegaConf.merge(merged, _load_yaml_config(candidate))
    return merged


def load_project_config(config_path: Optional[str] = None) -> Any:
    path = _resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Idea agent config not found at {path}")
    config = _load_yaml_config(path)
    defaults = config.get("defaults") if isinstance(config, dict) or hasattr(config, "get") else None
    if not defaults:
        return _resolve_custom_placeholders(config)
    merged = _load_defaults(defaults, path.parent)
    without_defaults = OmegaConf.create({k: v for k, v in config.items() if k != "defaults"})
    return _resolve_custom_placeholders(OmegaConf.merge(merged, without_defaults))


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
    if workspace_root in (None, "", "None"):
        try:
            project_config = load_project_config()
        except Exception:
            project_config = None
        workspace_root = get_config_value(project_config, "workspace.root", None)
    if workspace_root in (None, "", "None"):
        return (REPO_ROOT / "workspace").resolve()
    return Path(str(workspace_root)).expanduser().resolve()


def resolve_workspace_path(path_value: Any, config: Optional[Any] = None) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return raw
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    repo_root = REPO_ROOT.resolve()
    workspace_root = get_workspace_root(config)

    for root in (repo_root, workspace_root):
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)

    repo_relative_roots = {"src", "models", "data", "database", "docs", "graph", "tests"}
    anchor = candidate.parts[0] if candidate.parts else ""
    base_dir = repo_root if anchor in repo_relative_roots else workspace_root
    return str((base_dir / candidate).resolve())
