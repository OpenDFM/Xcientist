"""Configuration loader for blog_agent."""

import os
import re
from pathlib import Path
from typing import Optional, Any

import yaml


_INTERPOLATION_PATTERN = re.compile(r"\$\{([^}]+)\}")
_REPO_ROOT = Path(__file__).absolute().parents[4]


def get_config_path() -> Path:
    """Get the default config file path."""
    return Path(__file__).resolve().parents[3] / "config" / "default.yaml"


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load blog agent configuration from the unified Xcientist config.

    Args:
        config_path: Optional path to config file. If None, uses default.

    Returns:
        Configuration dictionary
    """
    path = Path(config_path) if config_path else get_config_path()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    _resolve_workspace_root(config)
    blog_config = config.get("blog", config)

    return _resolve_value(blog_config, config)


def get_config_value(config: dict, key: str, default: Any = None) -> Any:
    """Get a config value with a default."""
    return config.get(key, default)


def _resolve_value(value: Any, root: dict) -> Any:
    """Resolve simple ${path.to.value} and ${oc.env:VAR,default} references."""
    if isinstance(value, dict):
        return {key: _resolve_value(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_value(item, root) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        expression = match.group(1)
        if expression.startswith("oc.env:"):
            raw_env = expression.removeprefix("oc.env:")
            name, _, default = raw_env.partition(",")
            return os.environ.get(name, default)
        referenced_value = _get_dotted_value(root, expression)
        resolved_value = _resolve_value(referenced_value, root)
        return "" if resolved_value is None else str(resolved_value)

    return _INTERPOLATION_PATTERN.sub(replace, value)


def _get_dotted_value(root: dict, dotted_path: str) -> Any:
    current: Any = root
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return ""
        current = current[part]
    return current


def _resolve_workspace_root(config: dict) -> None:
    workspace = config.get("workspace")
    if not isinstance(workspace, dict) or "root" not in workspace:
        return
    root = str(workspace["root"])
    if not Path(root).expanduser().is_absolute():
        workspace["root"] = str((_REPO_ROOT / root).absolute())
