"""Configuration loader for blog_agent."""

import os
from pathlib import Path
from typing import Optional, Any
import yaml


def get_config_path() -> Path:
    """Get the default config file path."""
    return Path(__file__).parent / "config.yaml"


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load blog agent configuration.

    Args:
        config_path: Optional path to config file. If None, uses default.

    Returns:
        Configuration dictionary
    """
    path = Path(config_path) if config_path else get_config_path()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_config_value(config: dict, key: str, default: Any = None) -> Any:
    """Get a config value with a default."""
    return config.get(key, default)
