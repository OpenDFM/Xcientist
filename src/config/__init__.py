"""Unified configuration loader for X-Scientist"""

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from omegaconf import OmegaConf, DictConfig

# Try to load .env file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

# Global config cache
_config: Optional[DictConfig] = None


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """Load configuration file

    Args:
        config_path: Path to config file, uses default if None

    Returns:
        Parsed configuration object
    """
    global _config

    if _config is not None and config_path is None:
        return _config

    config_file = Path(config_path) if config_path else Path(__file__).parent / "default.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Read raw YAML content and preprocess variables
    # This prevents OmegaConf from trying to resolve them as interpolations
    raw_content = _preprocess_yaml_content(config_file.read_text(encoding="utf-8"))

    # Load config from preprocessed content
    _config = OmegaConf.create(raw_content)

    # Resolve ${workspace} variable
    _config = _resolve_workspace(_config)

    return _config


def _preprocess_yaml_content(content: str) -> str:
    """Preprocess YAML content to resolve custom variables before OmegaConf loads.

    This handles:
    - ${env:VAR_NAME} -> actual env var values
    - ${workspace} -> current working directory (will be resolved later)
    """
    workspace = os.getcwd()

    # First handle ${env:VAR_NAME} patterns
    env_pattern = re.compile(r'\$\{env:([^}]+)\}')
    content = env_pattern.sub(lambda m: os.environ.get(m.group(1), ""), content)

    # Escape ${workspace} for now (will be resolved later)
    # Use a placeholder that won't conflict
    content = content.replace("${workspace}", "__WORKSPACE__")

    return content


def _resolve_workspace(config: DictConfig) -> DictConfig:
    """Resolve __WORKSPACE__ placeholder to workspace.root from config or current working directory"""
    # Use workspace.root from config if available, otherwise fallback to cwd
    if "workspace" in config and "root" in config.workspace:
        workspace = config.workspace.root
    else:
        workspace = os.getcwd()

    def _replace_workspace(value):
        if isinstance(value, str):
            return value.replace("__WORKSPACE__", workspace)
        return value

    return _recursive_apply(config, _replace_workspace)


def _recursive_apply(config, func):
    """Recursively apply function to config"""
    if isinstance(config, DictConfig):
        result = OmegaConf.create()
        for key in config:
            result[key] = _recursive_apply(config[key], func)
        return result
    elif isinstance(config, list):
        return [_recursive_apply(item, func) for item in config]
    else:
        return func(config)


def get_config() -> DictConfig:
    """Get global config (load first if not loaded)"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: Optional[str] = None) -> DictConfig:
    """Reload configuration

    Args:
        config_path: Path to config file

    Returns:
        Reloaded configuration
    """
    global _config
    _config = None
    return load_config(config_path)


def get_survey_config() -> DictConfig:
    """Get Survey Agent configuration"""
    return get_config().survey


def get_idea_config() -> DictConfig:
    """Get Idea Agent configuration"""
    return get_config().idea


def get_experiment_config() -> DictConfig:
    """Get Experiment Agent configuration"""
    return get_config().experiment


def get_pipeline_config() -> DictConfig:
    """Get Pipeline configuration"""
    return get_config().pipeline


def get_project_config() -> DictConfig:
    """Get global project configuration"""
    return get_config().project


def to_container(config: DictConfig, resolve: bool = True) -> dict:
    """Convert DictConfig to plain dict

    Args:
        config: DictConfig object
        resolve: Whether to resolve variable references

    Returns:
        Converted dictionary
    """
    return OmegaConf.to_container(config, resolve=resolve)
