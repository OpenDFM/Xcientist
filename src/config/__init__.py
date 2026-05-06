"""Unified configuration loader for Xcientist."""

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).absolute().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"
_ENV_CANDIDATES = (
    REPO_ROOT / ".env",
    Path(__file__).parent / ".env",
)

for _env_path in _ENV_CANDIDATES:
    if _env_path.exists():
        load_dotenv(_env_path, override=False)

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

    config_source = config_path or os.environ.get("XCIENTIST_CONFIG_PATH")
    config_file = Path(config_source) if config_source else DEFAULT_CONFIG_PATH

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
    - ${oc.env:VAR_NAME[,default]} -> actual env var values
    - ${repo_root} -> current working directory (will be resolved later)
    - ${workspace} -> workspace.root (will be resolved later)
    """
    # First handle ${env:VAR_NAME} patterns
    env_pattern = re.compile(r'\$\{env:([^}]+)\}')
    content = env_pattern.sub(lambda m: os.environ.get(m.group(1), ""), content)

    # Also support OmegaConf's built-in env interpolation with optional default
    oc_env_pattern = re.compile(r'\$\{oc\.env:([^,}]+)(?:,([^}]*))?\}')
    content = oc_env_pattern.sub(
        lambda m: os.environ.get(m.group(1), m.group(2) or ""),
        content,
    )

    # Escape custom path placeholders for later resolution.
    content = content.replace("${repo_root}", "__REPO_ROOT__")
    content = content.replace("${workspace}", "__WORKSPACE__")

    return content


def _resolve_workspace(config: DictConfig) -> DictConfig:
    """Resolve custom path placeholders against repo root and workspace root."""
    repo_root = str(REPO_ROOT)

    # Use workspace.root from config if available, otherwise fallback to cwd/workspace
    if "workspace" in config and "root" in config.workspace:
        workspace = str(config.workspace.root).replace("__REPO_ROOT__", repo_root)
        if not Path(workspace).expanduser().is_absolute():
            workspace = str((REPO_ROOT / workspace).absolute())
        config.workspace.root = workspace
    else:
        workspace = os.path.join(repo_root, "workspace")

    def _replace_workspace(value):
        if isinstance(value, str):
            return value.replace("__REPO_ROOT__", repo_root).replace("__WORKSPACE__", workspace)
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
