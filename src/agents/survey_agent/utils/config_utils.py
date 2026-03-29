from pathlib import Path

from omegaconf import DictConfig, OmegaConf


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "src" / "config" / "default.yaml"


def merge_with_default_survey_config(config: DictConfig) -> DictConfig:
    """Merge a survey preset config on top of src/config/default.yaml::survey."""
    default_root = OmegaConf.load(_DEFAULT_CONFIG_PATH)
    default_survey = OmegaConf.select(default_root, "survey")
    if default_survey is None:
        raise ValueError(f"Missing 'survey' section in default config: {_DEFAULT_CONFIG_PATH}")

    current_survey = OmegaConf.select(config, "survey")
    survey_config = current_survey if current_survey is not None else config
    return OmegaConf.merge(default_survey, survey_config)


def resolve_repo_relative_path(path_str: str) -> str:
    """Resolve a config path relative to the repository root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())
