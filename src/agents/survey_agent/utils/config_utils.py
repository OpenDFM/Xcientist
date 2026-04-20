from pathlib import Path

from omegaconf import DictConfig, OmegaConf


_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "src" / "config" / "default.yaml"


def _resolve_project_placeholders(config: DictConfig) -> DictConfig:
    repo_root = str(_PROJECT_ROOT.resolve())
    workspace = str((_PROJECT_ROOT / "workspace").resolve())

    def _replace(value):
        if isinstance(value, str):
            return value.replace("${repo_root}", repo_root).replace("${workspace}", workspace)
        if isinstance(value, dict):
            return {key: _replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_replace(item) for item in value]
        return value

    return OmegaConf.create(_replace(OmegaConf.to_container(config, resolve=False)))


def merge_with_default_survey_config(config: DictConfig) -> DictConfig:
    """Merge a survey preset config on top of src/config/default.yaml::survey."""
    default_root = OmegaConf.load(_DEFAULT_CONFIG_PATH)
    default_survey = OmegaConf.select(default_root, "survey")
    if default_survey is None:
        raise ValueError(f"Missing 'survey' section in default config: {_DEFAULT_CONFIG_PATH}")

    current_survey = OmegaConf.select(config, "survey")
    survey_config = current_survey if current_survey is not None else config
    return OmegaConf.merge(
        _resolve_project_placeholders(default_survey),
        _resolve_project_placeholders(survey_config),
    )


def resolve_repo_relative_path(path_str: str) -> str:
    """Resolve a config path relative to the repository root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())
