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

    config_container = OmegaConf.to_container(config, resolve=False)
    if not isinstance(config_container, dict):
        raise TypeError("Survey config must be a mapping")

    top_level_config = OmegaConf.create(
        {key: value for key, value in config_container.items() if key != "survey"}
    )
    survey_overrides = OmegaConf.select(config, "survey")

    return OmegaConf.merge(
        _resolve_project_placeholders(default_survey),
        _resolve_project_placeholders(top_level_config),
        _resolve_project_placeholders(survey_overrides) if survey_overrides is not None else OmegaConf.create(),
    )


def resolve_repo_relative_path(path_str: str) -> str:
    """Resolve a config path relative to the repository root."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())
