"""OpenHarness-oriented experiment-agent configuration helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from src.config import get_experiment_config, to_container


BACKEND_OPENHARNESS = "openharness"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _first_configured(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return None


def _experiment_dict() -> Dict[str, Any]:
    return dict(to_container(get_experiment_config(), resolve=True) or {})


def _openharness_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("openharness", {}) or {})


def _external_tools_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("external_tools", {}) or {})


def _execution_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("execution", {}) or {})


def _memory_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("memory", {}) or {})


def _workspace_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("workspace", {}) or {})


def _api_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("api", {}) or {})


def normalize_workspace_path(path: str) -> str:
    raw = os.path.abspath(os.path.expanduser(path))
    if raw.startswith("/aistor/"):
        return os.path.realpath(raw)
    aistor_candidate = os.path.join("/aistor", raw.lstrip("/"))
    if os.path.exists(aistor_candidate):
        return os.path.realpath(aistor_candidate)
    return os.path.realpath(raw)


def get_backend_name() -> str:
    return str(_experiment_dict().get("backend") or BACKEND_OPENHARNESS).strip() or BACKEND_OPENHARNESS


def get_openharness_role_models() -> Dict[str, str]:
    raw = dict(_openharness_cfg().get("role_models", {}) or {})
    api_cfg = _api_cfg()
    default_large = (
        raw.get("default")
        or _openharness_cfg().get("default_model")
        or api_cfg.get("large_model")
        or api_cfg.get("default_model")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-5.4"
    )
    default_worker = (
        raw.get("worker")
        or _openharness_cfg().get("worker_model")
        or api_cfg.get("mini_model")
        or api_cfg.get("default_model")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-5-mini"
    )
    defaults = {
        "planner": str(default_large),
        "worker": str(default_worker),
        "reviewer": str(default_large),
        "master": str(default_large),
    }
    role_models = {role: str(model).strip() for role, model in raw.items() if str(model).strip()}
    defaults.update(role_models)
    return defaults


def get_openharness_default_model() -> str:
    cfg = _openharness_cfg()
    return str(
        cfg.get("default_model")
        or _api_cfg().get("large_model")
        or _api_cfg().get("default_model")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-5.4"
    ).strip() or "gpt-5.4"


def get_openharness_config() -> Dict[str, Any]:
    cfg = _openharness_cfg()
    api_cfg = _api_cfg()
    return {
        "default_model": get_openharness_default_model(),
        "role_models": get_openharness_role_models(),
        "api_key": str(
            cfg.get("api_key")
            or api_cfg.get("openai_api_key")
            or api_cfg.get("api_key")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENHARNESS_OPENAI_API_KEY")
            or ""
        ).strip(),
        "base_url": str(
            cfg.get("base_url")
            or api_cfg.get("openai_api_base")
            or api_cfg.get("base_url")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ).strip(),
        "timeout_seconds": _as_int(
            _first_configured(cfg.get("timeout_seconds"), api_cfg.get("timeout_sec")),
            900,
        ),
        "request_max_retries": _as_int(cfg.get("request_max_retries"), 6),
        "request_retry_base_delay": _as_float(cfg.get("request_retry_base_delay"), 1.0),
        "request_retry_max_delay": _as_float(cfg.get("request_retry_max_delay"), 60.0),
        "structured_output_max_hook_blocks": _as_int(cfg.get("structured_output_max_hook_blocks"), 6),
        "max_tokens": _as_int(cfg.get("max_tokens"), 16384),
        "max_turns": _as_int(cfg.get("max_turns"), get_execution_config()["worker_max_turns"]),
        "context_window_tokens": (
            _as_int(cfg.get("context_window_tokens"), 0)
            if cfg.get("context_window_tokens") is not None
            else None
        ),
        "auto_compact_threshold_tokens": (
            _as_int(cfg.get("auto_compact_threshold_tokens"), 0)
            if cfg.get("auto_compact_threshold_tokens") is not None
            else None
        ),
        "runtime_dir_name": str(cfg.get("runtime_dir_name") or ".openharness_runtime"),
        "max_budget_usd": float(cfg.get("max_budget_usd") or 0),
    }


def get_external_tool_config() -> Dict[str, str]:
    cfg = _external_tools_cfg()
    return {
        "huggingface_endpoint": str(
            cfg.get("huggingface_endpoint")
            or os.environ.get("HF_ENDPOINT")
            or "https://hf-mirror.com"
        ),
        "github_ai_token": str(
            cfg.get("github_ai_token")
            or os.environ.get("GITHUB_AI_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or ""
        ),
        "serper_api_key": str(
            cfg.get("serper_api_key")
            or os.environ.get("SERPER_API_KEY")
            or ""
        ),
        "jina_api_key": str(
            cfg.get("jina_api_key")
            or os.environ.get("JINA_API_KEY")
            or ""
        ),
        "tavily_api_key": get_workspace_config()["tavily_api_key"],
        "tavily_remote_url_template": get_workspace_config()["tavily_remote_url_template"],
        "tavily_enabled": str(get_workspace_config()["tavily_enabled"]),
    }


def get_api_config() -> Dict[str, str]:
    return get_external_tool_config()


def get_models_config() -> Dict[str, str]:
    role_models = get_openharness_role_models()
    return {
        "prepare": role_models["planner"],
        "code": role_models["planner"],
        "master": role_models["master"],
        "science": role_models["planner"],
        "default": get_openharness_default_model(),
    }


_AGENT_ROLE_ALIASES: Dict[str, List[str]] = {
    "prepareagent": ["planner"],
    "prepare_agent": ["planner"],
    "prepare_repo_worker": ["worker"],
    "prepare_env_worker": ["worker"],
    "prepare_dataset_worker": ["worker"],
    "prepare_model_worker": ["worker"],
    "prepare_synthesis_worker": ["worker"],
    "prepare_reviewer": ["reviewer"],
    "code": ["planner"],
    "code_agent": ["planner"],
    "code_worker": ["worker"],
    "code_reviewer": ["reviewer"],
    "master": ["master"],
    "master_agent": ["master"],
    "science": ["planner"],
    "science_agent": ["planner"],
    "science_worker": ["worker"],
    "science_reviewer": ["reviewer"],
    "finalization": ["worker"],
    "finalization_agent": ["worker"],
    "finalization_worker": ["worker"],
}


def _normalize_agent_model_key(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def get_agent_model(agent_name: str, phase_hint: Optional[str] = None) -> str:
    role_models = get_openharness_role_models()
    normalized = _normalize_agent_model_key(agent_name)
    candidates = list(_AGENT_ROLE_ALIASES.get(normalized, []))
    if phase_hint:
        phase_key = _normalize_agent_model_key(phase_hint)
        if phase_key:
            if phase_key in role_models:
                candidates.append(phase_key)
            elif phase_key in {"prepare", "code", "science"}:
                candidates.append("planner")
            elif phase_key == "master":
                candidates.append("master")
    for role in candidates:
        value = str(role_models.get(role) or "").strip()
        if value:
            return value
    return get_openharness_default_model()


def get_prepare_agent_model() -> str:
    return get_agent_model("prepare_agent", "prepare")


def get_code_agent_model() -> str:
    return get_agent_model("code_agent", "code")


def get_master_agent_model() -> str:
    return get_agent_model("master_agent", "master")


def get_science_agent_model() -> str:
    return get_agent_model("science_agent", "science")


def get_default_model_name() -> str:
    return get_openharness_default_model()


def get_agent_models_config() -> Dict[str, str]:
    return dict(get_openharness_role_models())


def get_execution_config() -> Dict[str, Any]:
    cfg = _execution_cfg()
    return {
        "delegate_max_children": _as_int(cfg.get("delegate_max_children"), 1),
        "planner_max_turns": _as_int(cfg.get("planner_max_turns"), 0),
        "worker_max_turns": _as_int(cfg.get("worker_max_turns"), 0),
        "bash_timeout_seconds": _as_int(cfg.get("bash_timeout_seconds"), 600000),
        "mcp_timeout_seconds": _as_int(cfg.get("mcp_timeout_seconds"), 120),
    }
def get_delegate_max_children() -> int:
    return get_execution_config()["delegate_max_children"]


def get_planner_max_turns() -> int:
    return get_execution_config()["planner_max_turns"]


def get_worker_max_turns() -> int:
    return get_execution_config()["worker_max_turns"]


def get_memory_config() -> Dict[str, Any]:
    cfg = _memory_cfg()
    return {
        "enabled": _as_bool(cfg.get("enabled"), True),
        "shared_dir": os.path.abspath(
            os.path.expanduser(
                str(cfg.get("shared_dir") or "~/.researchagent/shared_memory")
            )
        ),
        "embedding_model_path": str(
            cfg.get("embedding_model_path")
            or "/hpc_stor03/sjtu_home/hanqi.li/ckpts/huggingface/all-MiniLM-L6-v2"
        ),
        "llm_name": str(cfg.get("llm_name") or "gpt-5-mini"),
        "query_method": str(cfg.get("query_method") or "embedding").strip().lower(),
        "writeback_enabled": _as_bool(cfg.get("writeback_enabled"), True),
        "tool_logs_enabled": _as_bool(cfg.get("tool_logs_enabled"), False),
        "prompt_injection_enabled": _as_bool(cfg.get("prompt_injection_enabled"), True),
        "max_slots_per_task": _as_int(cfg.get("max_slots_per_task"), 100),
    }


def get_workspace_config() -> Dict[str, Any]:
    cfg = _workspace_cfg()
    default_root = Path(__file__).resolve().parents[3] / "workspace"
    raw_seed = str(cfg.get("model_candidate_seed") or "").strip()
    return {
        "root": normalize_workspace_path(str(cfg.get("root") or default_root)),
        "prepare_clone_depth": _as_int(cfg.get("prepare_clone_depth"), 1),
        "model_candidate_seed": normalize_workspace_path(raw_seed) if raw_seed else "",
        "tavily_enabled": _as_bool(cfg.get("tavily_enabled"), True),
        "tavily_api_key": str(cfg.get("tavily_api_key") or os.environ.get("TAVILY_API_KEY") or ""),
        "tavily_remote_url_template": str(
            cfg.get("tavily_remote_url_template")
            or "https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
        ),
    }


DEFAULT_MODEL: str = get_default_model_name()
CODE_AGENT_MODEL: str = get_code_agent_model()
PREPARE_AGENT_MODEL: str = get_prepare_agent_model()
MASTER_AGENT_MODEL: str = get_master_agent_model()
SCIENCE_AGENT_MODEL: str = get_science_agent_model()

DELEGATE_MAX_CHILDREN: int = get_delegate_max_children()
PLANNER_MAX_TURNS: int = get_planner_max_turns()
WORKER_MAX_TURNS: int = get_worker_max_turns()

MEMORY_ENABLED: bool = get_memory_config()["enabled"]
MEMORY_SHARED_DIR: str = get_memory_config()["shared_dir"]
MEMORY_EMBEDDING_MODEL_PATH: str = get_memory_config()["embedding_model_path"]
MEMORY_LLM_NAME: str = get_memory_config()["llm_name"]
MEMORY_QUERY_METHOD: str = get_memory_config()["query_method"]
MEMORY_WRITEBACK_ENABLED: bool = get_memory_config()["writeback_enabled"]
MEMORY_TOOL_LOGS_ENABLED: bool = get_memory_config()["tool_logs_enabled"]
MEMORY_PROMPT_INJECTION_ENABLED: bool = get_memory_config()["prompt_injection_enabled"]
MEMORY_MAX_SLOTS_PER_TASK: int = get_memory_config()["max_slots_per_task"]

BASE_WORKSPACES_DIR: str = get_workspace_config()["root"]
WORKSPACE_ROOT: str = ""
PROJECT_ROOT: str = ""
ENABLE_TRACING: bool = False
LOG_LEVEL: str = "INFO"
COLORED_LOGS: bool = True
VERBOSE_OUTPUT: bool = True


if "AGENT_BASH_TIMEOUT_SECONDS" not in os.environ:
    os.environ["AGENT_BASH_TIMEOUT_SECONDS"] = str(
        get_execution_config()["bash_timeout_seconds"]
    )


def get_workspace_dir(experiment_id: str) -> str:
    if "EXPERIMENT_AGENT_WORKSPACE_DIR" in os.environ:
        return normalize_workspace_path(os.environ["EXPERIMENT_AGENT_WORKSPACE_DIR"])
    return normalize_workspace_path(os.path.join(get_workspace_config()["root"], experiment_id))


def get_project_dir(experiment_id: str) -> str:
    return os.path.join(get_workspace_dir(experiment_id), "project")


def get_idea_input_path(experiment_id: str) -> str:
    from .runtime.manifests import resolve_prepare_idea_path

    workspace = get_workspace_dir(experiment_id)
    agent_md_path = resolve_prepare_idea_path(workspace)
    json_path = os.path.join(workspace, "idea.json")
    result_json_path = os.path.join(workspace, "idea_result.json")
    if os.path.exists(agent_md_path):
        return agent_md_path
    if os.path.exists(json_path):
        return json_path
    if os.path.exists(result_json_path):
        return result_json_path
    return json_path


def get_logs_dir(experiment_id: str) -> str:
    return os.path.join(get_workspace_dir(experiment_id), "logs")


def get_cache_dir(experiment_id: str) -> str:
    return os.path.join(get_workspace_dir(experiment_id), "cached")


def get_dataset_dir(experiment_id: str) -> str:
    return os.path.join(get_workspace_dir(experiment_id), "dataset_candidate")


def get_model_dir(experiment_id: str) -> str:
    return os.path.join(get_workspace_dir(experiment_id), "model_candidate")


def get_model_share_dir(experiment_id: str) -> str:
    return os.path.join(get_model_dir(experiment_id), "model_share")


def get_model_candidate_seed() -> str:
    return normalize_workspace_path(get_workspace_config()["model_candidate_seed"])


def get_results_dir(experiment_id: str) -> str:
    return os.path.join(get_workspace_dir(experiment_id), "results")


def get_reports_dir(experiment_id: str) -> str:
    return os.path.join(get_workspace_dir(experiment_id), "agent_reports")


def get_blueprint_path(experiment_id: str) -> str:
    return os.path.join(get_project_dir(experiment_id), "_blueprint.json")


def get_repos_dir(experiment_id: str) -> str:
    return os.path.join(get_workspace_dir(experiment_id), "repos")


def get_reference_repos(experiment_id: str) -> List[str]:
    repos_dir = get_repos_dir(experiment_id)
    if not os.path.isdir(repos_dir):
        return []
    return sorted(
        os.path.join(repos_dir, name)
        for name in os.listdir(repos_dir)
        if name and not name.startswith(".") and os.path.isdir(os.path.join(repos_dir, name))
    )


def get_venv_path(project_root: str) -> str:
    return os.path.join(project_root, ".venv")


def get_venv_python(project_root: str) -> str:
    return os.path.join(get_venv_path(project_root), "bin", "python")


def get_venv_activate_command(project_root: str) -> str:
    return f"source {os.path.join(get_venv_path(project_root), 'bin', 'activate')}"


def wrap_command_with_venv(command: str, project_root: str) -> str:
    return f"{get_venv_activate_command(project_root)} && {command}"


def get_path_config(experiment_id: str) -> Dict[str, Any]:
    workspace_dir = get_workspace_dir(experiment_id)
    project_dir = get_project_dir(experiment_id)
    return {
        "workspace_dir": workspace_dir,
        "project_dir": project_dir,
        "idea_input": get_idea_input_path(experiment_id),
        "logs_dir": get_logs_dir(experiment_id),
        "cache_dir": get_cache_dir(experiment_id),
        "dataset_dir": get_dataset_dir(experiment_id),
        "model_dir": get_model_dir(experiment_id),
        "model_share_dir": get_model_share_dir(experiment_id),
        "model_candidate_seed": get_model_candidate_seed(),
        "results_dir": get_results_dir(experiment_id),
        "reports_dir": get_reports_dir(experiment_id),
        "repos_dir": get_repos_dir(experiment_id),
        "reference_repos": get_reference_repos(experiment_id),
        "blueprint_path": get_blueprint_path(experiment_id),
    }


def _ensure_seed_symlink(link_path: str, target_path: str) -> None:
    link_real = os.path.abspath(os.path.expanduser(link_path))
    target_real = os.path.realpath(os.path.abspath(os.path.expanduser(target_path)))
    parent_dir = os.path.dirname(link_real)
    os.makedirs(parent_dir, exist_ok=True)
    if not os.path.exists(target_real):
        raise FileNotFoundError(f"model_candidate_seed does not exist: {target_real}")
    if os.path.lexists(link_real):
        if os.path.islink(link_real):
            current_target = os.path.realpath(
                os.path.join(os.path.dirname(link_real), os.readlink(link_real))
            )
            if current_target == target_real:
                return
            raise RuntimeError(
                f"Refusing to replace model_candidate link {link_real}: points to {current_target}, expected {target_real}"
            )
        raise RuntimeError(
            f"Refusing to replace existing non-symlink model_candidate path: {link_real}"
        )
    os.symlink(target_real, link_real)


def _ensure_model_share_mount(model_dir: str, seed_path: str) -> str:
    model_dir_real = os.path.abspath(os.path.expanduser(model_dir))
    share_link = os.path.join(model_dir_real, "model_share")
    os.makedirs(model_dir_real, exist_ok=True)

    seed_path = str(seed_path or "").strip()
    if not seed_path:
        return share_link
    seed_real = os.path.realpath(os.path.abspath(os.path.expanduser(seed_path)))
    if not os.path.exists(seed_real):
        return share_link

    if os.path.islink(model_dir_real):
        current_target = os.path.realpath(
            os.path.join(os.path.dirname(model_dir_real), os.readlink(model_dir_real))
        )
        if current_target != seed_real:
            raise RuntimeError(
                f"Refusing to replace model_candidate link {model_dir_real}: points to {current_target}, expected {seed_real}"
            )
        os.unlink(model_dir_real)
        os.makedirs(model_dir_real, exist_ok=True)
    elif os.path.exists(model_dir_real) and not os.path.isdir(model_dir_real):
        raise RuntimeError(
            f"Refusing to replace existing non-directory model_candidate path: {model_dir_real}"
        )
    _ensure_seed_symlink(share_link, seed_real)
    return share_link


def ensure_experiment_dirs(experiment_id: str) -> Dict[str, Any]:
    paths = get_path_config(experiment_id)
    for key in (
        "workspace_dir",
        "project_dir",
        "logs_dir",
        "cache_dir",
        "dataset_dir",
        "model_dir",
        "results_dir",
        "reports_dir",
        "repos_dir",
    ):
        os.makedirs(paths[key], exist_ok=True)
    paths["model_share_dir"] = _ensure_model_share_mount(
        paths["model_dir"], paths["model_candidate_seed"]
    )
    os.makedirs(os.path.join(paths["workspace_dir"], "templates"), exist_ok=True)
    os.makedirs(os.path.join(paths["results_dir"], "science"), exist_ok=True)
    from src.agents.experiment_agent.runtime.openharness_runner import (
        ensure_openharness_runtime_env,
    )

    ensure_openharness_runtime_env(paths["workspace_dir"])
    return paths


def copy_prepared_data_to_workspace(workspace_dir: str) -> None:
    """Copy experiment-specific prepared data into workspace's dataset_candidate/.

    Looks for data in data/prepared/<experiment_id>/. The experiment_id is inferred
    from the workspace directory name (e.g. workspace/mlp -> mlp).
    """
    experiment_id = os.path.basename(os.path.normpath(workspace_dir))
    source_dir = Path(__file__).resolve().parents[3] / "data" / "prepared" / experiment_id
    destination_dir = Path(workspace_dir) / "dataset_candidate"
    if not source_dir.is_dir():
        return
    for item in source_dir.iterdir():
        destination = destination_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def write_workspace_env_file(experiment_id: str) -> str:
    """Write a minimal env file for generated project code and tools."""
    from .runtime.manifests import write_env_file

    paths = get_path_config(experiment_id)
    env_path = os.path.join(paths["workspace_dir"], ".env")

    env_vars: Dict[str, str] = {}
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        value = str(os.environ.get(key) or "").strip()
        if value:
            env_vars[key] = value
    env_name_map = {
        "huggingface_endpoint": "HF_ENDPOINT",
        "github_ai_token": "GITHUB_AI_TOKEN",
        "serper_api_key": "SERPER_API_KEY",
        "jina_api_key": "JINA_API_KEY",
    }
    for key, value in get_external_tool_config().items():
        env_name = env_name_map.get(key, key.upper())
        if value:
            env_vars[env_name] = value

    if env_vars:
        write_env_file(env_path, env_vars)

    return env_path


class ProjectContext:
    _instance: Optional["ProjectContext"] = None

    def __init__(self) -> None:
        self.workspace_root = ""
        self.project_root = ""
        self.project_id = ""
        self.reference_repos: List[str] = []
        self.initialized = False

    @classmethod
    def get_instance(cls) -> "ProjectContext":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def initialize(
        cls,
        project_root: str,
        workspace_root: Optional[str] = None,
        project_id: Optional[str] = None,
        reference_repos: Optional[List[str]] = None,
    ) -> "ProjectContext":
        instance = cls.get_instance()
        instance.project_root = os.path.abspath(project_root)
        instance.workspace_root = workspace_root or os.path.dirname(instance.project_root)
        instance.project_id = project_id or os.path.basename(project_root)
        instance.reference_repos = list(reference_repos or [])
        instance.initialized = True

        global WORKSPACE_ROOT, PROJECT_ROOT
        WORKSPACE_ROOT = instance.workspace_root
        PROJECT_ROOT = instance.project_root
        return instance

    def get_cache_dir(self) -> str:
        cache_dir = os.path.join(self.project_root, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def get_logs_dir(self) -> str:
        logs_dir = os.path.join(self.project_root, ".logs")
        os.makedirs(logs_dir, exist_ok=True)
        return logs_dir

    def get_blueprint_path(self) -> str:
        return os.path.join(self.project_root, "_blueprint.json")


_context = ProjectContext.get_instance()


def get_openai_config(model: Optional[str] = None) -> Dict[str, Any]:
    """OpenAI-compatible client settings for memory and utility code."""
    api_cfg = _api_cfg()
    model_name = str(model or os.environ.get("OPENAI_MODEL") or "gpt-5-mini").strip()
    base_url = str(
        os.environ.get("OPENAI_BASE_URL")
        or api_cfg.get("openai_api_base")
        or ""
    ).strip()
    cfg: Dict[str, Any] = {
        "api_key": str(
            os.environ.get("OPENAI_API_KEY") or api_cfg.get("openai_api_key") or ""
        ).strip(),
        "model_name": model_name,
        "provider_prefix": "",
        "is_minimax": False,
    }
    if base_url:
        cfg["base_url"] = base_url
    return cfg


def get_llm_config(model: Optional[str] = None) -> Dict[str, Any]:
    return get_openai_config(model)


def get_model_config() -> Dict[str, Any]:
    return {
        "backend": get_backend_name(),
        "openharness": get_openharness_config(),
        "prepare": {"agent": get_prepare_agent_model()},
        "code": {"agent": get_code_agent_model()},
        "science": {"agent": get_science_agent_model()},
        "default": get_default_model_name(),
        "agents": get_agent_models_config(),
    }


def setup_openai_api(model: Optional[str] = None, verbose: bool = True) -> bool:
    try:
        import httpx

        cfg = get_openai_config(model)
        if not cfg.get("api_key"):
            if verbose:
                print("  ! OPENAI_API_KEY is not set; skipping optional OpenAI client probe")
            return False
        client_kwargs = {
            "api_key": cfg["api_key"],
            "timeout": httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            "max_retries": 10,
            "http_client": httpx.AsyncClient(trust_env=False),
        }
        if cfg.get("base_url"):
            client_kwargs["base_url"] = cfg["base_url"]
            if verbose:
                print(f"  Using API base: {cfg['base_url']}")
        AsyncOpenAI(**client_kwargs)
        if verbose:
            print("  ✓ Optional OpenAI API setup completed")
        return True
    except Exception as exc:
        if verbose:
            print(f"  ✗ Failed to setup optional OpenAI API: {exc}")
        return False


def validate_config() -> tuple[bool, List[str]]:
    errors: List[str] = []
    if get_backend_name() != BACKEND_OPENHARNESS:
        errors.append(f"experiment.backend must be `{BACKEND_OPENHARNESS}`")
    if not get_default_model_name():
        errors.append("experiment.openharness.default_model is not set")
    for name, value in {
        "PREPARE_AGENT_MODEL": get_prepare_agent_model(),
        "CODE_AGENT_MODEL": get_code_agent_model(),
        "MASTER_AGENT_MODEL": get_master_agent_model(),
        "SCIENCE_AGENT_MODEL": get_science_agent_model(),
    }.items():
        if not value:
            errors.append(f"{name} is not set")
    return len(errors) == 0, errors


def _mask_secret(value: str) -> str:
    if not value:
        return "NOT SET"
    suffix = value[-4:] if len(value) >= 4 else value
    return f"{'*' * 10}...{suffix}"


def print_config() -> None:
    from src.agents.experiment_agent.telemetry import print_kv_table

    harness_cfg = get_openharness_config()
    external_cfg = get_external_tool_config()
    workspace_rows = {
        "base_workspaces_dir": get_workspace_config()["root"],
        "model_candidate_seed": get_workspace_config()["model_candidate_seed"] or "(disabled)",
        "tavily_enabled": get_workspace_config()["tavily_enabled"],
    }
    if WORKSPACE_ROOT:
        workspace_rows["current_workspace"] = WORKSPACE_ROOT
    if PROJECT_ROOT:
        workspace_rows["current_project"] = PROJECT_ROOT

    print_kv_table(
        "Experiment Agent Configuration",
        {
            "backend": get_backend_name(),
            "default_model": harness_cfg["default_model"],
            "api_base_url": harness_cfg["base_url"] or "(default OpenAI endpoint)",
            "timeout_seconds": harness_cfg["timeout_seconds"],
            "request_retries": harness_cfg["request_max_retries"],
            "request_retry_base_delay": harness_cfg["request_retry_base_delay"],
            "request_retry_max_delay": harness_cfg["request_retry_max_delay"],
            "structured_output_hook_blocks": harness_cfg["structured_output_max_hook_blocks"],
            "max_tokens": harness_cfg["max_tokens"],
            "runtime_dir": harness_cfg["runtime_dir_name"],
        },
    )
    print_kv_table("Role Models", dict(sorted(harness_cfg["role_models"].items())))
    print_kv_table(
        "External Tools",
        {
            "huggingface_endpoint": external_cfg["huggingface_endpoint"],
            "github_ai_token": external_cfg["github_ai_token"],
            "serper_api_key": external_cfg["serper_api_key"],
            "jina_api_key": external_cfg["jina_api_key"],
        },
    )
    print_kv_table("Workspace", workspace_rows, mask_sensitive=False)
    print_kv_table(
        "Execution",
        {
            "delegate_max_children": get_delegate_max_children(),
            "planner_max_turns": get_planner_max_turns(),
            "worker_max_turns": get_worker_max_turns(),
        },
    )


if __name__ == "__main__":
    print_config()
    ok, errors = validate_config()
    if ok:
        print("\n✓ Configuration is valid!")
    else:
        print("\n✗ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
