"""Thin experiment-agent config layer backed by ``src/config/default.yaml``."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from openai import AsyncOpenAI

from src.config import get_experiment_config, to_container


API_PROVIDER: str = "openai"
MINIMAX_MODEL_EXTRA_BODY: dict = {"reasoning_split": True}
MINIMAX_MODELS: list[str] = ["MiniMax-M2.1", "MiniMax-M2.5", "MiniMax-M2.7", "MiniMax-Text-01"]


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


def _experiment_dict() -> Dict[str, Any]:
    return dict(to_container(get_experiment_config(), resolve=True) or {})


def _api_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("api", {}) or {})


def _models_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("models", {}) or {})


def _execution_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("execution", {}) or {})


def _memory_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("memory", {}) or {})


def _workspace_cfg() -> Dict[str, Any]:
    return dict(_experiment_dict().get("workspace", {}) or {})


def normalize_workspace_path(path: str) -> str:
    raw = os.path.abspath(os.path.expanduser(path))
    if raw.startswith("/aistor/"):
        return os.path.realpath(raw)
    aistor_candidate = os.path.join("/aistor", raw.lstrip("/"))
    if os.path.exists(aistor_candidate):
        return os.path.realpath(aistor_candidate)
    return os.path.realpath(raw)


def get_api_config() -> Dict[str, str]:
    cfg = _api_cfg()
    openai_api_base = str(
        cfg.get("openai_api_base")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    )
    return {
        "openai_api_key": str(cfg.get("openai_api_key") or ""),
        "openai_api_base": openai_api_base,
        "minimax_api_key": str(cfg.get("minimax_api_key") or ""),
        "minimax_api_base": str(cfg.get("minimax_api_base") or "https://api.minimaxi.com/v1"),
        "serper_api_key": str(cfg.get("serper_api_key") or ""),
        "jina_api_key": str(cfg.get("jina_api_key") or ""),
        "github_ai_token": str(cfg.get("github_ai_token") or ""),
        "huggingface_endpoint": str(cfg.get("huggingface_endpoint") or "https://huggingface.co"),
    }


def get_models_config() -> Dict[str, str]:
    cfg = _models_cfg()
    return {
        "prepare": str(cfg.get("prepare") or "MiniMax-M2.5"),
        "code": str(cfg.get("code") or "MiniMax-M2.5"),
        "master": str(cfg.get("master") or "MiniMax-M2.5"),
        "science": str(cfg.get("science") or "MiniMax-M2.5"),
        "default": str(cfg.get("default") or "MiniMax-M2.5"),
    }


def get_prepare_agent_model() -> str:
    return get_models_config()["prepare"]


def get_code_agent_model() -> str:
    return get_models_config()["code"]


def get_master_agent_model() -> str:
    return get_models_config()["master"]


def get_science_agent_model() -> str:
    return get_models_config()["science"]


def get_default_model_name() -> str:
    return get_models_config()["default"]


def get_execution_config() -> Dict[str, Any]:
    cfg = _execution_cfg()
    return {
        "max_iterations": _as_int(cfg.get("max_iterations"), 20),
        "delegate_max_children": _as_int(cfg.get("delegate_max_children"), 1),
        "planner_max_turns": _as_int(cfg.get("planner_max_turns"), 10000),
        "worker_max_turns": _as_int(cfg.get("worker_max_turns"), 4000),
        "prepare_validation_feedback_rounds": _as_int(
            cfg.get("prepare_validation_feedback_rounds"), 4
        ),
        "code_validation_feedback_rounds": _as_int(
            cfg.get("code_validation_feedback_rounds"), 2
        ),
        "science_validation_feedback_rounds": _as_int(
            cfg.get("science_validation_feedback_rounds"), 2
        ),
        "bash_timeout_seconds": _as_int(cfg.get("bash_timeout_seconds"), 600000),
        "mcp_timeout_seconds": _as_int(cfg.get("mcp_timeout_seconds"), 120),
    }


def get_science_max_iterations() -> int:
    return get_execution_config()["max_iterations"]


def get_delegate_max_children() -> int:
    return get_execution_config()["delegate_max_children"]


def get_planner_max_turns() -> int:
    return get_execution_config()["planner_max_turns"]


def get_worker_max_turns() -> int:
    return get_execution_config()["worker_max_turns"]


def get_prepare_validation_feedback_rounds() -> int:
    return get_execution_config()["prepare_validation_feedback_rounds"]


def get_code_validation_feedback_rounds() -> int:
    return get_execution_config()["code_validation_feedback_rounds"]


def get_science_validation_feedback_rounds() -> int:
    return get_execution_config()["science_validation_feedback_rounds"]


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
    return {
        "root": normalize_workspace_path(str(cfg.get("root") or default_root)),
        "prepare_clone_depth": _as_int(cfg.get("prepare_clone_depth"), 1),
        "model_candidate_seed": normalize_workspace_path(
            str(cfg.get("model_candidate_seed") or "/public/share/model")
        ),
        "tavily_enabled": _as_bool(cfg.get("tavily_enabled"), True),
        "tavily_api_key": str(cfg.get("tavily_api_key") or os.environ.get("TAVILY_API_KEY") or ""),
        "tavily_remote_url_template": str(
            cfg.get("tavily_remote_url_template")
            or "https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
        ),
    }


OPENAI_API_KEY: str = get_api_config()["openai_api_key"]
OPENAI_API_BASE: str = get_api_config()["openai_api_base"]
MINIMAX_API_KEY: str = get_api_config()["minimax_api_key"]
MINIMAX_API_BASE: str = get_api_config()["minimax_api_base"]
SERPER_API_KEY: str = get_api_config()["serper_api_key"]
JINA_API_KEY: str = get_api_config()["jina_api_key"]
GITHUB_AI_TOKEN: str = get_api_config()["github_ai_token"]

CODE_AGENT_MODEL: str = get_code_agent_model()
PREPARE_AGENT_MODEL: str = get_prepare_agent_model()
MASTER_AGENT_MODEL: str = get_master_agent_model()
SCIENCE_AGENT_MODEL: str = get_science_agent_model()
DEFAULT_MODEL: str = get_default_model_name()

SCIENCE_MAX_ITERATIONS: int = get_science_max_iterations()
DELEGATE_MAX_CHILDREN: int = get_delegate_max_children()
PLANNER_MAX_TURNS: int = get_planner_max_turns()
WORKER_MAX_TURNS: int = get_worker_max_turns()
PREPARE_VALIDATION_FEEDBACK_ROUNDS: int = get_prepare_validation_feedback_rounds()
CODE_VALIDATION_FEEDBACK_ROUNDS: int = get_code_validation_feedback_rounds()
SCIENCE_VALIDATION_FEEDBACK_ROUNDS: int = get_science_validation_feedback_rounds()

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
if "OPENHANDS_MCP_TIMEOUT" not in os.environ:
    os.environ["OPENHANDS_MCP_TIMEOUT"] = str(
        get_execution_config()["mcp_timeout_seconds"]
    )


def get_workspace_dir(experiment_id: str) -> str:
    if "EXPERIMENT_AGENT_WORKSPACE_DIR" in os.environ:
        return normalize_workspace_path(os.environ["EXPERIMENT_AGENT_WORKSPACE_DIR"])
    if "CODEAGENT_WORKSPACES_DIR" in os.environ:
        return normalize_workspace_path(os.environ["CODEAGENT_WORKSPACES_DIR"])
    return normalize_workspace_path(os.path.join(BASE_WORKSPACES_DIR, experiment_id))


def get_project_dir(experiment_id: str) -> str:
    return os.path.join(get_workspace_dir(experiment_id), "project")


def get_idea_input_path(experiment_id: str) -> str:
    workspace = get_workspace_dir(experiment_id)
    agent_md_path = os.path.join(workspace, "agent_reports", "prepare_idea.md")
    md_path = os.path.join(workspace, "idea.md")
    json_path = os.path.join(workspace, "idea.json")
    result_json_path = os.path.join(workspace, "idea_result.json")
    if os.path.exists(agent_md_path):
        return agent_md_path
    if os.path.exists(md_path):
        return md_path
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
    return os.path.join(project_root, "venv")


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
    seed_real = os.path.realpath(os.path.abspath(os.path.expanduser(seed_path)))
    share_link = os.path.join(model_dir_real, "model_share")

    if not os.path.exists(seed_real):
        raise FileNotFoundError(f"model_candidate_seed does not exist: {seed_real}")

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
    else:
        os.makedirs(model_dir_real, exist_ok=True)

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
    os.makedirs(os.path.join(paths["workspace_dir"], "specs"), exist_ok=True)
    os.makedirs(os.path.join(paths["workspace_dir"], "templates"), exist_ok=True)
    os.makedirs(os.path.join(paths["results_dir"], "standard"), exist_ok=True)
    os.makedirs(os.path.join(paths["results_dir"], "ablation"), exist_ok=True)
    return paths


def copy_prepared_data_to_workspace(workspace_dir: str) -> None:
    source_dir = Path(__file__).resolve().parents[3] / "data" / "prepared"
    destination_dir = Path(workspace_dir)
    for item in source_dir.iterdir():
        destination = destination_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)


def write_workspace_env_file(experiment_id: str) -> str:
    """Write API configuration to workspace .env file for generated code to use."""
    from .runtime.manifests import write_env_file

    paths = get_path_config(experiment_id)
    env_path = os.path.join(paths["workspace_dir"], ".env")

    api_cfg = get_api_config()
    env_vars = {}
    if api_cfg.get("openai_api_key"):
        env_vars["OPENAI_API_KEY"] = api_cfg["openai_api_key"]
    if api_cfg.get("openai_api_base"):
        env_vars["OPENAI_BASE_URL"] = api_cfg["openai_api_base"]
        env_vars["OPENAI_API_BASE"] = api_cfg["openai_api_base"]
    if api_cfg.get("minimax_api_key"):
        env_vars["MINIMAX_API_KEY"] = api_cfg["minimax_api_key"]
    if api_cfg.get("minimax_api_base"):
        env_vars["MINIMAX_API_BASE"] = api_cfg["minimax_api_base"]

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


def is_minimax_model(model_name: str) -> bool:
    if not model_name:
        return False
    return any(m.lower() in str(model_name).lower() for m in MINIMAX_MODELS)


def ensure_minimax_no_proxy_env(base_url: Optional[str] = None) -> None:
    host = urlparse(str(base_url or MINIMAX_API_BASE or "")).hostname
    if not host:
        return
    for env_name in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(env_name, "")
        entries = [item.strip() for item in current.split(",") if item.strip()]
        if host not in entries:
            entries.append(host)
            os.environ[env_name] = ",".join(entries)


def get_openai_config(model: Optional[str] = None) -> Dict[str, Any]:
    model_name = str(model or get_default_model_name() or "").strip()
    if is_minimax_model(model_name):
        ensure_minimax_no_proxy_env(MINIMAX_API_BASE)
        return {
            "api_key": get_api_config()["minimax_api_key"],
            "model_name": model_name,
            "base_url": get_api_config()["minimax_api_base"],
            "extra_body": MINIMAX_MODEL_EXTRA_BODY,
            "provider_prefix": "minimax",
            "is_minimax": True,
        }
    cfg: Dict[str, Any] = {
        "api_key": get_api_config()["openai_api_key"],
        "model_name": model_name,
        "provider_prefix": "",
        "is_minimax": False,
    }
    openai_base = get_api_config()["openai_api_base"]
    if openai_base:
        cfg["base_url"] = openai_base
    return cfg


def get_llm_config(model: Optional[str] = None) -> Dict[str, Any]:
    return get_openai_config(model)


def get_model_config() -> Dict[str, Any]:
    return {
        "prepare": {"agent": get_prepare_agent_model()},
        "code": {"agent": get_code_agent_model()},
        "science": {"agent": get_science_agent_model()},
        "default": get_default_model_name(),
    }


def setup_openai_api(model: Optional[str] = None, verbose: bool = True) -> bool:
    try:
        from httpx import Timeout

        cfg = get_openai_config(model)
        if cfg.get("is_minimax"):
            ensure_minimax_no_proxy_env(cfg.get("base_url"))
        client_kwargs = {
            "api_key": cfg["api_key"],
            "timeout": Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            "max_retries": 10,
        }
        if cfg.get("base_url"):
            client_kwargs["base_url"] = cfg["base_url"]
            if verbose:
                print(f"  Using API base: {cfg['base_url']}")
        AsyncOpenAI(**client_kwargs)
        if verbose:
            print("  ✓ OpenAI API setup completed")
        return True
    except Exception as exc:
        if verbose:
            print(f"  ✗ Failed to setup OpenAI API: {exc}")
        return False


def validate_config() -> tuple[bool, List[str]]:
    errors: List[str] = []
    if not get_default_model_name():
        errors.append("DEFAULT_MODEL is not set")
    api_cfg = get_api_config()
    if not api_cfg["openai_api_key"] and not api_cfg["minimax_api_key"]:
        errors.append("Either experiment.api.openai_api_key or experiment.api.minimax_api_key must be set")
    for name, value in {
        "PREPARE_AGENT_MODEL": get_prepare_agent_model(),
        "CODE_AGENT_MODEL": get_code_agent_model(),
        "MASTER_AGENT_MODEL": get_master_agent_model(),
        "SCIENCE_AGENT_MODEL": get_science_agent_model(),
    }.items():
        if not value:
            errors.append(f"{name} is not set")
    return len(errors) == 0, errors


def print_config() -> None:
    print("=" * 60)
    print("Experiment Agent Configuration")
    print("=" * 60)
    print("\n[API Configuration]")
    print(f"  API Provider: {API_PROVIDER}")
    api_cfg = get_api_config()
    print(f"  OpenAI API Key: {'*' * 10}...{api_cfg['openai_api_key'][-4:] if api_cfg['openai_api_key'] else 'NOT SET'}")
    if api_cfg["openai_api_base"]:
        print(f"  OpenAI API Base: {api_cfg['openai_api_base']}")
    print(f"  MiniMax API Key: {'*' * 10}...{api_cfg['minimax_api_key'][-4:] if api_cfg['minimax_api_key'] else 'NOT SET'}")
    print(f"  MiniMax API Base: {api_cfg['minimax_api_base']}")
    print(f"  GitHub AI Token: {'*' * 10}...{api_cfg['github_ai_token'][-4:] if api_cfg['github_ai_token'] else 'NOT SET'}")
    print("\n[Web Search Configuration]")
    print(f"  Serper API Key: {'*' * 10}...{api_cfg['serper_api_key'][-4:] if api_cfg['serper_api_key'] else 'NOT SET'}")
    print(f"  Jina API Key: {'*' * 10}...{api_cfg['jina_api_key'][-4:] if api_cfg['jina_api_key'] else 'NOT SET'}")
    print("\n[Model Configuration]")
    print(f"  Prepare: {get_prepare_agent_model()}")
    print(f"  Code: {get_code_agent_model()}")
    print(f"  Master: {get_master_agent_model()}")
    print(f"  Science: {get_science_agent_model()}")
    print(f"  Default: {get_default_model_name()}")
    print("\n[Workspace Configuration]")
    print(f"  Base Workspaces Dir: {get_workspace_config()['root']}")
    print(f"  Model Candidate Seed: {get_workspace_config()['model_candidate_seed']}")
    print(f"  Tavily Enabled: {get_workspace_config()['tavily_enabled']}")
    if WORKSPACE_ROOT:
        print(f"  Current Workspace: {WORKSPACE_ROOT}")
    if PROJECT_ROOT:
        print(f"  Current Project: {PROJECT_ROOT}")
    print("\n[Execution Configuration]")
    print(f"  Delegate Max Children: {get_delegate_max_children()}")
    print(f"  Planner Max Turns: {get_planner_max_turns()}")
    print(f"  Worker Max Turns: {get_worker_max_turns()}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
    ok, errors = validate_config()
    if ok:
        print("\n✓ Configuration is valid!")
    else:
        print("\n✗ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
