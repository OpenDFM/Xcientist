"""
Configuration file for CodeAgent System.

Provides:
- API configuration (OpenAI, MiniMax)
- Model configuration for each agent type
- Workspace and path management
- Context initialization
"""

import os
from typing import Optional, List, Dict

from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI


API_PROVIDER: str = "openai"

# OpenAI Configuration
OPENAI_API_KEY: Optional[str] = os.environ.get(
    "OPENAI_API_KEY", "sk-BWZ0Kqbk3PvdF0zRFf69B63901B84e85A5B4D8B1AfE27e2e"
)
OPENAI_API_BASE: Optional[str] = os.environ.get(
    "OPENAI_API_BASE", "https://api.xi-ai.cn/v1"
)

# MiniMax Configuration
MINIMAX_API_KEY: Optional[str] = os.environ.get(
    "MINIMAX_API_KEY",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLkvIEiLCJVc2VyTmFtZSI6IuS8gSIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTk4MjcyNjcyNjM0NTA3NTQ0IiwiUGhvbmUiOiIxODk4NTU0MDc2NiIsIkdyb3VwSUQiOiIxOTk4MjcyNjcyNjMwMzEzMjQwIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMTItMTAgMTQ6MTQ6NTMiLCJUb2tlblR5cGUiOjQsImlzcyI6Im1pbmltYXgifQ.hvIJx5NfyV-53iYcS7AMkwooAK4yLv00ZMW0CojFki_S0qXfBECOFozLVcSVcS_-Lbn1ttS6_ZQmuFOZLzZbMz679Svq_ffebftANne4fUQheFrdWMiI48JBvzVH5aDL85cxyLyLU4zfujrE1tpEkfOWddgASMpSzZmK-uiivOOPJqAoMQI76kyZbuVTIIMjXYmsTKsYpmj83ggnpHFT8E2pmXBnQyL_5IRwDRLyN4VKSRUjSRvjo8z4_QE_f1ubGLThJgnCeb0mS5nVtjg9rGcBHmRsvJoTwLKPSRv8lCaEvGTM9U8UVvOcMIt9Y3BgBT2tuUvDXJt-VGAnw3OfhA",
)
MINIMAX_API_BASE: Optional[str] = "https://api.minimaxi.com/v1"
MINIMAX_MODEL_EXTRA_BODY: dict = {"reasoning_split": True}
MINIMAX_MODELS: list = ["MiniMax-M2.1"]

XIAOMI_API_KEY: str = "sk-c8bwnop3bi1nahlzx98ga7o0kqgr8u9h0bpv6zri28hp2x20"
XIAOMI_API_BASE: str = "https://api.xiaomimimo.com/v1/"
XIAOMI_MODELS: list = ["mimo-v2-flash"]
# MINIMAX_API_KEY: Optional[str] = os.environ.get(
#     "MINIMAX_API_KEY",
#     "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLkvIEiLCJVc2VyTmFtZSI6IuS8gSIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTk4MjcyNjcyNjM0NTA3NTQ0IiwiUGhvbmUiOiIxODk4NTU0MDc2NiIsIkdyb3VwSUQiOiIxOTk4MjcyNjcyNjMwMzEzMjQwIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMTItMTAgMTQ6MTQ6NTMiLCJUb2tlblR5cGUiOjQsImlzcyI6Im1pbmltYXgifQ.hvIJx5NfyV-53iYcS7AMkwooAK4yLv00ZMW0CojFki_S0qXfBECOFozLVcSVcS_-Lbn1ttS6_ZQmuFOZLzZbMz679Svq_ffebftANne4fUQheFrdWMiI48JBvzVH5aDL85cxyLyLU4zfujrE1tpEkfOWddgASMpSzZmK-uiivOOPJqAoMQI76kyZbuVTIIMjXYmsTKsYpmj83ggnpHFT8E2pmXBnQyL_5IRwDRLyN4VKSRUjSRvjo8z4_QE_f1ubGLThJgnCeb0mS5nVtjg9rGcBHmRsvJoTwLKPSRv8lCaEvGTM9U8UVvOcMIt9Y3BgBT2tuUvDXJt-VGAnw3OfhA",
# )
# MINIMAX_API_BASE: Optional[str] = "https://api.minimaxi.com/v1"

# OPENAI_API_KEY: str = "sk-BWZ0Kqbk3PvdF0zRFf69B63901B84e85A5B4D8B1AfE27e2e"
# OPENAI_API_BASE: str = "https://api.xi-ai.cn/v1"

# OPENAI_API_KEY: Optional[str] = os.environ.get(
#     "OPENAI_API_KEY", "sk-Q1Aah6ovHJyPhlmi0yZtNazWo29XiMyBIMtaKZGtG6RzFp2W"
# )
# OPENAI_API_BASE: Optional[str] = os.environ.get(
#     "OPENAI_API_BASE", "https://www.dmxapi.cn/v1"


# Code Layer Models
CODE_ARCHITECT_MODEL: str = "gpt-5.1"
CODE_MANAGER_MODEL: str = "MiniMax-M2.1"
CODE_WORKER_MODEL: str = "MiniMax-M2.1"
CODE_INTEGRATOR_MODEL: str = "MiniMax-M2.1"

# Prepare Layer Models
PREPARE_AGENT_MODEL: str = "MiniMax-M2.1"

# Science Layer Models
SCIENCE_ARCHITECT_MODEL: str = "MiniMax-M2.1"
SCIENCE_MANAGER_MODEL: str = "MiniMax-M2.1"
SCIENCE_WORKER_MODEL: str = "MiniMax-M2.1"
SCIENCE_INTEGRATOR_MODEL: str = "MiniMax-M2.1"

# Default fallback
DEFAULT_MODEL: str = "MiniMax-M2.1"

# Science Layer Configuration
SCIENCE_MAX_ITERATIONS: int = int(os.environ.get("SCIENCE_MAX_ITERATIONS", "5"))

#
MEMORY_ENABLED: bool = os.environ.get(
    "EXPERIMENT_AGENT_MEMORY_ENABLED", "1"
).strip().lower() in ("1", "true", "yes", "y", "on")
MEMORY_SHARED_DIR: str = os.path.abspath(
    os.path.expanduser(
        os.environ.get(
            "EXPERIMENT_AGENT_SHARED_MEMORY_DIR",
            os.path.join(os.path.expanduser("~"), ".researchagent", "shared_memory"),
        )
    )
)
MEMORY_EMBEDDING_MODEL_PATH: str = os.environ.get(
    "EXPERIMENT_AGENT_MEMORY_MODEL_PATH",
    "/hpc_stor03/sjtu_home/hanqi.li/ckpts/huggingface/all-MiniLM-L6-v2",
)
MEMORY_LLM_NAME: str = os.environ.get(
    "EXPERIMENT_AGENT_MEMORY_LLM_NAME", "mimo-v2-flash"
)
MEMORY_QUERY_METHOD: str = (
    os.environ.get("EXPERIMENT_AGENT_MEMORY_QUERY_METHOD", "embedding").strip().lower()
)
MEMORY_WRITEBACK_ENABLED: bool = os.environ.get(
    "EXPERIMENT_AGENT_MEMORY_WRITEBACK", "1"
).strip().lower() in ("1", "true", "yes", "y", "on")
MEMORY_TOOL_LOGS_ENABLED: bool = os.environ.get(
    "EXPERIMENT_AGENT_MEMORY_TOOL_LOGS", "0"
).strip().lower() in ("1", "true", "yes", "y", "on")
MEMORY_PROMPT_INJECTION_ENABLED: bool = os.environ.get(
    "EXPERIMENT_AGENT_MEMORY_PROMPT_INJECTION", "1"
).strip().lower() in ("1", "true", "yes", "y", "on")


MEMORY_MAX_SLOTS_PER_TASK: int = 100


# Base directory for all CodeAgent workspaces
BASE_WORKSPACES_DIR: str = os.environ.get(
    "CODEAGENT_WORKSPACES_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "workspaces",
    ),
)

WORKSPACE_ROOT: str = ""
PROJECT_ROOT: str = ""


# Maximum parallel workers for code implementation
MAX_PARALLEL_WORKERS: int = 5

# Maximum attempts for each file implementation
MAX_IMPLEMENTATION_ATTEMPTS: int = 50

MAX_SCIENCE_ITERATIONS: int = 1

# Maximum tool-calling turns for agents
MAX_AGENT_TURNS: int = 999

MAX_FIX_ITERATIONS = 50

# Enable/disable tracing
ENABLE_TRACING: bool = False

# Timeout settings (in seconds)
SHELL_COMMAND_TIMEOUT: int = 6000000
AGENT_CALL_TIMEOUT: int = 6000000

# Logging
LOG_LEVEL: str = "INFO"
COLORED_LOGS: bool = True
VERBOSE_OUTPUT: bool = True


def get_workspace_dir(experiment_id: str) -> str:
    """
    Get the workspace directory for a specific experiment.

    Args:
        experiment_id: The unique identifier for the experiment

    Returns:
        Path to the experiment workspace directory
    """
    return os.path.join(BASE_WORKSPACES_DIR, experiment_id)


def get_project_dir(experiment_id: str) -> str:
    """Get the project directory for a specific experiment."""
    return os.path.join(get_workspace_dir(experiment_id), "project")


def get_idea_input_path(experiment_id: str) -> str:
    """Get the idea input path for a specific experiment."""
    # Check for both .md and .json formats
    workspace = get_workspace_dir(experiment_id)
    md_path = os.path.join(workspace, "idea.md")
    json_path = os.path.join(workspace, "idea.json")

    if os.path.exists(md_path):
        return md_path
    return json_path


def get_logs_dir(experiment_id: str) -> str:
    """Get the logs directory for a specific experiment."""
    return os.path.join(get_workspace_dir(experiment_id), "logs")


def get_cache_dir(experiment_id: str) -> str:
    """Get the cache directory for a specific experiment."""
    return os.path.join(get_workspace_dir(experiment_id), "cached")


def get_dataset_dir(experiment_id: str) -> str:
    """
    Get the dataset directory for a specific experiment.

    The dataset location is determined by `--experiment` at runtime and should not be hard-coded.
    """
    return os.path.join(get_workspace_dir(experiment_id), "dataset_candidate")


def get_blueprint_path(experiment_id: str) -> str:
    """Get the blueprint file path for a specific experiment."""
    return os.path.join(get_project_dir(experiment_id), "_blueprint.json")


def get_repos_dir(experiment_id: str) -> str:
    """Get the reference repositories directory for a specific experiment."""
    return os.path.join(get_workspace_dir(experiment_id), "repos")


def get_reference_repos(experiment_id: str) -> List[str]:
    """
    Get list of reference repository paths for a specific experiment.

    Automatically discovers all subdirectories in the repos/ folder.

    Args:
        experiment_id: The unique identifier for the experiment

    Returns:
        List of paths to reference repositories
    """
    repos_dir = get_repos_dir(experiment_id)

    if not os.path.exists(repos_dir):
        return []

    repos = []
    for item in os.listdir(repos_dir):
        item_path = os.path.join(repos_dir, item)
        # Only include directories (each subdirectory is a repo)
        if os.path.isdir(item_path):
            repos.append(item_path)

    return sorted(repos)


def get_venv_path(project_root: str) -> str:
    """Get the virtual environment path for a project."""
    return os.path.join(project_root, "venv")


def get_venv_python(project_root: str) -> str:
    """Get the Python executable path in venv."""
    venv_path = get_venv_path(project_root)
    return os.path.join(venv_path, "bin", "python")


def get_venv_activate_command(project_root: str) -> str:
    """
    Get the command to activate venv (for use in bash commands).

    Returns a source command that activates the venv.
    Example: "source /path/to/project/venv/bin/activate"
    """
    venv_path = get_venv_path(project_root)
    activate_script = os.path.join(venv_path, "bin", "activate")
    return f"source {activate_script}"


def wrap_command_with_venv(command: str, project_root: str) -> str:
    """
    Wrap a command to run within the project's venv.

    Args:
        command: The command to execute
        project_root: Path to the project directory

    Returns:
        Command wrapped with venv activation

    Example:
        Input: "python train.py --epochs 10"
        Output: "source /path/to/venv/bin/activate && python train.py --epochs 10"
    """
    activate_cmd = get_venv_activate_command(project_root)
    return f"{activate_cmd} && {command}"


def get_path_config(experiment_id: str) -> dict:
    """
    Get path configuration dictionary for a specific experiment.

    Args:
        experiment_id: The unique identifier for the experiment

    Returns:
        Dictionary with path configuration for the experiment
    """
    workspace_dir = get_workspace_dir(experiment_id)
    project_dir = get_project_dir(experiment_id)

    return {
        "workspace_dir": workspace_dir,
        "project_dir": project_dir,
        "idea_input": get_idea_input_path(experiment_id),
        "logs_dir": get_logs_dir(experiment_id),
        "cache_dir": get_cache_dir(experiment_id),
        "dataset_dir": get_dataset_dir(experiment_id),
        "repos_dir": get_repos_dir(experiment_id),
        "reference_repos": get_reference_repos(experiment_id),
        "blueprint_path": get_blueprint_path(experiment_id),
    }


def ensure_experiment_dirs(experiment_id: str) -> dict:
    """
    Ensure all experiment directories exist.

    Args:
        experiment_id: The unique identifier for the experiment

    Returns:
        Dictionary with all path configurations
    """
    paths = get_path_config(experiment_id)

    # Create directories
    os.makedirs(paths["workspace_dir"], exist_ok=True)
    os.makedirs(paths["project_dir"], exist_ok=True)
    os.makedirs(paths["logs_dir"], exist_ok=True)
    os.makedirs(paths["cache_dir"], exist_ok=True)
    os.makedirs(paths["dataset_dir"], exist_ok=True)
    # Derived directories/files (specs/, templates/, constitution.md) are intentionally NOT
    # returned here anymore. Use per-layer docs helpers instead:
    # - layers/code/docs.py
    # - layers/science/docs.py
    #
    # We still ensure these directories exist for backward compatibility with existing layouts.
    specs_dir = os.path.join(paths["workspace_dir"], "specs")
    templates_dir = os.path.join(paths["workspace_dir"], "templates")
    os.makedirs(specs_dir, exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)

    return paths


class ProjectContext:
    """
    Singleton class to manage project-level context.

    This tracks the current workspace and project directories,
    and provides helper methods for path management.
    """

    _instance = None

    def __init__(self):
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
    ):
        """
        Initialize the project context.

        Args:
            project_root: Root directory of the generated project
            workspace_root: Root directory of the workspace (defaults to parent of project_root)
            project_id: Optional project identifier
            reference_repos: Optional list of reference repository paths
        """
        instance = cls.get_instance()
        instance.project_root = os.path.abspath(project_root)
        instance.workspace_root = workspace_root or os.path.dirname(
            instance.project_root
        )
        instance.project_id = project_id or os.path.basename(project_root)
        instance.reference_repos = reference_repos or []
        instance.initialized = True

        # Update module-level globals
        global WORKSPACE_ROOT, PROJECT_ROOT
        WORKSPACE_ROOT = instance.workspace_root
        PROJECT_ROOT = instance.project_root

        return instance

    def get_cache_dir(self) -> str:
        """Get the cache directory for this project."""
        cache_dir = os.path.join(self.project_root, ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def get_logs_dir(self) -> str:
        """Get the logs directory for this project."""
        logs_dir = os.path.join(self.project_root, ".logs")
        os.makedirs(logs_dir, exist_ok=True)
        return logs_dir

    def get_blueprint_path(self) -> str:
        """Get the path to the blueprint file."""
        return os.path.join(self.project_root, "_blueprint.json")


_context = ProjectContext.get_instance()


def is_minimax_model(model_name: str) -> bool:
    """Check if model is a MiniMax model."""
    if not model_name:
        return False
    return any(m.lower() in model_name.lower() for m in MINIMAX_MODELS)


def is_xiaomi_model(model_name: str) -> bool:
    """Check if model is a Xiaomi model."""
    if not model_name:
        return False
    return any(m.lower() in model_name.lower() for m in XIAOMI_MODELS)


def get_openai_config(model: Optional[str] = None) -> dict:
    """
    Get OpenAI configuration dictionary.

    Args:
        model: Optional model name to use. If not provided, uses DEFAULT_MODEL.

    Returns:
        Dictionary with OpenAI configuration
    """
    model_to_use = model or DEFAULT_MODEL

    if is_minimax_model(model_to_use):
        return {
            "api_key": MINIMAX_API_KEY,
            "model_name": model_to_use,
            "base_url": MINIMAX_API_BASE,
            "extra_body": MINIMAX_MODEL_EXTRA_BODY,
            "is_minimax": True,
        }
    elif is_xiaomi_model(model_to_use):
        return {
            "api_key": XIAOMI_API_KEY,
            "model_name": model_to_use,
            "base_url": XIAOMI_API_BASE,
            "is_xiaomi": True,
        }
    else:
        config = {
            "api_key": OPENAI_API_KEY,
            "model_name": model_to_use,
            "is_minimax": False,
        }
        if OPENAI_API_BASE:
            config["base_url"] = OPENAI_API_BASE
        return config


def get_model_config() -> dict:
    """
    Get model configuration dictionary for all agents.

    Returns:
        Dictionary with model configuration for each agent
    """
    return {
        "prepare": {
            "agent": PREPARE_AGENT_MODEL,
        },
        "code": {
            "architect": CODE_ARCHITECT_MODEL,
            "manager": CODE_MANAGER_MODEL,
            "worker": CODE_WORKER_MODEL,
            "integrator": CODE_INTEGRATOR_MODEL,
        },
        "science": {
            "architect": SCIENCE_ARCHITECT_MODEL,
            "manager": SCIENCE_MANAGER_MODEL,
            "worker": SCIENCE_WORKER_MODEL,
            "integrator": SCIENCE_INTEGRATOR_MODEL,
        },
        "default": DEFAULT_MODEL,
    }


def setup_openai_api(model: Optional[str] = None, verbose: bool = True) -> bool:
    """
    Set up OpenAI API client for the agents library.

    Args:
        model: Optional model name to use for routing (e.g., MiniMax vs OpenAI).
        verbose: Whether to print setup status

    Returns:
        True if setup successful, False otherwise
    """
    try:
        from httpx import Timeout

        config = get_openai_config(model)

        client_kwargs = {
            "api_key": config["api_key"],
            "timeout": Timeout(
                connect=10.0,
                read=300.0,
                write=30.0,
                pool=10.0,
            ),
            "max_retries": 10,
        }

        if "base_url" in config and config["base_url"]:
            client_kwargs["base_url"] = config["base_url"]
            if verbose:
                print(f"  Using API base: {config['base_url']}")

        client = AsyncOpenAI(**client_kwargs)

        set_default_openai_client(client)
        set_default_openai_api("chat_completions")
        set_tracing_disabled(not ENABLE_TRACING)

        if verbose:
            print("  ✓ OpenAI API setup completed")

        return True

    except Exception as e:
        if verbose:
            print(f"  ✗ Failed to setup OpenAI API: {str(e)}")
        return False


def validate_config() -> tuple:
    """
    Validate configuration settings.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set")

    if not DEFAULT_MODEL:
        errors.append("DEFAULT_MODEL is not set")

    # Check each agent's model
    code_models = {
        "CODE_ARCHITECT_MODEL": CODE_ARCHITECT_MODEL,
        "CODE_MANAGER_MODEL": CODE_MANAGER_MODEL,
        "CODE_WORKER_MODEL": CODE_WORKER_MODEL,
        "CODE_INTEGRATOR_MODEL": CODE_INTEGRATOR_MODEL,
    }
    prepare_models = {
        "PREPARE_AGENT_MODEL": PREPARE_AGENT_MODEL,
    }
    science_models = {
        "SCIENCE_ARCHITECT_MODEL": SCIENCE_ARCHITECT_MODEL,
        "SCIENCE_MANAGER_MODEL": SCIENCE_MANAGER_MODEL,
        "SCIENCE_WORKER_MODEL": SCIENCE_WORKER_MODEL,
        "SCIENCE_INTEGRATOR_MODEL": SCIENCE_INTEGRATOR_MODEL,
    }

    for model_name, model_value in {
        **prepare_models,
        **code_models,
        **science_models,
    }.items():
        if not model_value:
            errors.append(f"{model_name} is not set")

    return len(errors) == 0, errors


def print_config():
    """Print current configuration (masking sensitive information)."""
    print("=" * 60)
    print("CodeAgent Configuration")
    print("=" * 60)

    print(f"\n[API Configuration]")
    print(f"  API Provider: {API_PROVIDER}")
    print(
        f"  OpenAI API Key: {'*' * 10}...{OPENAI_API_KEY[-4:] if OPENAI_API_KEY else 'NOT SET'}"
    )
    if OPENAI_API_BASE:
        print(f"  OpenAI API Base: {OPENAI_API_BASE}")

    print(f"\n[Model Configuration]")
    print(f"  Prepare Layer:")
    print(f"    Agent: {PREPARE_AGENT_MODEL}")
    print(f"  Code Layer:")
    print(f"    Architect: {CODE_ARCHITECT_MODEL}")
    print(f"    Manager: {CODE_MANAGER_MODEL}")
    print(f"    Worker: {CODE_WORKER_MODEL}")
    print(f"    Integrator: {CODE_INTEGRATOR_MODEL}")
    print(f"  Science Layer:")
    print(f"    Architect: {SCIENCE_ARCHITECT_MODEL}")
    print(f"    Manager: {SCIENCE_MANAGER_MODEL}")
    print(f"    Worker: {SCIENCE_WORKER_MODEL}")
    print(f"    Integrator: {SCIENCE_INTEGRATOR_MODEL}")
    print(f"  Default: {DEFAULT_MODEL}")

    print(f"\n[Workspace Configuration]")
    print(f"  Base Workspaces Dir: {BASE_WORKSPACES_DIR}")
    if WORKSPACE_ROOT:
        print(f"  Current Workspace: {WORKSPACE_ROOT}")
    if PROJECT_ROOT:
        print(f"  Current Project: {PROJECT_ROOT}")

    print(f"\n[Execution Configuration]")
    print(f"  Max Parallel Workers: {MAX_PARALLEL_WORKERS}")
    print(f"  Max Implementation Attempts: {MAX_IMPLEMENTATION_ATTEMPTS}")
    print(f"  Max Agent Turns: {MAX_AGENT_TURNS}")
    print(f"  Shell Command Timeout: {SHELL_COMMAND_TIMEOUT}s")
    print(f"  Tracing Enabled: {ENABLE_TRACING}")
    print(f"  Verbose Output: {VERBOSE_OUTPUT}")

    print("=" * 60)


if __name__ == "__main__":
    print_config()
    is_valid, errors = validate_config()
    if is_valid:
        print("\n✓ Configuration is valid!")
    else:
        print("\n✗ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
