"""
Configuration file for Experiment Agent System.
"""

import os
from typing import Optional


# =============================================================================
# API Provider Configuration
# =============================================================================

# API提供商选择: "azure" 或 "openai"
API_PROVIDER: str = "openai"

# -----------------------------------------------------------------------------
# Azure OpenAI 配置（当 API_PROVIDER=azure 时使用）
# -----------------------------------------------------------------------------
AZURE_ENDPOINT: Optional[str] = (
    "https://yikai-m870y3k5-eastus2.cognitiveservices.azure.com/"
)
AZURE_API_KEY: Optional[str] = (
    "PYPBhHVSnCL9J2i4whVg3X36uTArslFQWvYp5ENh7fFmR17gyiS1JQQJ99BCACHYHv6XJ3w3AAAAACOGruFq"
)
AZURE_API_VERSION: str = "2024-12-01-preview"
AZURE_DEPLOYMENT: str = "gpt-4o-2"

# -----------------------------------------------------------------------------
# OpenAI 配置（当 API_PROVIDER=openai 时使用）
# 支持OpenAI官方API或其他兼容API（如 https://www.dmxapi.cn/v1）
# -----------------------------------------------------------------------------
# OPENAI_API_KEY: Optional[str] = "sk-BWZ0Kqbk3PvdF0zRFf69B63901B84e85A5B4D8B1AfE27e2e"
# OPENAI_API_BASE: Optional[str] = "https://api.xi-ai.cn/v1"

OPENAI_API_KEY: Optional[str] = "sk-Q1Aah6ovHJyPhlmi0yZtNazWo29XiMyBIMtaKZGtG6RzFp2W"
OPENAI_API_BASE: Optional[str] = "https://www.dmxapi.cn/v1"

# =============================================================================
# Model Configuration - 每个Agent的模型配置
# =============================================================================

EXPERIMENT_MASTER_MODEL: str = "MiniMax-M2"

PRE_ANALYSIS_MODEL: str = "MiniMax-M2"

CODE_PLAN_MODEL: str = "MiniMax-M2"

CODE_IMPLEMENT_MODEL: str = "MiniMax-M2"

CODE_JUDGE_MODEL: str = "MiniMax-M2"

EXECUTE_EXPERIMENT_MODEL: str = "MiniMax-M2"

RESULT_ANALYSIS_MODEL: str = "MiniMax-M2"

DEFAULT_MODEL: str = "MiniMax-M2"

OUTPUT_UNIFIER_MODEL: str = "gpt-5-mini"


# =============================================================================
# Docker Configuration
# =============================================================================

# Docker container configuration
DOCKER_HOST: str = "localhost"
DOCKER_PORT: int = 18379
DOCKER_TIMEOUT: int = 3600

# Container name (for reference)
DOCKER_CONTAINER_NAME: str = "research-agent-container"


# =============================================================================
# Path Configuration
# =============================================================================

# Working directory in Docker container
DOCKER_WORKING_DIR: str = "/workspace"

# Project directory in Docker container (corresponding to PROJECT_DIR)
DOCKER_PROJECT_DIR: str = "/workspace/project"

# Base directory for all experiment workspaces
BASE_WORKSPACES_DIR: str = (
    "/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent/src/agents/experiment_agent/workspaces"
)

# Docker dataset path (mounted in container)
DOCKER_DATASET_DIR: str = "/workspace/dataset_candidate"


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
    return os.path.join(get_workspace_dir(experiment_id), "idea.json")


def get_paper_input_path(experiment_id: str) -> str:
    """Get the paper input path for a specific experiment."""
    return os.path.join(get_workspace_dir(experiment_id), "paper.tex")


def get_dataset_candidate_dir(experiment_id: str) -> str:
    """Get the dataset candidate directory for a specific experiment."""
    return os.path.join(get_workspace_dir(experiment_id), "dataset_candidate")


def get_logs_dir(experiment_id: str) -> str:
    """Get the logs directory for a specific experiment."""
    return os.path.join(get_workspace_dir(experiment_id), "logs")


def get_results_dir(experiment_id: str) -> str:
    """Get the results directory for a specific experiment."""
    return os.path.join(get_workspace_dir(experiment_id), "results")


def get_cache_dir(experiment_id: str) -> str:
    """Get the cache directory for a specific experiment."""
    return os.path.join(get_workspace_dir(experiment_id), "cached")


# Path Globals
# These are now managed via the ExperimentContext, but kept here for module-level access.
# They will be updated by ExperimentContext.initialize()
WORKSPACE_ROOT: str = ""
PROJECT_ROOT: str = ""


class ExperimentContext:
    _instance = None

    def __init__(self):
        self.workspace_root = ""
        self.project_root = ""
        self.experiment_id = ""
        self.initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def initialize(
        cls, workspace_root: str, project_root: str, experiment_id: str = None
    ):
        instance = cls.get_instance()
        instance.workspace_root = workspace_root
        instance.project_root = project_root
        instance.experiment_id = experiment_id
        instance.initialized = True

        # Update module-level globals for backward compatibility and direct access
        global WORKSPACE_ROOT, PROJECT_ROOT
        WORKSPACE_ROOT = workspace_root
        PROJECT_ROOT = project_root

        # Also update legacy global aliases if they exist in module scope
        global LOCAL_WORKSPACE_DIR, PROJECT_DIR
        LOCAL_WORKSPACE_DIR = workspace_root
        PROJECT_DIR = project_root


# Legacy support for direct imports (e.g., from config import LOCAL_WORKSPACE_DIR)
# This requires Python 3.7+
def __getattr__(name):
    if name == "LOCAL_WORKSPACE_DIR":
        return WORKSPACE_ROOT
    if name == "PROJECT_DIR":
        return PROJECT_ROOT
    if name in [
        "IDEA_INPUT_PATH",
        "PAPER_INPUT_PATH",
        "DATASET_CANDIDATE_DIR",
        "EXPERIMENT_LOGS_DIR",
        "RESULTS_OUTPUT_DIR",
        "CACHE_DIR",
    ]:
        # Return empty string or calculated path if experiment_id is known
        return ""
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Initialize global instance
_context = ExperimentContext.get_instance()


# =============================================================================
# Experiment Configuration
# =============================================================================

# Maximum iterations for the experiment workflow
MAX_WORKFLOW_ITERATIONS: int = 5

# Enable/disable tracing
ENABLE_TRACING: bool = False

# Enable/disable streaming for API calls
# Streaming provides real-time output but may be unstable with some API providers
ENABLE_STREAMING: bool = True

# Timeout settings (in seconds)
CODE_EXECUTION_TIMEOUT: int = 3600
AGENT_CALL_TIMEOUT: int = 600


# =============================================================================
# Logging Configuration
# =============================================================================

# Log level
LOG_LEVEL: str = "INFO"

# Enable colored output
COLORED_LOGS: bool = True


# =============================================================================
# Helper Functions
# =============================================================================


def get_openai_config(model: Optional[str] = None) -> dict:
    """
    Get OpenAI configuration dictionary.

    Args:
        model: Optional model name to use. If not provided, uses DEFAULT_MODEL.

    Returns:
        Dictionary with OpenAI configuration
    """
    model_to_use = model or DEFAULT_MODEL

    if API_PROVIDER == "azure":
        return {
            "use_azure": True,
            "endpoint": AZURE_ENDPOINT,
            "api_key": AZURE_API_KEY,
            "api_version": AZURE_API_VERSION,
            "deployment": AZURE_DEPLOYMENT,
            "model_name": model_to_use,
        }
    else:  # openai
        config = {
            "use_azure": False,
            "api_key": OPENAI_API_KEY,
            "model_name": model_to_use,
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
        "experiment_master": EXPERIMENT_MASTER_MODEL,
        "pre_analysis": PRE_ANALYSIS_MODEL,
        "code_plan": CODE_PLAN_MODEL,
        "code_implement": CODE_IMPLEMENT_MODEL,
        "code_judge": CODE_JUDGE_MODEL,
        "execute_experiment": EXECUTE_EXPERIMENT_MODEL,
        "result_analysis": RESULT_ANALYSIS_MODEL,
        "output_unifier": OUTPUT_UNIFIER_MODEL,
        "default": DEFAULT_MODEL,
    }


def get_docker_config() -> dict:
    """
    Get Docker configuration dictionary.

    Returns:
        Dictionary with Docker configuration
    """
    return {
        "host": DOCKER_HOST,
        "port": DOCKER_PORT,
        "timeout": DOCKER_TIMEOUT,
        "container_name": DOCKER_CONTAINER_NAME,
        "working_dir": DOCKER_WORKING_DIR,
        "project_dir": DOCKER_PROJECT_DIR,
    }


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
        "working_dir": workspace_dir,  # Workspace directory for this experiment
        "local_workspace": workspace_dir,  # Backward compatibility
        "project_dir": project_dir,  # Project root directory
        "docker_workspace": DOCKER_WORKING_DIR,
        "docker_project_dir": DOCKER_PROJECT_DIR,
        "dataset_candidate_dir": get_dataset_candidate_dir(experiment_id),
        "docker_dataset_dir": DOCKER_DATASET_DIR,
        "idea_input": get_idea_input_path(experiment_id),
        "paper_input": get_paper_input_path(experiment_id),
        "logs_dir": get_logs_dir(experiment_id),
        "results_dir": get_results_dir(experiment_id),
        "cache_dir": get_cache_dir(experiment_id),
    }


def validate_config(experiment_id: Optional[str] = None) -> tuple[bool, list[str]]:
    """
    Validate configuration settings.

    Args:
        experiment_id: Optional experiment ID. If provided, also validates and creates
                      experiment-specific directories.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check API Provider
    if API_PROVIDER not in ["azure", "openai"]:
        errors.append(
            f"Invalid API_PROVIDER: {API_PROVIDER}. Must be 'azure' or 'openai'"
        )

    # Check OpenAI configuration based on provider
    if API_PROVIDER == "azure":
        if not AZURE_ENDPOINT:
            errors.append("AZURE_ENDPOINT is not set")
        if not AZURE_API_KEY:
            errors.append("AZURE_API_KEY is not set")
    else:  # openai
        if not OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")

    # Check model configuration
    if not DEFAULT_MODEL:
        errors.append("DEFAULT_MODEL is not set")

    # Check each agent's model is configured
    agent_models = {
        "EXPERIMENT_MASTER_MODEL": EXPERIMENT_MASTER_MODEL,
        "PRE_ANALYSIS_MODEL": PRE_ANALYSIS_MODEL,
        "CODE_PLAN_MODEL": CODE_PLAN_MODEL,
        "CODE_IMPLEMENT_MODEL": CODE_IMPLEMENT_MODEL,
        "CODE_JUDGE_MODEL": CODE_JUDGE_MODEL,
        "EXECUTE_EXPERIMENT_MODEL": EXECUTE_EXPERIMENT_MODEL,
        "RESULT_ANALYSIS_MODEL": RESULT_ANALYSIS_MODEL,
    }

    for model_name, model_value in agent_models.items():
        if not model_value:
            errors.append(f"{model_name} is not set")

    # Check Docker configuration
    if not DOCKER_HOST:
        errors.append("DOCKER_HOST is not set")
    if DOCKER_PORT <= 0 or DOCKER_PORT > 65535:
        errors.append(f"Invalid DOCKER_PORT: {DOCKER_PORT}")

    # Check and create base workspaces directory
    os.makedirs(BASE_WORKSPACES_DIR, exist_ok=True)

    # If experiment_id is provided, create experiment-specific directories
    if experiment_id:
        workspace_dir = get_workspace_dir(experiment_id)
        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(get_project_dir(experiment_id), exist_ok=True)
        os.makedirs(get_logs_dir(experiment_id), exist_ok=True)
        os.makedirs(get_results_dir(experiment_id), exist_ok=True)
        os.makedirs(get_cache_dir(experiment_id), exist_ok=True)
        os.makedirs(get_dataset_candidate_dir(experiment_id), exist_ok=True)

    return len(errors) == 0, errors


def print_config(experiment_id: Optional[str] = None):
    """
    Print current configuration (masking sensitive information).

    Args:
        experiment_id: Optional experiment ID. If provided, also prints experiment-specific paths.
    """
    print("=" * 80)
    print("Experiment Agent Configuration")
    print("=" * 80)
    print(f"\n[API Configuration]")
    print(f"  API Provider: {API_PROVIDER}")
    if API_PROVIDER == "azure":
        print(f"  Azure Endpoint: {AZURE_ENDPOINT}")
        print(f"  Azure Deployment: {AZURE_DEPLOYMENT}")
        print(f"  Azure API Key: {'*' * 20}...")
    else:
        print(f"  OpenAI API Key: {'*' * 20}...")
        if OPENAI_API_BASE:
            print(f"  OpenAI API Base: {OPENAI_API_BASE}")

    print(f"\n[Model Configuration]")
    print(f"  Experiment Master: {EXPERIMENT_MASTER_MODEL}")
    print(f"  Pre-Analysis: {PRE_ANALYSIS_MODEL}")
    print(f"  Code Plan: {CODE_PLAN_MODEL}")
    print(f"  Code Implement: {CODE_IMPLEMENT_MODEL}")
    print(f"  Code Judge: {CODE_JUDGE_MODEL}")
    print(f"  Execute Experiment: {EXECUTE_EXPERIMENT_MODEL}")
    print(f"  Result Analysis: {RESULT_ANALYSIS_MODEL}")
    print(f"  Default: {DEFAULT_MODEL}")

    print(f"\n[Docker Configuration]")
    print(f"  Host: {DOCKER_HOST}")
    print(f"  Port: {DOCKER_PORT}")
    print(f"  Timeout: {DOCKER_TIMEOUT}s")
    print(f"  Container: {DOCKER_CONTAINER_NAME}")
    print(f"  Working Dir: {DOCKER_WORKING_DIR}")

    print(f"\n[Path Configuration]")
    print(f"  Base Workspaces Dir: {BASE_WORKSPACES_DIR}")

    if experiment_id:
        print(f"\n  [Experiment-Specific Paths for '{experiment_id}']")
        print(f"  Workspace Dir: {get_workspace_dir(experiment_id)}")
        print(f"  Project Dir: {get_project_dir(experiment_id)}")
        print(f"  Idea Input: {get_idea_input_path(experiment_id)}")
        print(f"  Paper Input: {get_paper_input_path(experiment_id)}")
        print(f"  Dataset Dir: {get_dataset_candidate_dir(experiment_id)}")
        print(f"  Logs Dir: {get_logs_dir(experiment_id)}")
        print(f"  Results Dir: {get_results_dir(experiment_id)}")
        print(f"  Cache Dir: {get_cache_dir(experiment_id)}")

    print(f"\n[Experiment Configuration]")
    print(f"  Max Iterations: {MAX_WORKFLOW_ITERATIONS}")
    print(f"  Enable Tracing: {ENABLE_TRACING}")
    print(f"  Streaming Mode: Always Enabled (real-time API output)")
    print(f"  Code Execution Timeout: {CODE_EXECUTION_TIMEOUT}s")
    print(f"  Agent Call Timeout: {AGENT_CALL_TIMEOUT}s")

    print("=" * 80)


if __name__ == "__main__":
    # Test configuration
    print_config()

    # Validate configuration
    is_valid, errors = validate_config()
    if is_valid:
        print("\n✓ Configuration is valid!")
    else:
        print("\n✗ Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
