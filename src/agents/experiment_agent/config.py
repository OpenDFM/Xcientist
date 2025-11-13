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
OPENAI_API_KEY: Optional[str] = "sk-Q1Aah6ovHJyPhlmi0yZtNazWo29XiMyBIMtaKZGtG6RzFp2W"
OPENAI_API_BASE: Optional[str] = "https://www.dmxapi.cn/v1"

# =============================================================================
# Model Configuration - 模型配置
# =============================================================================

EXPENSIVE_MODEL: str = "gpt-5"

CHEAP_MODEL: str = "gpt-5-mini"

# 默认模型名称（向后兼容）
MODEL_NAME: str = EXPENSIVE_MODEL

# 使用Azure标志（向后兼容）
USE_AZURE: bool = API_PROVIDER == "azure"


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

# Local workspace directory (shared with Docker)
LOCAL_WORKSPACE_DIR: str = (
    "/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent/src/agents/experiment_agent/workspace"
)

# Project implementation directory (where all code implementation and experiments happen)
PROJECT_DIR: str = os.path.join(LOCAL_WORKSPACE_DIR, "project")

# Input paths
IDEA_INPUT_PATH: str = os.path.join(LOCAL_WORKSPACE_DIR, "idea.json")

PAPER_INPUT_PATH: str = os.path.join(LOCAL_WORKSPACE_DIR, "paper.tex")

# Dataset paths
DATASET_CANDIDATE_DIR: str = os.path.join(LOCAL_WORKSPACE_DIR, "dataset_candidate")

# Docker dataset path (mounted in container)
DOCKER_DATASET_DIR: str = "/workspace/dataset_candidate"

# Output paths
EXPERIMENT_LOGS_DIR: str = os.path.join(LOCAL_WORKSPACE_DIR, "logs")

RESULTS_OUTPUT_DIR: str = os.path.join(LOCAL_WORKSPACE_DIR, "results")

CACHE_DIR: str = os.path.join(LOCAL_WORKSPACE_DIR, "cached")


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
        model: Optional model name to use. If not provided, uses MODEL_NAME.

    Returns:
        Dictionary with OpenAI configuration
    """
    model_to_use = model or MODEL_NAME

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
    Get model configuration dictionary.

    Returns:
        Dictionary with model configuration including expensive and cheap models
    """
    return {
        "expensive_model": EXPENSIVE_MODEL,
        "cheap_model": CHEAP_MODEL,
        "default_model": MODEL_NAME,
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


def get_path_config() -> dict:
    """
    Get path configuration dictionary.

    Note: working_dir refers to LOCAL_WORKSPACE_DIR (parent directory)
          project_dir is the actual project root where code is implemented

    Returns:
        Dictionary with path configuration
    """
    return {
        "working_dir": LOCAL_WORKSPACE_DIR,  # Workspace directory (parent)
        "local_workspace": LOCAL_WORKSPACE_DIR,  # Backward compatibility
        "project_dir": PROJECT_DIR,  # Project root directory (for code operations)
        "docker_workspace": DOCKER_WORKING_DIR,
        "docker_project_dir": DOCKER_PROJECT_DIR,
        "dataset_candidate_dir": DATASET_CANDIDATE_DIR,
        "docker_dataset_dir": DOCKER_DATASET_DIR,
        "idea_input": IDEA_INPUT_PATH,
        "paper_input": PAPER_INPUT_PATH,
        "logs_dir": EXPERIMENT_LOGS_DIR,
        "results_dir": RESULTS_OUTPUT_DIR,
        "cache_dir": CACHE_DIR,
    }


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate configuration settings.

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
    if not EXPENSIVE_MODEL:
        errors.append("EXPENSIVE_MODEL is not set")
    if not CHEAP_MODEL:
        errors.append("CHEAP_MODEL is not set")

    # Check Docker configuration
    if not DOCKER_HOST:
        errors.append("DOCKER_HOST is not set")
    if DOCKER_PORT <= 0 or DOCKER_PORT > 65535:
        errors.append(f"Invalid DOCKER_PORT: {DOCKER_PORT}")

    # Check paths exist (create if needed)
    os.makedirs(LOCAL_WORKSPACE_DIR, exist_ok=True)
    os.makedirs(PROJECT_DIR, exist_ok=True)
    os.makedirs(EXPERIMENT_LOGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    return len(errors) == 0, errors


def print_config():
    """Print current configuration (masking sensitive information)."""
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
    print(f"  Expensive Model (code plan, implement, judge): {EXPENSIVE_MODEL}")
    print(f"  Cheap Model (pre-analysis, execute, analysis): {CHEAP_MODEL}")
    print(f"  Default Model: {MODEL_NAME}")

    print(f"\n[Docker Configuration]")
    print(f"  Host: {DOCKER_HOST}")
    print(f"  Port: {DOCKER_PORT}")
    print(f"  Timeout: {DOCKER_TIMEOUT}s")
    print(f"  Container: {DOCKER_CONTAINER_NAME}")
    print(f"  Working Dir: {DOCKER_WORKING_DIR}")

    print(f"\n[Path Configuration]")
    print(f"  Local Workspace: {LOCAL_WORKSPACE_DIR}")
    print(f"  Project Dir: {PROJECT_DIR}")
    print(f"  Idea Input: {IDEA_INPUT_PATH}")
    print(f"  Paper Input: {PAPER_INPUT_PATH}")
    print(f"  Logs Dir: {EXPERIMENT_LOGS_DIR}")
    print(f"  Results Dir: {RESULTS_OUTPUT_DIR}")

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
