"""
Configuration file for Experiment Agent System.

Contains all necessary configuration variables for running experiments.
"""

import os
from typing import Optional


# =============================================================================
# OpenAI API Configuration
# =============================================================================

# Azure OpenAI Configuration (if using Azure)
AZURE_ENDPOINT: Optional[str] = os.getenv(
    "AZURE_ENDPOINT", "https://yikai-m870y3k5-eastus2.cognitiveservices.azure.com/"
)
AZURE_API_KEY: Optional[str] = os.getenv(
    "AZURE_API_KEY",
    "PYPBhHVSnCL9J2i4whVg3X36uTArslFQWvYp5ENh7fFmR17gyiS1JQQJ99BCACHYHv6XJ3w3AAAAACOGruFq",
)
AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
AZURE_DEPLOYMENT: str = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-2")

# OpenAI Configuration (if using OpenAI directly)
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

# Model Configuration
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o-2")

# Use Azure or OpenAI
USE_AZURE: bool = os.getenv("USE_AZURE", "true").lower() == "true"


# =============================================================================
# Docker Configuration
# =============================================================================

# Docker container configuration
DOCKER_HOST: str = os.getenv("DOCKER_HOST", "localhost")
DOCKER_PORT: int = int(os.getenv("DOCKER_PORT", "18379"))
DOCKER_TIMEOUT: int = int(os.getenv("DOCKER_TIMEOUT", "3600"))

# Container name (for reference)
DOCKER_CONTAINER_NAME: str = os.getenv(
    "DOCKER_CONTAINER_NAME", "research-agent-container"
)


# =============================================================================
# Path Configuration
# =============================================================================

# Working directory in Docker container
DOCKER_WORKING_DIR: str = os.getenv("DOCKER_WORKING_DIR", "/workspace")

# Project directory in Docker container (corresponding to PROJECT_DIR)
DOCKER_PROJECT_DIR: str = os.getenv("DOCKER_PROJECT_DIR", "/workspace/project")

# Local workspace directory (shared with Docker)
LOCAL_WORKSPACE_DIR: str = os.getenv(
    "LOCAL_WORKSPACE_DIR",
    "/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent/src/agents/experiment_agent/workspace",
)

# Project implementation directory (where all code implementation and experiments happen)
PROJECT_DIR: str = os.getenv(
    "PROJECT_DIR",
    os.path.join(LOCAL_WORKSPACE_DIR, "project"),
)

# Input paths
IDEA_INPUT_PATH: str = os.getenv(
    "IDEA_INPUT_PATH",
    os.path.join(LOCAL_WORKSPACE_DIR, "idea.json"),
)

PAPER_INPUT_PATH: str = os.getenv(
    "PAPER_INPUT_PATH",
    os.path.join(LOCAL_WORKSPACE_DIR, "paper.tex"),
)

# Dataset paths
DATASET_CANDIDATE_DIR: str = os.getenv(
    "DATASET_CANDIDATE_DIR",
    "/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/AI-Researcher/benchmark/process/dataset_candidate",
)

# Docker dataset path (mounted in container)
DOCKER_DATASET_DIR: str = os.getenv(
    "DOCKER_DATASET_DIR", "/workspace/dataset_candidate"
)

# Output paths
EXPERIMENT_LOGS_DIR: str = os.getenv(
    "EXPERIMENT_LOGS_DIR",
    os.path.join(LOCAL_WORKSPACE_DIR, "logs"),
)

RESULTS_OUTPUT_DIR: str = os.getenv(
    "RESULTS_OUTPUT_DIR",
    os.path.join(LOCAL_WORKSPACE_DIR, "results"),
)

CACHE_DIR: str = os.getenv(
    "CACHE_DIR",
    os.path.join(LOCAL_WORKSPACE_DIR, "cached"),
)


# =============================================================================
# Experiment Configuration
# =============================================================================

# Maximum iterations for the experiment workflow
MAX_WORKFLOW_ITERATIONS: int = int(os.getenv("MAX_WORKFLOW_ITERATIONS", "5"))

# Enable/disable tracing
ENABLE_TRACING: bool = os.getenv("ENABLE_TRACING", "false").lower() == "true"

# Note: Streaming is always enabled for API calls (using Runner.run_stream)
# This provides real-time output from the AI agents

# Timeout settings (in seconds)
CODE_EXECUTION_TIMEOUT: int = int(os.getenv("CODE_EXECUTION_TIMEOUT", "3600"))
AGENT_CALL_TIMEOUT: int = int(os.getenv("AGENT_CALL_TIMEOUT", "600"))


# =============================================================================
# Logging Configuration
# =============================================================================

# Log level
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Enable colored output
COLORED_LOGS: bool = os.getenv("COLORED_LOGS", "true").lower() == "true"


# =============================================================================
# Helper Functions
# =============================================================================


def get_openai_config() -> dict:
    """
    Get OpenAI configuration dictionary.

    Returns:
        Dictionary with OpenAI configuration
    """
    if USE_AZURE:
        return {
            "use_azure": True,
            "endpoint": AZURE_ENDPOINT,
            "api_key": AZURE_API_KEY,
            "api_version": AZURE_API_VERSION,
            "deployment": AZURE_DEPLOYMENT,
            "model_name": MODEL_NAME,
        }
    else:
        return {
            "use_azure": False,
            "api_key": OPENAI_API_KEY,
            "model_name": MODEL_NAME,
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

    Returns:
        Dictionary with path configuration
    """
    return {
        "local_workspace": LOCAL_WORKSPACE_DIR,
        "project_dir": PROJECT_DIR,
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

    # Check OpenAI configuration
    if USE_AZURE:
        if not AZURE_ENDPOINT:
            errors.append("AZURE_ENDPOINT is not set")
        if not AZURE_API_KEY:
            errors.append("AZURE_API_KEY is not set")
    else:
        if not OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")

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
    print(f"\n[OpenAI Configuration]")
    print(f"  Use Azure: {USE_AZURE}")
    print(f"  Model: {MODEL_NAME}")
    if USE_AZURE:
        print(f"  Azure Endpoint: {AZURE_ENDPOINT}")
        print(f"  Azure Deployment: {AZURE_DEPLOYMENT}")
        print(f"  Azure API Key: {'*' * 20}...")
    else:
        print(f"  OpenAI API Key: {'*' * 20}...")

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
