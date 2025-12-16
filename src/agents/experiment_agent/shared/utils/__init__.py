"""
Shared utilities for SuperAgent.

Modules:
- cache: Thread-safe caching and state management
- config: Configuration and API setup
- dag: DAG scheduling utilities
"""

from src.agents.experiment_agent.shared.utils.cache import Cache
from src.agents.experiment_agent.shared.utils.dag import DAGScheduler, TaskStatus, build_dependency_graph_from_items
from src.agents.experiment_agent.shared.utils.config import (
    ProjectContext,
    setup_openai_api,
    ensure_experiment_dirs,
    get_reference_repos,
    get_idea_input_path,
    get_project_dir,
    get_dataset_dir,
    print_config,
    BASE_WORKSPACES_DIR,
    CODE_ARCHITECT_MODEL,
    CODE_MANAGER_MODEL,
    CODE_WORKER_MODEL,
    CODE_INTEGRATOR_MODEL,
    SCIENCE_ARCHITECT_MODEL,
    SCIENCE_MANAGER_MODEL,
    SCIENCE_WORKER_MODEL,
    SCIENCE_INTEGRATOR_MODEL,
)

__all__ = [
    # Cache
    "Cache",
    # DAG
    "DAGScheduler",
    "TaskStatus",
    "build_dependency_graph_from_items",
    # Config
    "ProjectContext",
    "setup_openai_api",
    "ensure_experiment_dirs",
    "get_reference_repos",
    "get_idea_input_path",
    "get_project_dir",
    "get_dataset_dir",
    "print_config",
    "BASE_WORKSPACES_DIR",
    # Code layer models
    "CODE_ARCHITECT_MODEL",
    "CODE_MANAGER_MODEL",
    "CODE_WORKER_MODEL",
    "CODE_INTEGRATOR_MODEL",
    # Science layer models
    "SCIENCE_ARCHITECT_MODEL",
    "SCIENCE_MANAGER_MODEL",
    "SCIENCE_WORKER_MODEL",
    "SCIENCE_INTEGRATOR_MODEL",
]
