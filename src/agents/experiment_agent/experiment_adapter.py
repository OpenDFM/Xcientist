#!/usr/bin/env python3
"""Experiment Agent Adapter - simplified CLI with workspace and config options.

Usage:
    python experiment_adapter.py --workspace /path/to/workspace --config /path/to/config.yaml
    python experiment_adapter.py -w /path/to/workspace -c /path/to/config.yaml
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


async def run_with_config(workspace: str, config_path: str) -> int:
    """Run experiment agent with specified workspace and config.

    Args:
        workspace: Path to workspace directory
        config_path: Path to runtime config YAML file

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Setup environment variables
    workspace = os.path.abspath(os.path.expanduser(workspace))
    config_path = os.path.abspath(os.path.expanduser(config_path))

    # Validate paths exist
    if not os.path.isdir(workspace):
        raise ValueError(f"Workspace directory does not exist: {workspace}")
    if not os.path.isfile(config_path):
        raise ValueError(f"Config file does not exist: {config_path}")

    # Set environment variables
    os.environ["EXPERIMENT_AGENT_WORKSPACE_DIR"] = workspace
    os.environ["EXPERIMENT_AGENT_CONFIG_PATH"] = config_path

    # Ensure workspace subdirectories exist
    os.makedirs(workspace, exist_ok=True)
    for subdir in ["project", "logs", "cached", "dataset_candidate", "model_candidate", "results", "agent_reports", "repos"]:
        os.makedirs(os.path.join(workspace, subdir), exist_ok=True)

    # Reload config with custom path BEFORE importing any config-dependent modules
    from src.config import reload_config
    reload_config(config_path)

    # Now import modules that depend on config
    from src.agents.experiment_agent.main import main_async
    from src.agents.experiment_agent.config import (
        get_science_max_iterations,
        get_prepare_validation_feedback_rounds,
        get_code_validation_feedback_rounds,
        get_science_validation_feedback_rounds,
    )

    # Get experiment ID from workspace directory name
    experiment_id = os.path.basename(workspace)

    # Load other settings from config
    max_iterations = get_science_max_iterations()
    prepare_validation_rounds = get_prepare_validation_feedback_rounds()
    code_validation_rounds = get_code_validation_feedback_rounds()
    science_validation_rounds = get_science_validation_feedback_rounds()

    print(f"Experiment ID: {experiment_id}")
    print(f"Workspace: {workspace}")
    print(f"Config: {config_path}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Prepare Validation Rounds: {prepare_validation_rounds}")
    print(f"Code Validation Rounds: {code_validation_rounds}")
    print(f"Science Validation Rounds: {science_validation_rounds}")

    # Create args object compatible with main_async
    class ExperimentArgs:
        pass

    args = ExperimentArgs()
    args.experiment = experiment_id
    args.prepare_only = False  # Run full experiment
    args.skip_prepare = False  # Prepare is required
    args.force = False
    args.clone_depth = 1
    args.skip_repos = False
    args.skip_datasets = False
    args.max_iterations = max_iterations
    args.resume = False
    args.verbose = False

    # Run the experiment
    return await main_async(args)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment Agent Adapter - Run experiment with specified workspace and config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with workspace and config
    python experiment_adapter.py --workspace /path/to/workspace --config /path/to/config.yaml

    # Short form
    python experiment_adapter.py -w /path/to/workspace -c /path/to/config.yaml
        """
    )

    parser.add_argument(
        "--workspace", "-w",
        required=True,
        help="Path to workspace directory"
    )

    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to runtime config YAML file"
    )

    parsed_args = parser.parse_args()

    try:
        return asyncio.run(run_with_config(parsed_args.workspace, parsed_args.config))
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        return 130
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
