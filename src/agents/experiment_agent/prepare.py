#!/usr/bin/env python3
"""
PrepareAgent entrypoint

Initialize an experiment workspace from a CoI-Agent result.json-like file using an LLM tool-calling agent:
- create workspaces/<experiment_id> directory structure
- write idea.md + source_result.json
- clone reference repos into repos/
- download datasets into dataset_candidate/ (HuggingFace only; skip missing)
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.agents.experiment_agent.layers.prepare.entry import run_prepare
from src.agents.experiment_agent.shared.utils.config import print_config


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare experiment workspace from result.json (LLM tool-calling)"
    )
    parser.add_argument(
        "--experiment", "-e", required=True, help="Experiment ID (workspace name)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite idea.md and re-download/clone"
    )
    parser.add_argument(
        "--clone-depth", type=int, default=1, help="git clone depth (default: 1)"
    )
    parser.add_argument("--skip-repos", action="store_true", help="Skip cloning repos")
    parser.add_argument(
        "--skip-datasets", action="store_true", help="Skip downloading datasets"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose agent output")
    return parser.parse_args()


async def main_async(args) -> int:
    print_config()
    report = await run_prepare(
        experiment_id=str(args.experiment),
        force=bool(args.force),
        clone_depth=int(args.clone_depth),
        skip_repos=bool(args.skip_repos),
        skip_datasets=bool(args.skip_datasets),
        verbose=bool(args.verbose),
    )

    print("\n[Prepare Summary]")
    print(f"  Experiment: {report.experiment_id}")
    print(f"  Workspace: {report.workspace_dir}")
    print(f"  Project dir: {report.project_dir}")
    print(f"  idea.md: {report.idea_md_path}")
    print(f"  Repos dir: {report.repos_dir}")
    print(f"  Dataset dir: {report.dataset_dir}")
    return 0


def main(args) -> int:
    return int(asyncio.run(main_async(args)))


if __name__ == "__main__":
    raise SystemExit(main(get_args()))
