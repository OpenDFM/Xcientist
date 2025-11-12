"""
Helper functions for preparing workspace information.

These functions scan and index reference materials (codebases, papers, datasets)
for efficient access during implementation.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional


def scan_reference_codebases(repos_dir: str) -> Dict[str, any]:
    """
    Scan the repos directory and generate codebase information.

    Args:
        repos_dir: Path to the repos directory

    Returns:
        Dictionary containing codebase information
    """
    repos_path = Path(repos_dir)

    if not repos_path.exists():
        return {
            "success": False,
            "error": f"Repos directory not found: {repos_dir}",
            "codebases": [],
        }

    codebases = []

    try:
        # List all directories in repos
        for item in repos_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                # Extract repo name without author prefix
                # e.g., "atomicarchitects_equiformer" -> "equiformer"
                # e.g., "brain-research_mpnn" -> "mpnn"
                dir_name = item.name
                display_name = dir_name

                # Try to extract repo name after underscore or dash
                if "_" in dir_name:
                    parts = dir_name.split("_", 1)
                    if len(parts) > 1:
                        display_name = parts[1]
                elif "-" in dir_name:
                    parts = dir_name.split("-", 1)
                    if len(parts) > 1 and "_" in parts[1]:
                        # Handle cases like "brain-research_mpnn"
                        display_name = (
                            parts[1].split("_", 1)[1] if "_" in parts[1] else parts[1]
                        )

                codebase_info = {
                    "name": display_name,  # Display name without author prefix
                    "directory_name": dir_name,  # Actual directory name
                    "path": str(item),
                    "relative_path": f"../repos/{dir_name}",
                }

                # Check for common files to determine project type
                readme_path = item / "README.md"
                setup_py = item / "setup.py"
                requirements_txt = item / "requirements.txt"
                pyproject_toml = item / "pyproject.toml"

                if readme_path.exists():
                    codebase_info["has_readme"] = True
                if setup_py.exists() or pyproject_toml.exists():
                    codebase_info["has_setup"] = True
                if requirements_txt.exists():
                    codebase_info["has_requirements"] = True

                # Count Python files
                python_files = list(item.rglob("*.py"))
                codebase_info["python_files_count"] = len(python_files)

                codebases.append(codebase_info)

        return {"success": True, "codebases": codebases, "total_count": len(codebases)}

    except Exception as e:
        return {
            "success": False,
            "error": f"Error scanning repos: {str(e)}",
            "codebases": [],
        }


def format_codebase_list_for_prompt(codebases: List[Dict]) -> str:
    """
    Format codebase list for inclusion in agent prompts.

    Args:
        codebases: List of codebase dictionaries

    Returns:
        Formatted string for prompt
    """
    if not codebases:
        return "No reference codebases available."

    lines = []
    for i, cb in enumerate(codebases, 1):
        line = f"{i}. **{cb['name']}**"
        details = []
        if cb.get("python_files_count", 0) > 0:
            details.append(f"{cb['python_files_count']} Python files")
        if cb.get("has_readme"):
            details.append("README available")

        if details:
            line += f" - {', '.join(details)}"

        # Use directory_name for the actual path, but show cleaned name
        dir_name = cb.get("directory_name", cb["name"])
        line += f"\n   Path: `../repos/{dir_name}`"
        lines.append(line)

    return "\n".join(lines)


def save_prepare_info_to_cache(cache_dir: str, domain: str, prepare_info: Dict) -> bool:
    """
    Save preparation information to cache.

    Args:
        cache_dir: Cache directory path
        domain: Domain name (e.g., "mlff", "cv")
        prepare_info: Preparation information dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        cache_path = Path(cache_dir) / domain
        cache_path.mkdir(parents=True, exist_ok=True)

        prepare_file = cache_path / "step0_prepare.json"

        with open(prepare_file, "w", encoding="utf-8") as f:
            json.dump(prepare_info, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error saving prepare info to cache: {e}")
        return False


def load_prepare_info_from_cache(cache_dir: str, domain: str) -> Optional[Dict]:
    """
    Load preparation information from cache.

    Args:
        cache_dir: Cache directory path
        domain: Domain name

    Returns:
        Prepare info dictionary or None if not found
    """
    try:
        prepare_file = Path(cache_dir) / domain / "step0_prepare.json"

        if not prepare_file.exists():
            return None

        with open(prepare_file, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"Error loading prepare info from cache: {e}")
        return None


def prepare_workspace_info(workspace_dir: str) -> Dict:
    """
    Prepare comprehensive workspace information.

    Args:
        workspace_dir: Workspace directory path

    Returns:
        Dictionary with workspace information
    """
    workspace_path = Path(workspace_dir)

    # Scan reference codebases
    repos_dir = workspace_path / "repos"
    codebase_scan = scan_reference_codebases(str(repos_dir))

    # Scan papers (if needed)
    papers_dir = workspace_path / "papers"
    papers_count = 0
    if papers_dir.exists():
        papers_count = len(
            list(papers_dir.glob("*.tex")) + list(papers_dir.glob("*.pdf"))
        )

    # Scan datasets
    dataset_dir = workspace_path / "dataset_candidate"
    dataset_count = 0
    if dataset_dir.exists():
        dataset_count = len([d for d in dataset_dir.iterdir() if d.is_dir()])

    prepare_info = {
        "timestamp": str(Path.cwd()),  # Can be improved with actual timestamp
        "workspace_dir": str(workspace_dir),
        "reference_codebases": {
            "scan_result": codebase_scan,
            "formatted_list": (
                format_codebase_list_for_prompt(codebase_scan.get("codebases", []))
                if codebase_scan.get("success")
                else "Error scanning codebases"
            ),
        },
        "reference_papers": {"count": papers_count, "directory": str(papers_dir)},
        "datasets": {"count": dataset_count, "directory": str(dataset_dir)},
    }

    return prepare_info
