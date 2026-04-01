"""
Deterministic checks that `project/` remains self-contained at runtime.

The policy is generic: project code may read `repos/` as reference material
and may copy selected implementation into `project/`, but the runnable
implementation inside `project/` must not depend on `repos/` at runtime via
imports, path injection, repo-local installs, or local-path dependencies.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List


TEXT_FILE_EXTENSIONS = {
    ".py",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".toml",
    ".txt",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".rst",
    ".env",
    ".pth",
    ".pyi",
}

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".eggs",
}

MAX_FILE_BYTES = 1024 * 1024
VALID_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_FILE_EXTENSIONS:
        return True
    try:
        sample = path.read_bytes()[:4096]
    except Exception:
        return False
    return b"\x00" not in sample


def _iter_project_files(project_root: Path) -> List[Path]:
    files: List[Path] = []
    for current_root, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIR_NAMES]
        current = Path(current_root)
        if current.name == "venv":
            dirnames[:] = []
            continue
        for name in filenames:
            path = current / name
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except Exception:
                continue
            if _is_probably_text(path):
                files.append(path)
    return files


def _path_markers(workspace_root: str) -> List[str]:
    repos_root = os.path.realpath(os.path.join(workspace_root, "repos"))
    return [
        repos_root.replace("\\", "/"),
        "/repos/",
        "../repos/",
        "..\\repos\\",
        "file://../repos/",
        "file:///repos/",
    ]


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _matches_repo_reference(line: str, markers: List[str]) -> bool:
    normalized = line.replace("\\", "/")
    return any(marker in normalized for marker in markers)


def _violation(rule: str, path: Path, line_no: int, line: str, project_root: Path) -> Dict[str, Any]:
    return {
        "rule": rule,
        "path": _relative_path(path, project_root),
        "line": line_no,
        "snippet": line.strip()[:400],
    }


def scan_project_self_contained(project_root: str, workspace_root: str) -> Dict[str, Any]:
    project_dir = Path(os.path.realpath(project_root))
    workspace_dir = os.path.realpath(workspace_root)
    repos_root = os.path.realpath(os.path.join(workspace_dir, "repos"))
    markers = _path_markers(workspace_dir)
    violations: List[Dict[str, Any]] = []
    checked_files = 0

    for path in _iter_project_files(project_dir):
        checked_files += 1
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            normalized = line.replace("\\", "/")
            if re.search(r"^\s*(from|import)\s+repos(\.|$)", normalized):
                violations.append(_violation("direct_repos_import", path, index, line, project_dir))
                continue
            if "sys.path" in normalized and _matches_repo_reference(normalized, markers):
                violations.append(_violation("sys_path_repo_injection", path, index, line, project_dir))
                continue
            if "PYTHONPATH" in normalized and _matches_repo_reference(normalized, markers):
                violations.append(_violation("pythonpath_repo_injection", path, index, line, project_dir))
                continue
            if (
                re.search(r"\b(pip|uv)\s+install\b", normalized)
                or "python -m pip install" in normalized
                or "poetry add" in normalized
            ) and _matches_repo_reference(normalized, markers):
                violations.append(_violation("repo_local_install", path, index, line, project_dir))
                continue
            if path.name.endswith(".pth") and _matches_repo_reference(normalized, markers):
                violations.append(_violation("pth_repo_reference", path, index, line, project_dir))
                continue
            if path.name in {"requirements.txt", "requirements-dev.txt", "requirements.in", "setup.py", "pyproject.toml"} and _matches_repo_reference(normalized, markers):
                violations.append(_violation("repo_local_dependency", path, index, line, project_dir))
                continue
    return {
        "self_contained_project": len(violations) == 0,
        "self_contained_violations": violations,
        "checked_files": checked_files,
        "project_root": str(project_dir),
        "repos_root": repos_root,
        "provenance_manifest_path": os.path.join(workspace_dir, "agent_reports", "project_code_provenance.json"),
        "provenance_manifest_present": os.path.exists(
            os.path.join(workspace_dir, "agent_reports", "project_code_provenance.json")
        ),
        "policy": {
            "repos_policy": "reference_or_copy",
            "project_must_be_self_contained": True,
        },
    }


__all__ = ["scan_project_self_contained", "VALID_ENV_NAME"]
