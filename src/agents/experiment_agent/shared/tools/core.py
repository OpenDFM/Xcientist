"""
Core tools for SuperAgent - used by all agents.

Tools:
- bash: Execute shell commands
- file_viewer: View file with line numbers
- write_file: Write file to disk
- edit_file: Edit file with string replacement

Validation utilities are imported from src.agents.experiment_agent.shared.tools.validation:
- run_linter: Run syntax and style checks
- validate_code_against_spec: Validate code matches specification
- extract_interface_stub: Extract interface stub from Python file

Tool collections:
- get_architect_tools: Tools for Architect agents
- get_worker_tools: Tools for Worker agents
- get_integrator_tools: Tools for Integrator agents
"""

import subprocess
import os
import logging
from typing import List

from agents import function_tool

# Import validation utilities from dedicated module
from src.agents.experiment_agent.shared.tools.validation import (
    run_linter,
    validate_code_against_spec,
    extract_interface_stub,
)


logger = logging.getLogger(__name__)

DEFAULT_BASH_TIMEOUT_SECONDS = int(os.getenv("AGENT_BASH_TIMEOUT_SECONDS", "600"))


# =============================================================================
# Security Context
# =============================================================================


class SecurityContext:
    """Manages security boundaries for file operations."""

    _project_root: str = ""
    _workspace_root: str = ""

    @classmethod
    def set_roots(cls, project_root: str, workspace_root: str = None):
        cls._project_root = os.path.abspath(project_root)
        cls._workspace_root = os.path.abspath(workspace_root or project_root)

    @classmethod
    def get_project_root(cls) -> str:
        return cls._project_root

    @classmethod
    def get_workspace_root(cls) -> str:
        return cls._workspace_root


# =============================================================================
# Core Tools (Agent-callable)
# =============================================================================


@function_tool
def bash(command: str, working_dir: str = "") -> dict:
    """
    Execute a shell command. Common usages:
    - Search code: grep -rn "pattern" path/ --include="*.py"
    - List files: ls -la, find . -name "*.py"
    - Directory structure: tree -L 2
    - Check syntax: python -m py_compile file.py
    - Run linter: flake8 --max-line-length=120 file.py

    After using grep to find code, use file_viewer to examine it.

    Args:
        command: Shell command to execute
        working_dir: Working directory (defaults to workspace root)

    Returns:
        Dict with stdout, stderr, return_code, success
    """
    try:
        # Default to workspace root
        cwd = working_dir if working_dir else SecurityContext.get_workspace_root()
        if not cwd:
            cwd = SecurityContext.get_project_root()
        if not cwd:
            cwd = os.getcwd()

        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=DEFAULT_BASH_TIMEOUT_SECONDS,
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {command[:50]}...")
        return {
            "success": False,
            "stderr": f"Command timed out after {DEFAULT_BASH_TIMEOUT_SECONDS} seconds",
            "return_code": -1,
        }
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return {
            "success": False,
            "stderr": str(e),
            "return_code": -1,
        }


@function_tool
def file_viewer(file_path: str, start_line: int = 1, end_line: int = -1) -> dict:
    """
    View file content with line numbers.

    Workflow: grep -> find line numbers -> file_viewer to see context

    Args:
        file_path: Path to the file
        start_line: Starting line number (1-indexed, default: 1)
        end_line: Ending line number (-1 for end of file)

    Returns:
        Dict with numbered content
    """
    try:
        # Handle relative paths
        if not os.path.isabs(file_path):
            project_root = SecurityContext.get_project_root()
            workspace_root = SecurityContext.get_workspace_root()

            if project_root and os.path.exists(os.path.join(project_root, file_path)):
                full_path = os.path.join(project_root, file_path)
            elif workspace_root and os.path.exists(
                os.path.join(workspace_root, file_path)
            ):
                full_path = os.path.join(workspace_root, file_path)
            else:
                full_path = file_path
        else:
            full_path = file_path

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start_idx = max(0, start_line - 1)
        end_idx = total_lines if end_line == -1 else min(end_line, total_lines)

        numbered_lines = []
        for i in range(start_idx, end_idx):
            line_num = i + 1
            numbered_lines.append(f"{line_num:4d}|{lines[i].rstrip()}")

        return {
            "success": True,
            "content": "\n".join(numbered_lines),
            "file_path": full_path,
            "total_lines": total_lines,
            "showing": f"lines {start_idx + 1}-{end_idx}",
        }
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"file_viewer failed: {e}")
        return {"success": False, "error": str(e)}


@function_tool
def write_file(file_path: str, content: str) -> dict:
    """
    Write content to a file. Creates parent directories if needed.

    Args:
        file_path: Path to the file to write
        content: Content to write to the file

    Returns:
        Dict with success status
    """
    try:
        if not os.path.isabs(file_path):
            project_root = SecurityContext.get_project_root()
            if project_root:
                full_path = os.path.join(project_root, file_path)
            else:
                full_path = file_path
        else:
            full_path = file_path

        parent_dir = os.path.dirname(full_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.debug(f"Wrote {len(content)} chars to {full_path}")
        return {
            "success": True,
            "file_path": full_path,
            "message": f"Written {len(content)} chars, {len(content.splitlines())} lines",
        }
    except Exception as e:
        logger.error(f"write_file failed: {e}")
        return {"success": False, "error": str(e)}


@function_tool
def edit_file(file_path: str, old_string: str, new_string: str) -> dict:
    """
    Edit a file by replacing old_string with new_string.

    Args:
        file_path: Path to the file to edit
        old_string: String to find and replace
        new_string: String to replace with

    Returns:
        Dict with success status
    """
    try:
        if not os.path.isabs(file_path):
            project_root = SecurityContext.get_project_root()
            if project_root:
                full_path = os.path.join(project_root, file_path)
            else:
                full_path = file_path
        else:
            full_path = file_path

        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        if old_string not in content:
            return {"success": False, "error": "old_string not found in file"}

        new_content = content.replace(old_string, new_string, 1)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        logger.debug(f"Edited {full_path}")
        return {"success": True, "file_path": full_path, "message": "File edited"}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"edit_file failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# Validation Utilities - Re-exported from src.agents.experiment_agent.shared.tools.validation
# =============================================================================
# These functions are imported from src.agents.experiment_agent.shared.tools.validation for backward compatibility:
# - run_linter
# - validate_code_against_spec
# - extract_interface_stub


# =============================================================================
# Tool Collections
# =============================================================================


def get_architect_tools() -> List:
    """
    Get tools available to Architect agents.

    Architect needs: exploration (bash, file_viewer)
    Does NOT need: write/edit (only designs, doesn't implement)
    """
    return [bash, file_viewer]


def get_worker_tools() -> List:
    """
    Get tools available to Worker agents.

    Worker needs: exploration (bash, file_viewer) + writing (write_file, edit_file)
    """
    return [bash, file_viewer, write_file, edit_file]


def get_integrator_tools() -> List:
    """
    Get tools available to Integrator agents.

    Integrator needs: exploration (bash, file_viewer) for verification
    May optionally need edit for fixes
    """
    return [bash, file_viewer]


# =============================================================================
# Backward Compatibility Exports
# =============================================================================

__all__ = [
    # Security
    "SecurityContext",
    # Core tools
    "bash",
    "file_viewer",
    "write_file",
    "edit_file",
    # Validation
    "run_linter",
    "validate_code_against_spec",
    "extract_interface_stub",
    # Tool collections
    "get_architect_tools",
    "get_worker_tools",
    "get_integrator_tools",
]
