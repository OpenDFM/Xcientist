"""
File operation tools for experiment agents.

Provides tools for reading, writing, and managing files.
Optimized for minimal toolset strategy:
- read_file: Core perception
- write_file: Core creation
- edit_file: Core modification
- list_directory: Core perception (structured)

Other operations (rm, mkdir, cp, etc.) should be performed via run_shell_command.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from agents import function_tool


def _validate_path_security(
    file_path: str,
    allowed_root: Optional[str] = None,
    allow_read_only: bool = False,
) -> tuple[bool, str, str]:
    """
    Validate path security with read/write scope separation.

    Strategy:
    - Write operations: Must be within PROJECT_ROOT
    - Read operations: Can be within WORKSPACE_ROOT (if allow_read_only=True)

    Args:
        file_path: The path to validate
        allowed_root: The root directory to enforce. If None, loads from config based on mode.
        allow_read_only: If True, allows access to WORKSPACE_ROOT (for reading papers/repos).
    """
    try:
        # Load config
        from src.agents.experiment_agent.config import PROJECT_ROOT, WORKSPACE_ROOT

        # Determine effective root
        # If allowed_root explicitly provided, use it.
        # Else if allow_read_only, use WORKSPACE_ROOT (broader scope).
        # Else (default write mode), use PROJECT_ROOT (narrow scope).
        if allowed_root:
            root_dir = allowed_root
        elif allow_read_only:
            root_dir = WORKSPACE_ROOT if WORKSPACE_ROOT else PROJECT_ROOT
        else:
            root_dir = PROJECT_ROOT

        # If config not initialized (fallback), allow absolute paths temporarily (legacy behavior)
        if not root_dir:
            abs_path = os.path.abspath(os.path.expanduser(file_path))
            return True, abs_path, ""

        # Resolution Strategy for Relative Paths:
        # 1. If allow_read_only=True (read mode):
        #    a. Try checking if it exists in PROJECT_ROOT (primary implementation context)
        #    b. Try checking if it exists in WORKSPACE_ROOT (secondary reference context)
        #    c. Default to WORKSPACE_ROOT for validation (broadest scope)
        # 2. If allow_read_only=False (write mode):
        #    a. Always resolve against PROJECT_ROOT

        abs_file_path_raw = os.path.abspath(os.path.expanduser(file_path))

        # Check if it's already an absolute path
        if os.path.isabs(os.path.expanduser(file_path)):
            # If absolute, validation logic below will handle it against root_dir
            abs_path = abs_file_path_raw
            abs_root = os.path.abspath(os.path.expanduser(root_dir))
        else:
            # Relative path resolution
            from src.agents.experiment_agent.config import PROJECT_ROOT, WORKSPACE_ROOT

            if allow_read_only:
                # READ MODE: Always validate against WORKSPACE_ROOT for read-only operations
                # This allows reading from project/, repos/, dataset_candidate/, etc.
                workspace_root_abs = (
                    os.path.abspath(os.path.expanduser(WORKSPACE_ROOT))
                    if WORKSPACE_ROOT
                    else ""
                )
                project_root_abs = (
                    os.path.abspath(os.path.expanduser(PROJECT_ROOT))
                    if PROJECT_ROOT
                    else ""
                )

                # Try resolving the path
                project_abs = (
                    os.path.abspath(os.path.join(project_root_abs, file_path))
                    if project_root_abs
                    else ""
                )
                workspace_abs = (
                    os.path.abspath(os.path.join(workspace_root_abs, file_path))
                    if workspace_root_abs
                    else ""
                )

                # Determine which resolved path exists
                if project_abs and os.path.exists(project_abs):
                    abs_path = project_abs
                elif workspace_abs and os.path.exists(workspace_abs):
                    abs_path = workspace_abs
                else:
                    # Default to workspace-based resolution
                    abs_root = (
                        workspace_root_abs if workspace_root_abs else project_root_abs
                    )
                    abs_path = os.path.abspath(os.path.join(abs_root, file_path))

                # For read-only operations, ALWAYS validate against WORKSPACE_ROOT (broader scope)
                # This is the key fix: abs_root should be WORKSPACE_ROOT, not PROJECT_ROOT
                abs_root = (
                    workspace_root_abs if workspace_root_abs else project_root_abs
                )
            else:
                # WRITE MODE: Always PROJECT_ROOT
                abs_root = os.path.abspath(os.path.expanduser(PROJECT_ROOT))
                abs_path = os.path.abspath(os.path.join(abs_root, file_path))

        # Check common path
        try:
            common = os.path.commonpath([abs_root, abs_path])
            if common == abs_root or abs_path.startswith(abs_root + os.sep):
                return True, abs_path, ""

            error_msg = (
                f"Security Error: Path '{file_path}' is outside the allowed scope.\n"
                f"Scope Root: {abs_root}\n"
                f"Attempted: {abs_path}\n"
            )
            return False, abs_path, error_msg

        except ValueError:
            return False, abs_path, "Path validation failed (drive mismatch)."

    except Exception as e:
        return False, "", f"Validation error: {str(e)}"


# Limits for large file handling
MAX_READ_BYTES = 100 * 1024  # 100KB
MAX_READ_LINES = 2000


@function_tool
def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Read content from a file.
    Scope: WORKSPACE_ROOT (Can read project/, repos/, papers/, datasets/)

    For large files (>100KB or >2000 lines), returns the first portion with a truncation notice.
    """
    # Allow reading from entire workspace
    is_valid, abs_path, error_msg = _validate_path_security(
        file_path, allow_read_only=True
    )

    if not is_valid:
        return {"success": False, "error": error_msg}

    try:
        file_size = os.path.getsize(abs_path)
        truncated = False
        truncation_reason = ""

        # Read file with size limit
        with open(abs_path, "r", encoding=encoding, errors="replace") as f:
            if file_size > MAX_READ_BYTES:
                # Read only first MAX_READ_BYTES
                content = f.read(MAX_READ_BYTES)
                truncated = True
                truncation_reason = f"File too large ({file_size:,} bytes). Showing first {MAX_READ_BYTES:,} bytes."
            else:
                content = f.read()

        # Check line count and truncate if needed
        lines = content.split("\n")
        total_lines_read = len(lines)

        if not truncated and total_lines_read > MAX_READ_LINES:
            # Truncate by lines
            content = "\n".join(lines[:MAX_READ_LINES])
            truncated = True
            truncation_reason = (
                f"File has many lines. Showing first {MAX_READ_LINES} lines."
            )

        # Count total lines in original file if truncated
        if truncated:
            # For large files, estimate total lines
            if file_size > MAX_READ_BYTES:
                # Estimate based on average line length from what we read
                avg_line_len = MAX_READ_BYTES / max(total_lines_read, 1)
                estimated_total_lines = (
                    int(file_size / avg_line_len)
                    if avg_line_len > 0
                    else total_lines_read
                )
                line_count_info = f"~{estimated_total_lines:,} (estimated)"
            else:
                line_count_info = str(total_lines_read)

            content += f"\n\n... [TRUNCATED] {truncation_reason} ..."
        else:
            line_count_info = str(content.count("\n") + 1)

        result = {
            "success": True,
            "content": content,
            "file_path": abs_path,
            "size_bytes": file_size,
            "line_count": line_count_info,
            "truncated": truncated,
        }

        if truncated:
            result["truncation_note"] = truncation_reason

        return result

    except Exception as e:
        return {"success": False, "error": f"Read error: {str(e)}"}


@function_tool
def write_file(
    file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
) -> Dict[str, Any]:
    """
    Write content to a file.
    Scope: PROJECT_ROOT (Strictly limited to project directory)
    """
    # Strict write scope
    is_valid, abs_path, error_msg = _validate_path_security(
        file_path, allow_read_only=False
    )

    if not is_valid:
        return {"success": False, "error": error_msg}

    try:
        if create_dirs:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        with open(abs_path, "w", encoding=encoding) as f:
            f.write(content)

        return {
            "success": True,
            "message": f"Written to {abs_path}",
            "file_path": abs_path,
            "size_bytes": len(content.encode(encoding)),
        }
    except Exception as e:
        return {"success": False, "error": f"Write error: {str(e)}"}


@function_tool
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Edit a file.
    Scope: PROJECT_ROOT
    """
    is_valid, abs_path, error_msg = _validate_path_security(
        file_path, allow_read_only=False
    )
    if not is_valid:
        return {"success": False, "error": error_msg}

    # ... (rest of logic remains same, just wrapping security check)
    try:
        if not os.path.exists(abs_path):
            return {"success": False, "error": f"File not found: {abs_path}"}

        with open(abs_path, "r", encoding=encoding) as f:
            content = f.read()

        if old_string not in content:
            return {"success": False, "error": "String not found"}

        count = content.count(old_string)
        if replace_all:
            new_content = content.replace(old_string, new_string)
            replaced = count
        else:
            new_content = content.replace(old_string, new_string, 1)
            replaced = 1

        with open(abs_path, "w", encoding=encoding) as f:
            f.write(new_content)

        return {"success": True, "replaced_count": replaced, "file_path": abs_path}

    except Exception as e:
        return {"success": False, "error": f"Edit error: {str(e)}"}


@function_tool
def list_directory(
    directory_path: str, pattern: Optional[str] = None, recursive: bool = False
) -> Dict[str, Any]:
    """
    List files.
    Scope: WORKSPACE_ROOT (Can explore repos/ papers/ etc.)
    """
    # Allow listing entire workspace
    is_valid, abs_path, error_msg = _validate_path_security(
        directory_path, allow_read_only=True
    )

    if not is_valid:
        return {"success": False, "error": error_msg}

    try:
        path = Path(abs_path)
        if not path.exists():
            return {"success": False, "error": "Directory not found"}

        # ... (rest of logic same)
        if recursive and pattern:
            files = [str(p) for p in path.rglob(pattern)]
        elif recursive:
            files = [str(p) for p in path.rglob("*")]
        elif pattern:
            files = [str(p) for p in path.glob(pattern)]
        else:
            files = [str(p) for p in path.iterdir()]

        file_list = []
        dir_list = []
        for item in files:
            p = Path(item)
            if p.is_file():
                file_list.append(
                    {"path": str(p), "name": p.name, "size": p.stat().st_size}
                )
            elif p.is_dir():
                dir_list.append({"path": str(p), "name": p.name})

        return {
            "success": True,
            "directory": abs_path,
            "files": file_list,
            "directories": dir_list,
        }
    except Exception as e:
        return {"success": False, "error": f"List error: {str(e)}"}
