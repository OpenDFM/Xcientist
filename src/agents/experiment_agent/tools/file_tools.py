"""
File operation tools for experiment agents.

Provides tools for reading, writing, and managing files and directories.
Compatible with openai-agents SDK.

Security: All file operations are restricted to the working directory (project root)
and its subdirectories to prevent unauthorized file access.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from agents import function_tool


def _validate_path_security(
    file_path: str, working_dir: Optional[str] = None
) -> tuple[bool, str, str]:
    """
    Validate that the file path is within the allowed working directory.

    Args:
        file_path: The path to validate
        working_dir: The allowed working directory (project root). If None, loads from config.

    Returns:
        Tuple of (is_valid, absolute_path, error_message)
        - is_valid: True if path is safe, False otherwise
        - absolute_path: Resolved absolute path
        - error_message: Error message if not valid, empty string otherwise
    """
    try:
        # Load working_dir from config if not provided
        if working_dir is None:
            from src.agents.experiment_agent.config import get_path_config

            path_config = get_path_config()
            working_dir = path_config.get("working_dir")

        # If still no working_dir, allow operation (backward compatibility)
        if not working_dir:
            abs_path = os.path.abspath(os.path.expanduser(file_path))
            return True, abs_path, ""

        # Resolve both paths to absolute
        abs_working_dir = os.path.abspath(os.path.expanduser(working_dir))
        abs_file_path = os.path.abspath(os.path.expanduser(file_path))

        # Check if file_path is within working_dir or its subdirectories
        # Use os.path.commonpath to check if they share a common root
        try:
            common_path = os.path.commonpath([abs_working_dir, abs_file_path])
            # File is safe if common path is the working directory
            if common_path == abs_working_dir or abs_file_path.startswith(
                abs_working_dir + os.sep
            ):
                return True, abs_file_path, ""
            else:
                error_msg = (
                    f"Security Error: Path '{file_path}' is outside the allowed working directory.\n"
                    f"Allowed: {abs_working_dir} and its subdirectories\n"
                    f"Attempted: {abs_file_path}\n"
                    f"All file operations must be within the project directory."
                )
                return False, abs_file_path, error_msg
        except ValueError:
            # Paths are on different drives (Windows)
            error_msg = (
                f"Security Error: Path '{file_path}' is on a different drive than working directory.\n"
                f"Allowed: {abs_working_dir}\n"
                f"Attempted: {abs_file_path}"
            )
            return False, abs_file_path, error_msg

    except Exception as e:
        return False, "", f"Path validation error: {str(e)}"


@function_tool
def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Read content from a file.

    Security: Only files within the project directory (working_dir) can be read.

    Args:
        file_path: Path to the file to read (must be within working directory)
        encoding: File encoding (default: utf-8)

    Returns:
        Dictionary with success status, content, and metadata
    """
    # Validate path security
    is_valid, abs_path, error_msg = _validate_path_security(file_path)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
        }

    try:
        with open(abs_path, "r", encoding=encoding) as f:
            content = f.read()

        file_size = os.path.getsize(abs_path)
        line_count = content.count("\n") + 1

        return {
            "success": True,
            "content": content,
            "file_path": abs_path,
            "size_bytes": file_size,
            "line_count": line_count,
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"File not found: {abs_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {str(e)}",
        }


@function_tool
def write_file(
    file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
) -> Dict[str, Any]:
    """
    Write content to a file.

    Security: Only files within the project directory (working_dir) can be written.

    Args:
        file_path: Path to the file to write (must be within working directory)
        content: Content to write
        encoding: File encoding (default: utf-8)
        create_dirs: Create parent directories if they don't exist

    Returns:
        Dictionary with success status and message
    """
    # Validate path security
    is_valid, abs_path, error_msg = _validate_path_security(file_path)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
        }

    try:
        if create_dirs:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        with open(abs_path, "w", encoding=encoding) as f:
            f.write(content)

        file_size = os.path.getsize(abs_path)

        return {
            "success": True,
            "message": f"Successfully wrote {file_size} bytes to {abs_path}",
            "file_path": abs_path,
            "size_bytes": file_size,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error writing file: {str(e)}",
        }


@function_tool
def list_directory(
    directory_path: str, pattern: Optional[str] = None, recursive: bool = False
) -> Dict[str, Any]:
    """
    List files and directories in a directory.

    Security: Only directories within the project directory (working_dir) can be listed.

    Args:
        directory_path: Path to the directory (must be within working directory)
        pattern: Optional glob pattern to filter files (e.g., "*.py")
        recursive: List files recursively

    Returns:
        Dictionary with success status and list of files/directories
    """
    # Validate path security
    is_valid, abs_path, error_msg = _validate_path_security(directory_path)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
        }

    try:
        path = Path(abs_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {abs_path}",
            }

        if not path.is_dir():
            return {
                "success": False,
                "error": f"Path is not a directory: {abs_path}",
            }

        if recursive and pattern:
            files = [str(p) for p in path.rglob(pattern)]
        elif recursive:
            files = [str(p) for p in path.rglob("*")]
        elif pattern:
            files = [str(p) for p in path.glob(pattern)]
        else:
            files = [str(p) for p in path.iterdir()]

        # Separate files and directories
        file_list = []
        dir_list = []
        for item in files:
            item_path = Path(item)
            if item_path.is_file():
                file_list.append(
                    {
                        "path": str(item),
                        "name": item_path.name,
                        "size": item_path.stat().st_size,
                    }
                )
            elif item_path.is_dir():
                dir_list.append(
                    {
                        "path": str(item),
                        "name": item_path.name,
                    }
                )

        return {
            "success": True,
            "directory": abs_path,
            "files": file_list,
            "directories": dir_list,
            "total_files": len(file_list),
            "total_directories": len(dir_list),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error listing directory: {str(e)}",
        }


@function_tool
def create_directory(directory_path: str) -> Dict[str, Any]:
    """
    Create a directory (and parent directories if needed).

    Security: Only directories within the project directory (working_dir) can be created.

    Args:
        directory_path: Path to the directory to create (must be within working directory)

    Returns:
        Dictionary with success status and message
    """
    # Validate path security
    is_valid, abs_path, error_msg = _validate_path_security(directory_path)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
        }

    try:
        os.makedirs(abs_path, exist_ok=True)

        return {
            "success": True,
            "message": f"Directory created: {abs_path}",
            "path": abs_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error creating directory: {str(e)}",
        }


@function_tool
def delete_file(file_path: str) -> Dict[str, Any]:
    """
    Delete a file.

    Security: Only files within the project directory (working_dir) can be deleted.

    Args:
        file_path: Path to the file to delete (must be within working directory)

    Returns:
        Dictionary with success status and message
    """
    # Validate path security
    is_valid, abs_path, error_msg = _validate_path_security(file_path)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
        }

    try:
        if not os.path.exists(abs_path):
            return {
                "success": False,
                "error": f"File not found: {abs_path}",
            }

        if os.path.isdir(abs_path):
            return {
                "success": False,
                "error": f"Path is a directory, use delete_directory instead: {abs_path}",
            }

        os.remove(abs_path)

        return {
            "success": True,
            "message": f"File deleted: {abs_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error deleting file: {str(e)}",
        }


@function_tool
def copy_file(source_path: str, destination_path: str) -> Dict[str, Any]:
    """
    Copy a file from source to destination.

    Security: Both paths must be within the project directory (working_dir).

    Args:
        source_path: Path to the source file (must be within working directory)
        destination_path: Path to the destination (must be within working directory)

    Returns:
        Dictionary with success status and message
    """
    # Validate source path security
    is_valid_src, abs_src, error_msg_src = _validate_path_security(source_path)
    if not is_valid_src:
        return {
            "success": False,
            "error": f"Source path: {error_msg_src}",
        }

    # Validate destination path security
    is_valid_dst, abs_dst, error_msg_dst = _validate_path_security(destination_path)
    if not is_valid_dst:
        return {
            "success": False,
            "error": f"Destination path: {error_msg_dst}",
        }

    try:
        # Create destination directory if needed
        os.makedirs(os.path.dirname(abs_dst), exist_ok=True)

        shutil.copy2(abs_src, abs_dst)

        return {
            "success": True,
            "message": f"File copied from {abs_src} to {abs_dst}",
            "source": abs_src,
            "destination": abs_dst,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error copying file: {str(e)}",
        }


@function_tool
def file_exists(file_path: str) -> Dict[str, Any]:
    """
    Check if a file or directory exists.

    Security: Only paths within the project directory (working_dir) can be checked.

    Args:
        file_path: Path to check (must be within working directory)

    Returns:
        Dictionary with existence status and type
    """
    # Validate path security
    is_valid, abs_path, error_msg = _validate_path_security(file_path)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
        }

    try:
        exists = os.path.exists(abs_path)

        if exists:
            is_file = os.path.isfile(abs_path)
            is_dir = os.path.isdir(abs_path)

            return {
                "success": True,
                "exists": True,
                "path": abs_path,
                "is_file": is_file,
                "is_directory": is_dir,
            }
        else:
            return {
                "success": True,
                "exists": False,
                "path": abs_path,
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error checking file: {str(e)}",
        }


@function_tool
def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a file.

    Security: Only paths within the project directory (working_dir) can be accessed.

    Args:
        file_path: Path to the file (must be within working directory)

    Returns:
        Dictionary with file information
    """
    # Validate path security
    is_valid, abs_path, error_msg = _validate_path_security(file_path)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
        }

    try:
        path = Path(abs_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {abs_path}",
            }

        stat = path.stat()

        return {
            "success": True,
            "path": str(path),
            "name": path.name,
            "extension": path.suffix,
            "size_bytes": stat.st_size,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting file info: {str(e)}",
        }
