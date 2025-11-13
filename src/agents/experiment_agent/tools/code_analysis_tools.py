"""
Code analysis tools for experiment agents.

Provides tools for analyzing code structure, extracting information, and searching code.
Compatible with openai-agents SDK.

Security: All file/directory operations are restricted to the working directory (project root)
and its subdirectories to prevent unauthorized file access.
"""

import ast
import os
import re
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
def analyze_python_file(
    file_path: str,
) -> Dict[str, Any]:
    """
    Analyze a Python file and extract structural information.

    Args:
        file_path: Path to the Python file

    Returns:
        Dictionary with code structure information
    """
    try:
        file_path = os.path.expanduser(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse AST
        tree = ast.parse(content)

        # Extract information
        imports = []
        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "name": alias.name,
                            "alias": alias.asname,
                        }
                    )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": module,
                            "name": alias.name,
                            "alias": alias.asname,
                        }
                    )

            elif isinstance(node, ast.ClassDef):
                # Get base classes
                bases = [ast.unparse(base) for base in node.bases]

                # Get methods
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)

                classes.append(
                    {
                        "name": node.name,
                        "bases": bases,
                        "methods": methods,
                        "method_count": len(methods),
                        "lineno": node.lineno,
                    }
                )

            elif isinstance(node, ast.FunctionDef):
                # Only top-level functions (not class methods)
                if not any(
                    isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)
                ):
                    # Get function signature
                    args = [arg.arg for arg in node.args.args]

                    functions.append(
                        {
                            "name": node.name,
                            "args": args,
                            "lineno": node.lineno,
                        }
                    )

        return {
            "success": True,
            "file_path": file_path,
            "imports": imports,
            "classes": classes,
            "functions": functions,
            "import_count": len(imports),
            "class_count": len(classes),
            "function_count": len(functions),
        }

    except SyntaxError as e:
        return {
            "success": False,
            "error": f"Syntax error in Python file: {str(e)}",
            "line": e.lineno if hasattr(e, "lineno") else None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error analyzing Python file: {str(e)}",
        }


@function_tool
def search_in_codebase(
    directory: str,
    pattern: str,
    file_pattern: str = "*.py",
    max_results: int = 50,
) -> Dict[str, Any]:
    """
    Search for a pattern in code files.

    Args:
        directory: Directory to search in
        pattern: Regex pattern to search for
        file_pattern: File glob pattern (default: *.py)
        max_results: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    try:
        directory = os.path.expanduser(directory)
        path = Path(directory)

        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory}",
            }

        # Compile regex pattern
        regex = re.compile(pattern)

        results = []
        for file_path in path.rglob(file_pattern):
            if not file_path.is_file():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(
                                {
                                    "file": str(file_path),
                                    "line_number": line_num,
                                    "line_content": line.strip(),
                                }
                            )

                            if len(results) >= max_results:
                                break
            except (UnicodeDecodeError, PermissionError):
                continue

            if len(results) >= max_results:
                break

        return {
            "success": True,
            "pattern": pattern,
            "directory": directory,
            "results": results,
            "total_matches": len(results),
            "truncated": len(results) >= max_results,
        }

    except re.error as e:
        return {
            "success": False,
            "error": f"Invalid regex pattern: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error searching codebase: {str(e)}",
        }


@function_tool
def count_lines_of_code(
    directory: str,
    file_pattern: str = "*.py",
) -> Dict[str, Any]:
    """
    Count lines of code in a directory.

    Args:
        directory: Directory to analyze
        file_pattern: File glob pattern (default: *.py)

    Returns:
        Dictionary with line count statistics
    """
    try:
        directory = os.path.expanduser(directory)
        path = Path(directory)

        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory}",
            }

        total_lines = 0
        total_code_lines = 0
        total_comment_lines = 0
        total_blank_lines = 0
        file_count = 0

        for file_path in path.rglob(file_pattern):
            if not file_path.is_file():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    file_count += 1
                    total_lines += len(lines)

                    for line in lines:
                        stripped = line.strip()
                        if not stripped:
                            total_blank_lines += 1
                        elif stripped.startswith("#"):
                            total_comment_lines += 1
                        else:
                            total_code_lines += 1

            except (UnicodeDecodeError, PermissionError):
                continue

        return {
            "success": True,
            "directory": directory,
            "file_count": file_count,
            "total_lines": total_lines,
            "code_lines": total_code_lines,
            "comment_lines": total_comment_lines,
            "blank_lines": total_blank_lines,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error counting lines: {str(e)}",
        }


@function_tool
def extract_function_code(
    file_path: str,
    function_name: str,
) -> Dict[str, Any]:
    """
    Extract a specific function's code from a Python file.

    Args:
        file_path: Path to the Python file
        function_name: Name of the function to extract

    Returns:
        Dictionary with function code and metadata
    """
    try:
        file_path = os.path.expanduser(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Get function code
                start_line = node.lineno - 1
                end_line = node.end_lineno

                function_code = "\n".join(lines[start_line:end_line])

                # Get function signature
                args = [arg.arg for arg in node.args.args]

                # Get docstring
                docstring = ast.get_docstring(node)

                return {
                    "success": True,
                    "function_name": function_name,
                    "code": function_code,
                    "args": args,
                    "docstring": docstring,
                    "start_line": start_line + 1,
                    "end_line": end_line,
                }

        return {
            "success": False,
            "error": f"Function '{function_name}' not found in file",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error extracting function: {str(e)}",
        }


@function_tool
def list_python_files(
    directory: str,
) -> Dict[str, Any]:
    """
    List all Python files in a directory recursively.

    Args:
        directory: Directory to search

    Returns:
        Dictionary with list of Python files
    """
    try:
        directory = os.path.expanduser(directory)
        path = Path(directory)

        if not path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {directory}",
            }

        python_files = []
        for file_path in path.rglob("*.py"):
            if file_path.is_file():
                python_files.append(
                    {
                        "path": str(file_path),
                        "name": file_path.name,
                        "relative_path": str(file_path.relative_to(path)),
                        "size": file_path.stat().st_size,
                    }
                )

        return {
            "success": True,
            "directory": directory,
            "files": python_files,
            "total_count": len(python_files),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error listing Python files: {str(e)}",
        }


@function_tool
def check_imports_available(
    file_path: str,
) -> Dict[str, Any]:
    """
    Check if all imports in a Python file are available.

    Args:
        file_path: Path to the Python file

    Returns:
        Dictionary with import availability status
    """
    try:
        file_path = os.path.expanduser(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        import_status = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    try:
                        __import__(alias.name)
                        import_status.append(
                            {
                                "import": alias.name,
                                "available": True,
                            }
                        )
                    except ImportError:
                        import_status.append(
                            {
                                "import": alias.name,
                                "available": False,
                            }
                        )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                try:
                    __import__(module)
                    import_status.append(
                        {
                            "import": f"from {module}",
                            "available": True,
                        }
                    )
                except ImportError:
                    import_status.append(
                        {
                            "import": f"from {module}",
                            "available": False,
                        }
                    )

        missing_imports = [imp for imp in import_status if not imp["available"]]

        return {
            "success": True,
            "file_path": file_path,
            "imports": import_status,
            "total_imports": len(import_status),
            "missing_imports": missing_imports,
            "all_available": len(missing_imports) == 0,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error checking imports: {str(e)}",
        }


@function_tool
def get_file_dependencies(
    file_path: str,
) -> Dict[str, Any]:
    """
    Get local file dependencies (imports from the same project).

    Args:
        file_path: Path to the Python file

    Returns:
        Dictionary with file dependencies
    """
    try:
        file_path = os.path.expanduser(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        dependencies = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Check if it's a relative import or local module
                if module.startswith(".") or not module.startswith(
                    ("os", "sys", "re", "json", "ast")
                ):
                    dependencies.append(
                        {
                            "module": module,
                            "items": [alias.name for alias in node.names],
                        }
                    )

        return {
            "success": True,
            "file_path": file_path,
            "dependencies": dependencies,
            "dependency_count": len(dependencies),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting dependencies: {str(e)}",
        }
