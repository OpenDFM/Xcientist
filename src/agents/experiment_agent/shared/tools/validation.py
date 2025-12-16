"""
Validation Utilities - Code and Specification Validation

Provides:
- Linting (syntax and style checks)
- Code-spec validation
- Interface stub extraction
- File validation utilities

Used by Manager and Integrator agents.
"""

import os
import ast
import subprocess
import tempfile
import logging
from typing import List, Dict, Any, Optional

from src.agents.experiment_agent.layers.base.schemas import ValidationResult


logger = logging.getLogger(__name__)


def run_linter(file_path: str = "", code: str = "") -> dict:
    """
    Run linter on code or file.

    Checks:
    1. Python syntax validity (using ast.parse)
    2. Style issues (using flake8)

    Args:
        file_path: Path to file to lint
        code: Code string to lint (creates temp file)

    Returns:
        Dict with keys:
        - syntax_valid: bool
        - syntax_error: str (if syntax invalid)
        - clean: bool (no style issues)
        - issues: List[Dict] (style issues)
    """
    try:
        if code:
            # Check syntax first
            try:
                ast.parse(code)
            except SyntaxError as e:
                return {
                    "syntax_valid": False,
                    "syntax_error": f"Line {e.lineno}: {e.msg}",
                    "issues": [],
                }

            # Write to temp file for flake8
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                temp_path = f.name

            try:
                result = subprocess.run(
                    ["flake8", "--max-line-length=120", temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                issues = _parse_flake8_output(result.stdout)

                return {
                    "syntax_valid": True,
                    "clean": len(issues) == 0,
                    "issues": issues,
                }
            finally:
                os.unlink(temp_path)

        elif file_path:
            # Read and check syntax
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    ast.parse(f.read())
                except SyntaxError as e:
                    return {
                        "syntax_valid": False,
                        "syntax_error": f"Line {e.lineno}: {e.msg}",
                        "issues": [],
                    }

            result = subprocess.run(
                ["flake8", "--max-line-length=120", file_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            issues = _parse_flake8_output(result.stdout)

            return {
                "syntax_valid": True,
                "clean": len(issues) == 0,
                "issues": issues,
            }

        else:
            return {"error": "Either file_path or code must be provided"}

    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except subprocess.TimeoutExpired:
        return {"error": "Linter timed out after 30 seconds"}
    except Exception as e:
        logger.error(f"run_linter failed: {e}")
        return {"error": str(e)}


def _parse_flake8_output(output: str) -> List[Dict[str, Any]]:
    """
    Parse flake8 output into structured issues list.

    Args:
        output: Raw flake8 output

    Returns:
        List of issue dictionaries with line, column, message
    """
    issues = []
    if output:
        for line in output.strip().split("\n"):
            if line:
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    issues.append(
                        {
                            "line": int(parts[1]) if parts[1].isdigit() else 0,
                            "column": (
                                int(parts[2])
                                if len(parts) > 2 and parts[2].isdigit()
                                else 0
                            ),
                            "message": parts[3].strip() if len(parts) > 3 else "",
                        }
                    )
    return issues


def validate_code_against_spec(code: str, file_spec) -> dict:
    """
    Validate that code matches specification.

    Checks:
    - All required classes are defined
    - All required functions are defined
    - Syntax is valid

    Args:
        code: Python code string
        file_spec: FileSpec object with expected classes and functions

    Returns:
        Dict with:
        - valid: bool
        - errors: List[str]
    """
    errors = []

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"valid": False, "errors": [f"Syntax error: {e}"]}

    # Collect defined names
    defined_classes = set()
    defined_functions = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            defined_classes.add(node.name)
        elif isinstance(node, ast.FunctionDef):
            defined_functions.add(node.name)

    # Check required classes
    if hasattr(file_spec, "classes"):
        for cls in file_spec.classes:
            if cls.name not in defined_classes:
                errors.append(f"Missing class: {cls.name}")

    # Check required functions
    if hasattr(file_spec, "functions"):
        for func in file_spec.functions:
            if func.name not in defined_functions:
                errors.append(f"Missing function: {func.name}")

    return {"valid": len(errors) == 0, "errors": errors}


def extract_interface_stub(file_path: str) -> dict:
    """
    Extract interface stub from a Python file.

    Creates a minimal stub showing class/function signatures without implementations.
    Useful for providing dependency context to workers.

    Args:
        file_path: Path to Python file

    Returns:
        Dict with:
        - success: bool
        - stub: str (the generated stub)
        - error: str (if failed)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        stub_lines = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                stub_lines.append(f"class {node.name}:")
                docstring = ast.get_docstring(node)
                if docstring:
                    first_line = docstring.split("\n")[0]
                    stub_lines.append(f'    """{first_line}"""')

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        args = [arg.arg for arg in item.args.args]
                        stub_lines.append(
                            f"    def {item.name}({', '.join(args)}): ..."
                        )

                stub_lines.append("")

            elif isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                stub_lines.append(f"def {node.name}({', '.join(args)}): ...")
                stub_lines.append("")

        return {"success": True, "stub": "\n".join(stub_lines)}

    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {file_path}", "stub": ""}
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error: {e}", "stub": ""}
    except Exception as e:
        logger.error(f"extract_interface_stub failed: {e}")
        return {"success": False, "error": str(e), "stub": ""}


def validate_file_exists(file_path: str, project_root: str = "") -> ValidationResult:
    """
    Validate that a file exists.

    Args:
        file_path: Relative or absolute path to file
        project_root: Optional project root for relative paths

    Returns:
        ValidationResult
    """
    result = ValidationResult(valid=True)

    if project_root and not os.path.isabs(file_path):
        full_path = os.path.join(project_root, file_path)
    else:
        full_path = file_path

    if not os.path.exists(full_path):
        result.add_error(f"File not found: {full_path}")

    return result


def validate_python_file(file_path: str) -> ValidationResult:
    """
    Validate a Python file (syntax check).

    Args:
        file_path: Path to Python file

    Returns:
        ValidationResult with syntax errors if any
    """
    result = ValidationResult(valid=True)

    if not os.path.exists(file_path):
        result.add_error(f"File not found: {file_path}")
        return result

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        ast.parse(content)

    except SyntaxError as e:
        result.add_error(f"Syntax error at line {e.lineno}: {e.msg}")
    except Exception as e:
        result.add_error(f"Failed to read file: {e}")

    return result


def validate_imports(code: str, available_modules: List[str]) -> ValidationResult:
    """
    Validate that all imports in code are available.

    Args:
        code: Python code string
        available_modules: List of available module names

    Returns:
        ValidationResult with import errors if any
    """
    result = ValidationResult(valid=True)

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    # Skip standard library and third-party modules
                    if module_name not in available_modules:
                        result.add_warning(f"Unverified import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name not in available_modules:
                        result.add_warning(f"Unverified import: {node.module}")

    except SyntaxError as e:
        result.add_error(f"Syntax error: {e}")

    return result


def count_lines_of_code(code: str) -> Dict[str, int]:
    """
    Count lines of code, comments, and blank lines.

    Args:
        code: Python code string

    Returns:
        Dict with code_lines, comment_lines, blank_lines, total_lines
    """
    lines = code.split("\n")
    code_lines = 0
    comment_lines = 0
    blank_lines = 0

    in_multiline_string = False
    multiline_char = None

    for line in lines:
        stripped = line.strip()

        # Check for multiline strings
        if not in_multiline_string:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                multiline_char = stripped[:3]
                if stripped.count(multiline_char) >= 2:
                    # Single line docstring
                    comment_lines += 1
                else:
                    in_multiline_string = True
                    comment_lines += 1
                continue
        else:
            comment_lines += 1
            if multiline_char in stripped:
                in_multiline_string = False
            continue

        # Blank lines
        if not stripped:
            blank_lines += 1
            continue

        # Comment lines
        if stripped.startswith("#"):
            comment_lines += 1
            continue

        # Code lines
        code_lines += 1

    return {
        "code_lines": code_lines,
        "comment_lines": comment_lines,
        "blank_lines": blank_lines,
        "total_lines": len(lines),
    }


__all__ = [
    "run_linter",
    "validate_code_against_spec",
    "extract_interface_stub",
    "validate_file_exists",
    "validate_python_file",
    "validate_imports",
    "count_lines_of_code",
]
