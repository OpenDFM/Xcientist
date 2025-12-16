"""
Traceback Parser - Extract structured information from pytest output.

Parses pytest tracebacks and extracts:
- Call stack frames
- Root cause exceptions
- Assertion details
- Implementation files involved
"""

import os
import re
import ast
from typing import Dict, List, Tuple

from src.agents.experiment_agent.layers.code.schemas.integration import StackFrame, TestFailureInfo


class TracebackParser:
    """
    Parse pytest output and extract structured failure information.

    This class handles all traceback parsing logic, separated from the
    main Integrator agent for better maintainability and testability.
    """

    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)

    def parse_test_output(self, test_file: str, output: str) -> List[TestFailureInfo]:
        """
        Parse failure information from a single test file's pytest output.

        Args:
            test_file: Relative path to the test file
            output: Raw pytest output string

        Returns:
            List of TestFailureInfo objects for each failure
        """
        failures = []

        # Match FAILED lines: tests/test_xxx.py::TestClass::test_method - ErrorType: message
        failed_pattern = re.compile(
            r"FAILED\s+([^\s:]+)::([^\s]+)\s*-?\s*(.*)", re.MULTILINE
        )
        failed_matches = failed_pattern.findall(output)

        for match in failed_matches:
            file_path, test_name, error_hint = match

            # Extract the error block for this specific test
            test_block = self._extract_test_block(output, test_name)
            if not test_block:
                test_block = output

            # Parse structured information
            call_stack = self.parse_structured_traceback(test_block)
            error_type, root_cause = self.extract_root_cause(test_block)
            assertion_details = self.extract_assertion_details(test_block)
            impl_files = self.get_impl_files_from_stack(call_stack)

            # Fallback: extract impl files directly from traceback text
            if not impl_files:
                impl_files = self.extract_impl_files_from_traceback(test_block)
            if not impl_files:
                impl_files = self.infer_impl_files_from_test(test_file)

            # If still no call_stack, create frames from extracted impl_files
            if not call_stack and impl_files:
                for impl_file in impl_files[:3]:
                    # Find line numbers from traceback
                    line_match = re.search(rf"{re.escape(impl_file)}:(\d+)", test_block)
                    line_num = int(line_match.group(1)) if line_match else 0
                    call_stack.append(
                        StackFrame(
                            file_path=impl_file,
                            line_num=line_num,
                            func_name="<extracted>",
                            code_line="",
                            is_project_file=True,
                        )
                    )

            failures.append(
                TestFailureInfo(
                    test_file=test_file,
                    test_name=test_name,
                    error_type=error_type,
                    error_message=error_hint,
                    root_cause=root_cause,
                    call_stack=call_stack,
                    assertion_details=assertion_details,
                    impl_files=impl_files if impl_files else [test_file],
                    raw_traceback=test_block[:3000],
                )
            )

        # Fallback if no FAILED lines matched
        if not failures and "FAILED" in output:
            call_stack = self.parse_structured_traceback(output)
            error_type, root_cause = self.extract_root_cause(output)
            impl_files = self.get_impl_files_from_stack(call_stack)
            failures.append(
                TestFailureInfo(
                    test_file=test_file,
                    test_name="unknown",
                    error_type=error_type,
                    error_message="Test failed",
                    root_cause=root_cause,
                    call_stack=call_stack,
                    impl_files=impl_files if impl_files else [test_file],
                    raw_traceback=output[:3000],
                )
            )

        return failures

    def _extract_test_block(self, output: str, test_name: str) -> str:
        """Extract the error block for a specific test from pytest output."""
        # Try to find block starting with _____ test_name _____
        pattern = re.compile(
            rf"_{3,}\s*{re.escape(test_name)}\s*_{3,}\s*(.*?)(?=_{3,}|\Z)", re.DOTALL
        )
        match = pattern.search(output)
        if match:
            return match.group(0)
        return ""

    def parse_structured_traceback(self, text: str) -> List[StackFrame]:
        """
        Extract structured call stack from pytest traceback.

        Parses lines like:
            path/to/file.py:123: in function_name
                code_line_here
        Or standard Python traceback:
            File "path/to/file.py", line 123, in function_name
                code_line_here
        """
        frames = []
        lines = text.split("\n")

        # Pattern 1: pytest format "file.py:123: in func_name" (with optional leading whitespace)
        pytest_pattern = re.compile(
            r"^\s*([^\s:]+\.py):(\d+):\s*in\s+(\w+)", re.IGNORECASE
        )
        # Pattern 2: standard Python traceback
        python_pattern = re.compile(
            r'File\s+"([^"]+\.py)",\s*line\s+(\d+),\s*in\s+(\w+)', re.IGNORECASE
        )
        # Pattern 3: pytest short format "    file.py:123: Error"
        short_pattern = re.compile(r"^\s+([^\s:]+\.py):(\d+):", re.IGNORECASE)

        i = 0
        while i < len(lines):
            line = lines[i]

            # Try pytest format first
            m = pytest_pattern.search(line)
            if not m:
                m = python_pattern.search(line)

            if m:
                file_path, line_num, func_name = m.groups()
                code_line = ""
                # Next non-empty line is usually the code
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith(
                        ("File ", "E ", ">", "_")
                    ):
                        code_line = next_line

                is_project = self.is_project_file(file_path)
                rel_path = self.normalize_path(file_path)

                frames.append(
                    StackFrame(
                        file_path=rel_path,
                        line_num=int(line_num),
                        func_name=func_name,
                        code_line=code_line,
                        is_project_file=is_project,
                    )
                )
            else:
                # Try short pattern (no function name)
                m_short = short_pattern.search(line)
                if m_short:
                    file_path, line_num = m_short.groups()
                    is_project = self.is_project_file(file_path)
                    rel_path = self.normalize_path(file_path)
                    frames.append(
                        StackFrame(
                            file_path=rel_path,
                            line_num=int(line_num),
                            func_name="<unknown>",
                            code_line="",
                            is_project_file=is_project,
                        )
                    )
            i += 1

        return frames

    def extract_root_cause(self, text: str) -> Tuple[str, str]:
        """
        Extract the root cause exception from traceback.

        Returns (error_type, error_message).
        The root cause is typically the LAST exception in the traceback.
        """
        # Pattern for "ExceptionType: message" lines
        exception_pattern = re.compile(
            r"^E?\s*(\w+(?:Error|Exception|Warning))\s*:\s*(.+)$", re.MULTILINE
        )
        matches = exception_pattern.findall(text)

        if matches:
            # Take the last match as root cause
            error_type, message = matches[-1]
            return error_type.strip(), message.strip()

        # Fallback: look for lines starting with "E " (pytest assertion output)
        e_lines = re.findall(r"^E\s+(.+)$", text, re.MULTILINE)
        if e_lines:
            # Find the most informative E line
            for line in reversed(e_lines):
                if ":" in line and not line.startswith("+"):
                    return "AssertionError", line.strip()
            return "AssertionError", e_lines[-1].strip()

        return "TestFailure", "Unknown error"

    def extract_assertion_details(self, text: str) -> str:
        """
        Extract assertion comparison details from pytest output.

        Looks for patterns like:
            assert x == y
            +  where x = ...
            +  and y = ...
        """
        details = []

        # Find "assert ..." line
        assert_match = re.search(r"^>\s*(assert\s+.+)$", text, re.MULTILINE)
        if assert_match:
            details.append(assert_match.group(1).strip())

        # Find "+  where ..." lines
        where_matches = re.findall(r"^\+\s+(where\s+.+|and\s+.+)$", text, re.MULTILINE)
        for m in where_matches[:3]:
            details.append(m.strip())

        # Find comparison lines like "E       assert 1 == 2"
        compare_match = re.search(r"^E\s+(assert\s+.+)$", text, re.MULTILINE)
        if compare_match and not details:
            details.append(compare_match.group(1).strip())

        return " | ".join(details) if details else ""

    def get_impl_files_from_stack(self, call_stack: List[StackFrame]) -> List[str]:
        """Get implementation files from call stack, ordered by proximity to error."""
        impl_files = []
        seen = set()
        # Reverse to get files closest to the error first
        for frame in reversed(call_stack):
            if frame.is_project_file and frame.file_path not in seen:
                if "test_" not in frame.file_path and "/tests/" not in frame.file_path:
                    seen.add(frame.file_path)
                    impl_files.append(frame.file_path)
        return impl_files

    def is_project_file(self, file_path: str) -> bool:
        """Check if a file path is within the project (not stdlib/site-packages/conda)."""
        # Common external library patterns
        external_patterns = [
            "site-packages",
            "lib/python",
            "anaconda",
            "miniconda",
            "conda/envs",
            "/envs/",
            "venv/",
            ".venv/",
            "virtualenv",
            "/usr/lib/",
            "/usr/local/",
            "/opt/",
            "dist-packages",
        ]

        path_lower = file_path.lower()
        for pattern in external_patterns:
            if pattern.lower() in path_lower:
                return False

        # Absolute paths must be within project_root
        if self.project_root and os.path.isabs(file_path):
            return file_path.startswith(self.project_root)

        # Relative paths are assumed to be project files
        return True

    def normalize_path(self, file_path: str) -> str:
        """Normalize file path to be relative to project root."""
        if self.project_root and os.path.isabs(file_path):
            if file_path.startswith(self.project_root):
                return os.path.relpath(file_path, self.project_root)
        if file_path.startswith("./"):
            return file_path[2:]
        return file_path

    def extract_impl_files_from_traceback(self, traceback: str) -> List[str]:
        """Extract implementation file paths from traceback text."""
        impl_files = []
        seen = set()

        # Match file paths
        pattern = re.compile(r"([^\s:\"']+\.py):(\d+):", re.MULTILINE)
        matches = pattern.findall(traceback)

        for file_path, line_num in matches:
            # Skip test files
            if "test_" in file_path or "/tests/" in file_path:
                continue
            # Skip external libraries
            if "site-packages" in file_path or "lib/python" in file_path:
                continue

            # Normalize path
            if self.project_root and os.path.isabs(file_path):
                if file_path.startswith(self.project_root):
                    rel_path = os.path.relpath(file_path, self.project_root)
                else:
                    continue  # Not in project
            else:
                rel_path = file_path

            # Clean up path
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]

            if rel_path and rel_path not in seen and rel_path.endswith(".py"):
                seen.add(rel_path)
                impl_files.append(rel_path)

        return impl_files

    def infer_impl_files_from_test(self, test_file: str) -> List[str]:
        """Infer implementation files from test file imports."""
        impl_files = []

        test_path = os.path.join(self.project_root, test_file)
        if not os.path.exists(test_path):
            return impl_files

        try:
            with open(test_path, "r", encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Convert module.path -> module/path.py
                        module_path = node.module.replace(".", "/") + ".py"
                        if os.path.exists(os.path.join(self.project_root, module_path)):
                            impl_files.append(module_path)
        except Exception:
            pass

        return impl_files

    def parse_pytest_failures(self, output: str) -> List[Dict]:
        """
        Parse pytest output to extract failing implementation files.

        Returns list of dicts with 'file_path' and 'error' keys.
        """
        failing_files = []
        seen_files = set()

        # Pattern to match file paths in traceback
        traceback_pattern = re.compile(
            r"^([^\s:]+\.py):(\d+):\s*in\s+\w+", re.MULTILINE
        )

        # Pattern to match error messages
        error_pattern = re.compile(r"^E\s+(.+)$", re.MULTILINE)

        # Find all file references in traceback
        matches = traceback_pattern.findall(output)
        errors = error_pattern.findall(output)

        # Get the primary error message
        primary_error = errors[0] if errors else "Test assertion failed"

        for file_path, line_num in matches:
            # Skip test files
            if "test_" in file_path or "/tests/" in file_path:
                continue

            # Only attribute failures to files within project_root
            if not (
                self.project_root
                and os.path.isabs(file_path)
                and file_path.startswith(self.project_root)
            ):
                continue

            # Normalize path relative to project root
            if self.project_root and file_path.startswith(self.project_root):
                rel_path = os.path.relpath(file_path, self.project_root)
            else:
                rel_path = file_path

            # Clean up path
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]

            # Extract just the project-relative path
            parts = rel_path.split("/")
            if "workspaces" in parts:
                idx = parts.index("workspaces")
                if idx + 2 < len(parts):
                    rel_path = "/".join(parts[idx + 3 :])

            if rel_path and rel_path not in seen_files and rel_path.endswith(".py"):
                seen_files.add(rel_path)
                failing_files.append(
                    {
                        "file_path": rel_path,
                        "error": f"Test failure at line {line_num}: {primary_error}",
                    }
                )

        return failing_files
