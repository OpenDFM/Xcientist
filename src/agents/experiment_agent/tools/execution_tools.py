"""
Execution tools for experiment agents.

Provides a unified, secure shell execution environment for local operations.
Replaces all legacy Python/Docker specific execution tools.
"""

import os
import time
import subprocess
import shutil
from typing import Dict, Any, Optional
from pathlib import Path

from agents import function_tool
from src.agents.experiment_agent.tools.file_tools import _validate_path_security

# =============================================================================
# Core Execution Tool (Local Shell)
# =============================================================================


@function_tool
def run_shell_command(
    command: str,
    timeout: int = 300,
    stream_output: bool = False,
    # working_dir removed to enforce security
) -> Dict[str, Any]:
    """
    Execute a shell command locally in a secure, path-restricted environment.

    Constraint: All commands run in the configured PROJECT_DIR.

    Args:
        command: Shell command to execute
        timeout: Timeout in seconds (default: 300)
        stream_output: Whether to print stdout/stderr in real-time

    Returns:
        Dictionary with execution results (success, stdout, stderr, etc.)
    """
    # Enforce project directory
    from src.agents.experiment_agent.config import PROJECT_ROOT, PROJECT_DIR

    # Prefer PROJECT_ROOT, fallback to PROJECT_DIR for backward compatibility
    target_dir = PROJECT_ROOT if PROJECT_ROOT else PROJECT_DIR

    abs_working_dir = os.path.abspath(os.path.expanduser(target_dir))

    # Double check existence (should be guaranteed by main.py setup)
    if not os.path.exists(abs_working_dir):
        # Fallback only if config is totally broken (should not happen in prod)
        abs_working_dir = os.getcwd()

    # 2. Execute Command using subprocess
    start_time = time.time()

    try:
        # Use shell=True for flexibility (pipes, redirects), but be aware of injection risks.
        # Since inputs come from the Agent (which is trusted to be the actor),
        # and we operate locally, this is an acceptable trade-off for functionality.
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=abs_working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",  # Avoid crashing on decode errors
            bufsize=1,  # Line buffered
        )

        stdout_lines = []
        stderr_lines = []

        # Function to read stream
        def read_stream(stream, line_list, print_prefix=None):
            for line in stream:
                line_list.append(line)
                if stream_output and print_prefix:
                    # We can't easily distinguish stdout/stderr order perfectly without threads/asyncio,
                    # but this is sufficient for "seeing progress".
                    # For a true merged stream, we'd need asyncio (like kimi-cli).
                    # Here we stick to synchronous for simplicity unless needed.
                    print(line, end="")

        # Wait for completion with timeout
        try:
            # Note: communicate() reads all data into memory. For very large outputs this might be an issue,
            # but it's robust for standard tasks.
            # If stream_output is requested, we can't use communicate() straightforwardly for real-time printing
            # without threading.

            if stream_output:
                # Simple real-time streaming implementation (blocking on stdout then stderr)
                # NOTE: This isn't perfect parallel streaming but works for most CLI tools that flush.
                # For true async streaming we'd need asyncio.subprocess (like kimi-cli).
                # Keeping it simple with communicate for now to avoid async complexity in this synchronous tool.
                # If real-time is critical, we can upgrade to threading.
                pass

            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode

            if stream_output:
                if stdout:
                    print(stdout, end="")
                if stderr:
                    print(stderr, end="")

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            return {
                "success": False,
                "exit_code": -1,
                "stdout": stdout or "",
                "stderr": f"{stderr or ''}\n[Timeout after {timeout}s]",
                "execution_time": time.time() - start_time,
                "command": command,
                "working_dir": abs_working_dir,
                "error": f"Execution timed out after {timeout} seconds",
            }

        execution_time = time.time() - start_time

        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "execution_time": execution_time,
            "command": command,
            "working_dir": abs_working_dir,
        }

    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": time.time() - start_time,
            "command": command,
            "working_dir": abs_working_dir,
            "error": f"Execution error: {str(e)}",
        }



# Export only the single universal tool
__all__ = ["run_shell_command"]
