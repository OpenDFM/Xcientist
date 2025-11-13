"""
Code execution tools for experiment agents.

ALL CODE EXECUTION IS DONE IN DOCKER CONTAINER - LOCAL EXECUTION IS DISABLED.

Provides tools for running Python scripts, shell commands, and managing processes
in a Docker container via TCP server connection.

Compatible with openai-agents SDK.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from agents import function_tool

# =============================================================================
# Docker Client Management
# =============================================================================

# Global docker client instance (initialized when needed)
_docker_client = None


def get_docker_client():
    """Get or create the global Docker client instance."""
    global _docker_client
    if _docker_client is None:
        from src.agents.experiment_agent.environment import create_docker_client

        # Default configuration from environment variables
        _docker_client = create_docker_client(
            host=os.getenv("DOCKER_HOST", "localhost"),
            port=int(os.getenv("DOCKER_PORT", "8000")),
            timeout=int(os.getenv("DOCKER_TIMEOUT", "3600")),
        )
    return _docker_client


def set_docker_client(client):
    """
    Set the global Docker client instance.

    Args:
        client: DockerClient instance to use for execution
    """
    global _docker_client
    _docker_client = client


# =============================================================================
# Core Execution Tools (Docker-based)
# =============================================================================


@function_tool
def run_python_script(
    script_path: str,
    args: Optional[str] = None,
    working_dir: Optional[str] = None,
    timeout: int = 3600,
    stream_output: bool = False,
) -> Dict[str, Any]:
    """
    Run a Python script in Docker container.

    ⚠️ SECURITY: This tool ONLY executes code in Docker - local execution is disabled.

    Args:
        script_path: Path to the Python script (in container filesystem)
        args: Command line arguments as a string
        working_dir: Working directory for execution (in container)
        timeout: Timeout in seconds (default: 3600)
        stream_output: Whether to print output during execution

    Returns:
        Dictionary with execution results

    Environment variables:
        DOCKER_HOST: Docker host address (default: localhost)
        DOCKER_PORT: TCP server port (default: 8000)
        DOCKER_TIMEOUT: Command timeout in seconds (default: 3600)

    Example:
        result = run_python_script(
            "/workspace/train.py",
            args="--epochs 10 --batch_size 32",
            working_dir="/workspace"
        )
    """
    try:
        client = get_docker_client()

        # Check connection
        if not client.is_connected():
            return {
                "success": False,
                "error": f"Cannot connect to Docker container at {client.host}:{client.port}. "
                "Code execution requires Docker environment.",
                "exit_code": -1,
            }

        # Build command
        cmd = f"python {script_path}"
        if args:
            cmd += f" {args}"

        if working_dir:
            cmd = f"cd {working_dir} && {cmd}"

        # Stream callback
        output_lines = []

        def stream_callback(line):
            output_lines.append(line)
            if stream_output:
                print(line, end="")

        # Execute in Docker
        result = client.run_command(cmd, stream_callback=stream_callback)

        return {
            "success": result["status"] == 0,
            "exit_code": result["status"],
            "stdout": result["result"],
            "stderr": result["result"] if result["status"] != 0 else "",
            "execution_time": 0,  # TCP protocol doesn't track this
            "command": cmd,
            "working_dir": working_dir or "(container default)",
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Docker execution error: {str(e)}",
            "exit_code": -1,
        }


@function_tool
def run_shell_command(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 300,
    stream_output: bool = False,
) -> Dict[str, Any]:
    """
    Run a shell command in Docker container.

    ⚠️ SECURITY: This tool ONLY executes code in Docker - local execution is disabled.

    Args:
        command: Shell command to execute
        working_dir: Working directory for execution (in container)
        timeout: Timeout in seconds (default: 300)
        stream_output: Whether to print output during execution

    Returns:
        Dictionary with execution results

    Environment variables:
        DOCKER_HOST: Docker host address (default: localhost)
        DOCKER_PORT: TCP server port (default: 8000)

    Example:
        result = run_shell_command(
            "ls -la",
            working_dir="/workspace"
        )
    """
    try:
        client = get_docker_client()

        # Check connection
        if not client.is_connected():
            return {
                "success": False,
                "error": f"Cannot connect to Docker container at {client.host}:{client.port}. "
                "Code execution requires Docker environment.",
                "exit_code": -1,
            }

        # Prepare command
        if working_dir:
            command = f"cd {working_dir} && {command}"

        # Stream callback
        output_lines = []

        def stream_callback(line):
            output_lines.append(line)
            if stream_output:
                print(line, end="")

        # Execute in Docker
        result = client.run_command(command, stream_callback=stream_callback)

        return {
            "success": result["status"] == 0,
            "exit_code": result["status"],
            "stdout": result["result"],
            "stderr": result["result"] if result["status"] != 0 else "",
            "execution_time": 0,  # TCP protocol doesn't track this
            "command": command,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Docker execution error: {str(e)}",
            "exit_code": -1,
        }


@function_tool
def run_python_code(
    code: str,
    timeout: int = 60,
    stream_output: bool = False,
) -> Dict[str, Any]:
    """
    Run Python code snippet in Docker container.

    ⚠️ SECURITY: This tool ONLY executes code in Docker - local execution is disabled.

    Args:
        code: Python code to execute
        timeout: Timeout in seconds (default: 60)
        stream_output: Whether to print output during execution

    Returns:
        Dictionary with execution results

    Environment variables:
        DOCKER_HOST: Docker host address (default: localhost)
        DOCKER_PORT: TCP server port (default: 8000)

    Example:
        result = run_python_code(
            "print('Hello, World!')"
        )
    """
    try:
        client = get_docker_client()

        # Check connection
        if not client.is_connected():
            return {
                "success": False,
                "error": f"Cannot connect to Docker container at {client.host}:{client.port}. "
                "Code execution requires Docker environment.",
                "exit_code": -1,
            }

        # Escape single quotes in code and wrap in double quotes
        escaped_code = code.replace('"', '\\"').replace("$", "\\$")

        # Execute using python -c with double quotes
        command = f'python -c "{escaped_code}"'

        # Stream callback
        output_lines = []

        def stream_callback(line):
            output_lines.append(line)
            if stream_output:
                print(line, end="")

        # Execute in Docker
        result = client.run_command(command, stream_callback=stream_callback)

        return {
            "success": result["status"] == 0,
            "exit_code": result["status"],
            "stdout": result["result"],
            "stderr": result["result"] if result["status"] != 0 else "",
            "execution_time": 0,  # TCP protocol doesn't track this
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Docker execution error: {str(e)}",
            "exit_code": -1,
        }


# =============================================================================
# Package Management Tools (Docker-based)
# =============================================================================


@function_tool
def install_package(
    package_name: str,
    use_pip: bool = True,
    upgrade: bool = False,
    stream_output: bool = False,
) -> Dict[str, Any]:
    """
    Install a Python package using pip or conda in Docker container.

    ⚠️ SECURITY: This tool ONLY executes in Docker - local installation is disabled.

    Args:
        package_name: Name of the package to install
        use_pip: Use pip (True) or conda (False)
        upgrade: Whether to upgrade if already installed
        stream_output: Whether to print output during installation

    Returns:
        Dictionary with installation results

    Environment variables:
        DOCKER_HOST: Docker host address (default: localhost)
        DOCKER_PORT: TCP server port (default: 8000)

    Example:
        result = install_package("numpy", upgrade=True)
    """
    try:
        client = get_docker_client()

        # Check connection
        if not client.is_connected():
            return {
                "success": False,
                "error": f"Cannot connect to Docker container at {client.host}:{client.port}. "
                "Package installation requires Docker environment.",
            }

        # Build install command
        if use_pip:
            cmd = "python -m pip install"
            if upgrade:
                cmd += " --upgrade"
            cmd += f" {package_name}"
        else:
            cmd = f"conda install -y {package_name}"

        # Stream callback
        output_lines = []

        def stream_callback(line):
            output_lines.append(line)
            if stream_output:
                print(line, end="")

        # Execute in Docker
        result = client.run_command(cmd, stream_callback=stream_callback)

        return {
            "success": result["status"] == 0,
            "package": package_name,
            "stdout": result["result"],
            "stderr": result["result"] if result["status"] != 0 else "",
            "installed": result["status"] == 0,
            "command": cmd,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Docker execution error: {str(e)}",
        }


@function_tool
def list_installed_packages() -> Dict[str, Any]:
    """
    List all installed Python packages in Docker container.

    ⚠️ SECURITY: This tool queries the Docker environment - local packages are not listed.

    Returns:
        Dictionary with package list

    Environment variables:
        DOCKER_HOST: Docker host address (default: localhost)
        DOCKER_PORT: TCP server port (default: 8000)

    Example:
        result = list_installed_packages()
        print(f"Found {result['total_count']} packages")
    """
    try:
        client = get_docker_client()

        # Check connection
        if not client.is_connected():
            return {
                "success": False,
                "error": f"Cannot connect to Docker container at {client.host}:{client.port}.",
            }

        # List packages in JSON format
        result = client.run_command("python -m pip list --format=json", retry=False)

        if result["status"] == 0:
            import json

            try:
                packages = json.loads(result["result"])
                return {
                    "success": True,
                    "packages": packages,
                    "total_count": len(packages),
                }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Failed to parse package list JSON",
                    "raw_output": result["result"],
                }
        else:
            return {
                "success": False,
                "error": result["result"],
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error listing packages: {str(e)}",
        }


# =============================================================================
# Validation and Environment Tools
# =============================================================================


@function_tool
def check_python_syntax(
    file_path: str,
) -> Dict[str, Any]:
    """
    Check Python file syntax in Docker container without executing it.

    ⚠️ SECURITY: This tool checks syntax in Docker - local execution is disabled.

    Args:
        file_path: Path to Python file to check (in container filesystem)

    Returns:
        Dictionary with syntax check results

    Environment variables:
        DOCKER_HOST: Docker host address (default: localhost)
        DOCKER_PORT: TCP server port (default: 8000)

    Example:
        result = check_python_syntax("/workspace/script.py")
        if result['valid_syntax']:
            print("✓ Syntax is valid")
    """
    try:
        client = get_docker_client()

        # Check connection
        if not client.is_connected():
            return {
                "success": False,
                "error": f"Cannot connect to Docker container at {client.host}:{client.port}. "
                "Syntax check requires Docker environment.",
            }

        # Use python -m py_compile to check syntax
        cmd = f"python -m py_compile {file_path}"

        # Execute in Docker
        result = client.run_command(cmd, retry=False)

        if result["status"] == 0:
            return {
                "success": True,
                "valid_syntax": True,
                "file_path": file_path,
            }
        else:
            # Parse error message
            return {
                "success": True,
                "valid_syntax": False,
                "file_path": file_path,
                "syntax_error": {
                    "message": result["result"],
                },
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Docker execution error: {str(e)}",
        }


@function_tool
def get_environment_info() -> Dict[str, Any]:
    """
    Get information about the Python environment in Docker container.

    ⚠️ SECURITY: This tool queries the Docker environment - local environment is not accessible.

    Returns:
        Dictionary with environment information

    Environment variables:
        DOCKER_HOST: Docker host address (default: localhost)
        DOCKER_PORT: TCP server port (default: 8000)

    Example:
        info = get_environment_info()
        print(f"Python version: {info['python_version']}")
    """
    try:
        client = get_docker_client()

        # Check connection
        if not client.is_connected():
            return {
                "success": False,
                "error": f"Cannot connect to Docker container at {client.host}:{client.port}.",
            }

        # Get Python version
        python_version_result = client.run_command("python --version", retry=False)

        # Get platform info
        platform_result = client.run_command(
            "python -c 'import platform; print(platform.platform())'", retry=False
        )

        # Get system info
        system_result = client.run_command(
            "python -c 'import platform; print(platform.system())'", retry=False
        )

        # Get working directory
        pwd_result = client.run_command("pwd", retry=False)

        info = {
            "success": True,
            "python_version": (
                python_version_result.get("result", "").strip()
                if python_version_result["status"] == 0
                else "Unknown"
            ),
            "platform": (
                platform_result.get("result", "").strip()
                if platform_result["status"] == 0
                else "Unknown"
            ),
            "platform_system": (
                system_result.get("result", "").strip()
                if system_result["status"] == 0
                else "Unknown"
            ),
            "working_directory": (
                pwd_result.get("result", "").strip()
                if pwd_result["status"] == 0
                else "Unknown"
            ),
            "docker_host": client.host,
            "docker_port": client.port,
            "execution_mode": "Docker Container",
        }

        return info

    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting environment info: {str(e)}",
        }


# =============================================================================
# Logging Tools (Local filesystem - safe)
# =============================================================================


@function_tool
def create_log_file(
    log_dir: str,
    prefix: str = "execution",
) -> Dict[str, Any]:
    """
    Create a timestamped log file.

    Note: This creates files on the LOCAL filesystem, not in Docker.

    Args:
        log_dir: Directory to create log file in
        prefix: Prefix for log filename

    Returns:
        Dictionary with log file path

    Example:
        result = create_log_file("./logs", "experiment")
        log_path = result['log_path']
    """
    try:
        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{prefix}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)

        # Create empty file
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Log file created at {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n")

        return {
            "success": True,
            "log_path": log_path,
            "log_dir": log_dir,
            "filename": log_filename,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error creating log file: {str(e)}",
        }


@function_tool
def append_to_log(
    log_path: str,
    content: str,
) -> Dict[str, Any]:
    """
    Append content to a log file.

    Note: This writes to LOCAL filesystem, not in Docker.

    Args:
        log_path: Path to the log file
        content: Content to append

    Returns:
        Dictionary with success status

    Example:
        result = append_to_log("/path/to/log.txt", "Log message\\n")
    """
    try:
        log_path = os.path.expanduser(log_path)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")

        return {
            "success": True,
            "log_path": log_path,
            "bytes_written": len(content),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error appending to log: {str(e)}",
        }


# =============================================================================
# High-level Docker Execution Tools
# =============================================================================


@function_tool
def run_in_docker(
    command: str,
    stream_output: bool = False,
) -> Dict[str, Any]:
    """
    Execute a command in Docker container via TCP server.

    This connects to an existing Docker container with a TCP server and
    executes the command remotely.

    Args:
        command: Shell command to execute
        stream_output: If True, output is printed during execution

    Returns:
        Dictionary with execution results including status and output

    Environment variables:
        DOCKER_HOST: Docker host address (default: localhost)
        DOCKER_PORT: TCP server port (default: 8000)
        DOCKER_TIMEOUT: Command timeout in seconds (default: 3600)

    Example:
        result = run_in_docker("ls -la /workspace")
        print(result['stdout'])
    """
    try:
        client = get_docker_client()

        # Check connection
        if not client.is_connected():
            return {
                "success": False,
                "status": -1,
                "stdout": "",
                "stderr": f"Cannot connect to Docker container at {client.host}:{client.port}",
                "error": "Connection failed",
            }

        # Stream callback if needed
        output_lines = []

        def stream_callback(line):
            output_lines.append(line)
            if stream_output:
                print(line, end="")

        # Execute command
        result = client.run_command(command, stream_callback=stream_callback)

        return {
            "success": result["status"] == 0,
            "status": result["status"],
            "stdout": result["result"],
            "stderr": result["result"] if result["status"] != 0 else "",
            "execution_time": 0,  # TCP protocol doesn't track this
        }

    except Exception as e:
        return {
            "success": False,
            "status": -1,
            "stdout": "",
            "stderr": str(e),
            "error": f"Docker execution error: {str(e)}",
        }


@function_tool
def run_python_in_docker(
    script_path: str,
    args: Optional[str] = None,
    working_dir: Optional[str] = None,
    stream_output: bool = False,
) -> Dict[str, Any]:
    """
    Run a Python script in Docker container.

    Args:
        script_path: Path to Python script (in container filesystem)
        args: Command-line arguments as a string (e.g., "--epochs 10 --batch_size 32")
        working_dir: Working directory in container
        stream_output: If True, output is printed during execution

    Returns:
        Dictionary with execution results

    Example:
        result = run_python_in_docker(
            "train.py",
            args="--epochs 10 --batch_size 32",
            working_dir="/workspace"
        )
    """
    try:
        client = get_docker_client()

        # Check connection
        if not client.is_connected():
            return {
                "success": False,
                "status": -1,
                "stdout": "",
                "stderr": f"Cannot connect to Docker container at {client.host}:{client.port}",
                "error": "Connection failed",
            }

        # Stream callback if needed
        output_lines = []

        def stream_callback(line):
            output_lines.append(line)
            if stream_output:
                print(line, end="")

        # Execute Python script - convert args string format for client
        # The client expects a dict, so we need to handle string args differently
        result = client.run_python_script(
            script_path=script_path,
            args=args,  # Pass as string, let client handle it
            cwd=working_dir,
            stream_callback=stream_callback,
        )

        return {
            "success": result["status"] == 0,
            "status": result["status"],
            "stdout": result["result"],
            "stderr": result["result"] if result["status"] != 0 else "",
            "script_path": script_path,
        }

    except Exception as e:
        return {
            "success": False,
            "status": -1,
            "stdout": "",
            "stderr": str(e),
            "error": f"Docker Python execution error: {str(e)}",
        }


@function_tool
def test_docker_connection() -> Dict[str, Any]:
    """
    Test connection to Docker container's TCP server.

    Returns:
        Dictionary with connection test results

    Example:
        result = test_docker_connection()
        if result['success']:
            print("✓ Docker connection OK")
        else:
            print(f"✗ Connection failed: {result['message']}")
    """
    try:
        client = get_docker_client()
        result = client.test_connection()

        return {
            "success": result["success"],
            "message": result["message"],
            "host": client.host,
            "port": client.port,
            "output": result.get("output", ""),
            "error": result.get("error", ""),
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Connection test failed: {str(e)}",
            "error": str(e),
        }


# =============================================================================
# Local Execution Tools (No Docker Required)
# =============================================================================

import subprocess
import signal


@function_tool
def run_pytest_local(
    test_path: str,
    working_dir: Optional[str] = None,
    timeout: int = 300,
    extra_args: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run pytest tests locally (no Docker required).

    Args:
        test_path: Path to test file or directory (can be relative or absolute).
                   If relative, will be resolved relative to working_dir.
        working_dir: Working directory for pytest execution. This should be the project root
                    (where Python modules are importable). If None, automatically determines
                    the project root from test_path.
        timeout: Maximum execution time in seconds (default: 300)
        extra_args: Additional pytest arguments (e.g., "-v -s")

    Returns:
        Dictionary with execution results:
        - success: True if all tests passed
        - exit_code: pytest exit code (0=passed, 1=failed, 2=interrupted, 3=error, 4=usage error, 5=no tests)
        - stdout: Test output
        - stderr: Error output
        - execution_time: Time taken in seconds

    Example:
        result = run_pytest_local(
            "tests/test_module.py",
            working_dir="/path/to/project",  # Project root for imports
            timeout=60,
            extra_args="-v --tb=short"
        )
    """
    import time

    start_time = time.time()

    try:
        # Resolve working directory (project root)
        if working_dir:
            # Use provided working_dir (should be absolute path to project root)
            cwd = os.path.abspath(working_dir)
        else:
            # Auto-detect project root from test_path
            abs_test_path = os.path.abspath(test_path)

            if os.path.isfile(abs_test_path):
                # Test file - use its directory as starting point
                search_dir = os.path.dirname(abs_test_path)
            else:
                # Test directory
                search_dir = abs_test_path

            # If test_path is or contains "tests/", go up to find project root
            if search_dir.endswith("/tests") or search_dir.endswith("\\tests"):
                # tests/ directory itself - parent is project root
                cwd = os.path.dirname(search_dir)
            elif "/tests/" in search_dir or "\\tests\\" in search_dir:
                # Inside tests/ subdirectory - find project root
                parts = search_dir.replace("\\", "/").split("/tests/")
                cwd = parts[0]
            else:
                # Not in tests/ - assume current dir is project root
                cwd = search_dir

        # Convert test_path to be relative to cwd if it's relative
        if not os.path.isabs(test_path):
            # test_path is relative - make it relative to cwd
            test_path_for_pytest = test_path
        else:
            # test_path is absolute - try to make it relative to cwd for prettier output
            try:
                test_path_for_pytest = os.path.relpath(test_path, cwd)
            except ValueError:
                # Can't make relative (different drives on Windows) - use absolute
                test_path_for_pytest = test_path

        # Build pytest command
        cmd = ["pytest", test_path_for_pytest]
        if extra_args:
            cmd.extend(extra_args.split())

        # Run pytest with timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding="utf-8",
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            # Kill the process if timeout
            process.kill()
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            return {
                "success": False,
                "exit_code": -1,
                "stdout": stdout,
                "stderr": f"Test execution timeout after {timeout}s\n{stderr}",
                "execution_time": execution_time,
                "error": f"Timeout after {timeout}s",
            }

        execution_time = time.time() - start_time

        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": execution_time,
            "command": " ".join(cmd),
            "working_dir": cwd,
        }

    except FileNotFoundError:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "pytest not found. Please install pytest: pip install pytest",
            "execution_time": time.time() - start_time,
            "error": "pytest not installed",
        }
    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": time.time() - start_time,
            "error": f"Execution error: {str(e)}",
        }


@function_tool
def run_python_script_local(
    script_path: str,
    args: Optional[str] = None,
    working_dir: Optional[str] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    Run a Python script locally (no Docker required).

    Args:
        script_path: Path to the Python script
        args: Command line arguments as a string
        working_dir: Working directory for execution
        timeout: Maximum execution time in seconds (default: 300)

    Returns:
        Dictionary with execution results:
        - success: True if exit code is 0
        - exit_code: Process exit code
        - stdout: Standard output
        - stderr: Standard error
        - execution_time: Time taken in seconds

    Example:
        result = run_python_script_local(
            "train.py",
            args="--epochs 5 --lr 0.001",
            working_dir="/path/to/project",
            timeout=60
        )
    """
    import time

    start_time = time.time()

    try:
        # Build command
        cmd = ["python", script_path]
        if args:
            cmd.extend(args.split())

        # Set working directory
        cwd = (
            working_dir
            if working_dir
            else os.path.dirname(os.path.abspath(script_path))
        )

        # Run script with timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            text=True,
            encoding="utf-8",
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            # Kill the process if timeout
            process.kill()
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            return {
                "success": False,
                "exit_code": -1,
                "stdout": stdout,
                "stderr": f"Script execution timeout after {timeout}s\n{stderr}",
                "execution_time": execution_time,
                "error": f"Timeout after {timeout}s",
            }

        execution_time = time.time() - start_time

        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": execution_time,
            "command": " ".join(cmd),
            "working_dir": cwd,
        }

    except FileNotFoundError as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"File not found: {str(e)}",
            "execution_time": time.time() - start_time,
            "error": f"File not found: {str(e)}",
        }
    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": time.time() - start_time,
            "error": f"Execution error: {str(e)}",
        }


@function_tool
def run_python_code_local(
    code: str,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Run Python code snippet locally (no Docker required).

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 60)

    Returns:
        Dictionary with execution results:
        - success: True if exit code is 0
        - exit_code: Process exit code
        - stdout: Standard output
        - stderr: Standard error
        - execution_time: Time taken in seconds

    Example:
        result = run_python_code_local(
            "import sys; print(sys.version)"
        )
    """
    import time

    start_time = time.time()

    try:
        # Run code using python -c
        process = subprocess.Popen(
            ["python", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            # Kill the process if timeout
            process.kill()
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            return {
                "success": False,
                "exit_code": -1,
                "stdout": stdout,
                "stderr": f"Code execution timeout after {timeout}s\n{stderr}",
                "execution_time": execution_time,
                "error": f"Timeout after {timeout}s",
            }

        execution_time = time.time() - start_time

        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": execution_time,
        }

    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": time.time() - start_time,
            "error": f"Execution error: {str(e)}",
        }
