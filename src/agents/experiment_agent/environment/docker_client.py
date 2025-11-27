"""
Docker Client for connecting to existing TCP server in Docker container.

This client connects to an already running Docker container with a TCP server,
allowing command execution without managing the container lifecycle.
"""

import socket
import json
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class DockerClientConfig:
    """Configuration for Docker client."""

    host: str = "localhost"
    port: int = 8000
    timeout: int = 300  # 5 minutes default timeout
    buffer_size: int = 4096
    max_retries: int = 10
    retry_delay: float = 1.0


class DockerClient:
    """
    Client for connecting to TCP server in Docker container.

    This client does not manage container lifecycle - it assumes the container
    is already running with a TCP server listening on the specified port.

    Usage:
        client = DockerClient(host="localhost", port=8000)

        # Check connection
        if client.is_connected():
            result = client.run_command("python script.py")
            print(result['result'])
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout: int = 300,
        buffer_size: int = 4096,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Docker client.

        Args:
            host: Hostname or IP of the Docker host
            port: TCP server port in the container
            timeout: Command execution timeout in seconds
            buffer_size: Socket buffer size for receiving data
            max_retries: Maximum connection retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.buffer_size = buffer_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def is_connected(self) -> bool:
        """
        Check if the TCP server is reachable.

        Returns:
            True if connection succeeds, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  # Short timeout for connection check
                s.connect((self.host, self.port))
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False

    def run_command(
        self,
        command: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        retry: bool = True,
    ) -> Dict[str, any]:
        """
        Execute a command in the Docker container via TCP server.

        Args:
            command: Shell command to execute
            stream_callback: Optional callback for streaming output.
                           Called with each line of output as it arrives.
            retry: Whether to retry on connection failures

        Returns:
            Dictionary with:
                - status: Exit code (0 for success, -1 for error)
                - result: Command output or error message

        Example:
            def print_stream(line):
                print(f"[Stream] {line}", end='')

            result = client.run_command(
                "python train.py --epochs 10",
                stream_callback=print_stream
            )

            if result['status'] == 0:
                print("Success!")
            else:
                print(f"Error: {result['result']}")
        """
        attempts = self.max_retries if retry else 1
        last_error = None

        for attempt in range(attempts):
            try:
                return self._execute_command(command, stream_callback)
            except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as e:
                last_error = e
                if attempt < attempts - 1:
                    print(
                        f"Connection attempt {attempt + 1}/{attempts} failed, retrying..."
                    )
                    time.sleep(self.retry_delay)
                continue
            except socket.timeout:
                return {
                    "status": -1,
                    "result": f"Command execution timeout ({self.timeout}s). The command may be taking too long or the container is not responding.",
                }
            except Exception as e:
                return {
                    "status": -1,
                    "result": f"Unexpected error: {type(e).__name__}: {str(e)}",
                }

        # All retries failed
        return {
            "status": -1,
            "result": f"Failed to connect to Docker container at {self.host}:{self.port} after {attempts} attempts. Error: {last_error}\n\nPlease ensure:\n1. Docker container is running\n2. TCP server is started in the container\n3. Port {self.port} is correctly mapped/accessible",
        }

    def _execute_command(
        self,
        command: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, any]:
        """Internal method to execute command via socket."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout)
            s.connect((self.host, self.port))

            # Send command
            s.sendall(command.encode())

            # Receive response
            partial_line = ""
            while True:
                chunk = s.recv(self.buffer_size)

                if not chunk:
                    break

                try:
                    data = partial_line + chunk.decode("utf-8")
                except UnicodeDecodeError as e:
                    print(f"Unicode decode error: {e}, skipping chunk")
                    continue

                lines = data.split("\n")

                # Process all complete lines except the last one
                for line in lines[:-1]:
                    if line:
                        try:
                            response = json.loads(line)
                            if response["type"] == "chunk":
                                # Stream output
                                if stream_callback:
                                    stream_callback(response["data"])
                            elif response["type"] == "final":
                                # Final result
                                return {
                                    "status": response["status"],
                                    "result": response["result"],
                                }
                        except json.JSONDecodeError:
                            print(f"Invalid JSON: {line}")

                # Save possibly incomplete last line
                partial_line = lines[-1]

        # Connection closed without final response
        return {
            "status": -1,
            "result": "Connection closed without final response",
        }

    def run_python_script(
        self,
        script_path: str,
        args: Optional[Dict[str, any]] = None,
        cwd: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, any]:
        """
        Convenience method to run a Python script.

        Args:
            script_path: Path to Python script (relative to container working directory)
            args: Optional dictionary of command-line arguments
            cwd: Optional working directory for the command
            stream_callback: Optional callback for streaming output

        Returns:
            Dictionary with status and result

        Example:
            result = client.run_python_script(
                "train.py",
                args={"epochs": 10, "batch_size": 32},
                cwd="/workspace"
            )
        """
        # Build command
        cmd_parts = ["python", script_path]

        if args:
            for key, value in args.items():
                cmd_parts.append(f"--{key}")
                cmd_parts.append(str(value))

        command = " ".join(cmd_parts)

        if cwd:
            command = f"cd {cwd} && {command}"

        return self.run_command(command, stream_callback=stream_callback)

    def test_connection(self) -> Dict[str, any]:
        """
        Test the connection with a simple command.

        Returns:
            Dictionary with connection test results
        """
        if not self.is_connected():
            return {
                "success": False,
                "message": f"Cannot connect to {self.host}:{self.port}",
            }

        result = self.run_command("echo 'Connection test successful'", retry=False)

        if result["status"] == 0:
            return {
                "success": True,
                "message": "Connection test successful",
                "output": result["result"],
            }
        else:
            return {
                "success": False,
                "message": "Connection succeeded but command execution failed",
                "error": result["result"],
            }

    def get_working_directory(self) -> Optional[str]:
        """
        Get the current working directory in the container.

        Returns:
            Working directory path or None if failed
        """
        result = self.run_command("pwd", retry=False)
        if result["status"] == 0:
            return result["result"].strip()
        return None

    def list_directory(self, path: str = ".") -> Optional[list]:
        """
        List contents of a directory in the container.

        Args:
            path: Directory path to list

        Returns:
            List of file/directory names or None if failed
        """
        result = self.run_command(f"ls -1 {path}", retry=False)
        if result["status"] == 0:
            return [
                line.strip() for line in result["result"].split("\n") if line.strip()
            ]
        return None


def create_docker_client(
    host: str = "localhost",
    port: int = 8000,
    timeout: int = 300,
) -> DockerClient:
    """
    Factory function to create a DockerClient instance.

    Args:
        host: Docker host
        port: TCP server port
        timeout: Command timeout in seconds

    Returns:
        DockerClient instance
    """
    return DockerClient(host=host, port=port, timeout=timeout)


# Example usage
if __name__ == "__main__":
    import sys

    # Example 1: Basic usage
    print("=" * 60)
    print("Example 1: Basic Connection Test")
    print("=" * 60)

    client = create_docker_client(host="localhost", port=8000)

    # Test connection
    test_result = client.test_connection()
    if test_result["success"]:
        print(f"✓ {test_result['message']}")
        print(f"  Output: {test_result['output']}")
    else:
        print(f"✗ {test_result['message']}")
        sys.exit(1)

    # Example 2: Run simple command
    print("\n" + "=" * 60)
    print("Example 2: Run Simple Command")
    print("=" * 60)

    result = client.run_command("python --version")
    print(f"Status: {result['status']}")
    print(f"Output:\n{result['result']}")

    # Example 3: Stream output
    print("\n" + "=" * 60)
    print("Example 3: Stream Output")
    print("=" * 60)

    def print_stream(line):
        print(f"[Stream] {line}", end="")

    result = client.run_command(
        "python -c 'import time; [print(i) or time.sleep(0.1) for i in range(5)]'",
        stream_callback=print_stream,
    )
    print(f"\nFinal status: {result['status']}")

    # Example 4: Run Python script
    print("\n" + "=" * 60)
    print("Example 4: Run Python Script")
    print("=" * 60)

    # This assumes there's a script in the container
    # result = client.run_python_script(
    #     "train.py",
    #     args={"epochs": 10, "batch_size": 32},
    #     stream_callback=print_stream
    # )
