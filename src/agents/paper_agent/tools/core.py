import os
import subprocess
from typing import List

from agents import function_tool

from src.agents.paper_agent.utils.config import get_bash_timeout_seconds


DEFAULT_BASH_TIMEOUT_SECONDS = int(get_bash_timeout_seconds())


class SecurityContext:
    _project_root: str = ""
    _workspace_root: str = ""
    _read_roots: List[str] = []
    _write_roots: List[str] = []

    @classmethod
    def set_roots(cls, project_root: str, workspace_root: str = "") -> None:
        cls._project_root = os.path.abspath(str(project_root or ""))
        cls._workspace_root = os.path.abspath(str(workspace_root or project_root or ""))

    @classmethod
    def set_access(cls, read_roots: List[str], write_roots: List[str]) -> None:
        cls._read_roots = [
            os.path.abspath(str(p)) for p in (read_roots or []) if str(p).strip()
        ]
        cls._write_roots = [
            os.path.abspath(str(p)) for p in (write_roots or []) if str(p).strip()
        ]

    @classmethod
    def _is_within_any_root(cls, path: str, roots: List[str]) -> bool:
        try:
            ap = os.path.abspath(str(path or ""))
            for r in roots or []:
                rr = os.path.abspath(str(r or ""))
                if not rr:
                    continue
                try:
                    if os.path.commonpath([ap, rr]) == rr:
                        return True
                except Exception:
                    continue
            return False
        except Exception:
            return False

    @classmethod
    def get_project_root(cls) -> str:
        return cls._project_root

    @classmethod
    def get_workspace_root(cls) -> str:
        return cls._workspace_root


@function_tool
def bash(command: str, working_dir: str = "") -> dict:
    try:
        cwd = str(
            working_dir
            or SecurityContext.get_workspace_root()
            or SecurityContext.get_project_root()
            or os.getcwd()
        )
        p = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=DEFAULT_BASH_TIMEOUT_SECONDS,
        )
        return {
            "success": p.returncode == 0,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "return_code": p.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {DEFAULT_BASH_TIMEOUT_SECONDS} seconds",
            "return_code": -1,
        }
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "return_code": -1}


@function_tool
def file_viewer(file_path: str, start_line: int = 1, end_line: int = -1) -> dict:
    try:
        full_path = str(file_path or "")
        if not os.path.isabs(full_path):
            root = (
                SecurityContext.get_project_root()
                or SecurityContext.get_workspace_root()
                or os.getcwd()
            )
            full_path = os.path.join(root, full_path)
        if SecurityContext._read_roots and (
            not SecurityContext._is_within_any_root(
                full_path, SecurityContext._read_roots
            )
        ):
            return {
                "success": False,
                "error": f"read forbidden by SecurityContext: {full_path}",
            }

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)
        s = max(0, int(start_line) - 1)
        e = total if int(end_line) == -1 else min(total, int(end_line))
        numbered = []
        for i in range(s, e):
            numbered.append(f"{i + 1:4d}|{lines[i].rstrip()}")
        return {
            "success": True,
            "file_path": full_path,
            "total_lines": total,
            "showing": f"lines {s + 1}-{e}",
            "content": "\n".join(numbered),
        }
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {file_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def write_file(file_path: str, content: str) -> dict:
    try:
        full_path = str(file_path or "")
        if not os.path.isabs(full_path):
            root = SecurityContext.get_project_root() or os.getcwd()
            full_path = os.path.join(root, full_path)
        if SecurityContext._write_roots and (
            not SecurityContext._is_within_any_root(
                full_path, SecurityContext._write_roots
            )
        ):
            return {
                "success": False,
                "error": f"write forbidden by SecurityContext: {full_path}",
            }
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(str(content or ""))
        return {
            "success": True,
            "file_path": full_path,
            "message": f"Written {len(str(content or ''))} chars",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def edit_file(file_path: str, old_string: str, new_string: str) -> dict:
    try:
        full_path = str(file_path or "")
        if not os.path.isabs(full_path):
            root = SecurityContext.get_project_root() or os.getcwd()
            full_path = os.path.join(root, full_path)
        if SecurityContext._write_roots and (
            not SecurityContext._is_within_any_root(
                full_path, SecurityContext._write_roots
            )
        ):
            return {
                "success": False,
                "error": f"edit forbidden by SecurityContext: {full_path}",
            }
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        if str(old_string) not in content:
            return {"success": False, "error": "old_string not found in file"}
        new_content = content.replace(str(old_string), str(new_string), 1)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return {"success": True, "file_path": full_path, "message": "File edited"}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {file_path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_architect_tools() -> List:
    return [bash, file_viewer, write_file, edit_file]


def get_writer_tools() -> List:
    return [bash, file_viewer, write_file, edit_file]
