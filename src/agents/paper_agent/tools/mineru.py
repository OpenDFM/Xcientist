import os
import subprocess
from typing import Optional

from agents import function_tool

from src.agents.paper_agent.utils.config import get_mineru_cmd


def _which(cmd: str) -> str:
    try:
        import shutil

        return shutil.which(cmd) or ""
    except Exception:
        return ""


@function_tool
def mineru_parse_pdf(
    pdf_path: str, output_dir: str, mineru_cmd: str = "", timeout_sec: int = 600
) -> dict:
    """
    Parse a PDF with MinerU (external tool).

    Args:
        pdf_path: PDF file path.
        output_dir: Directory to write MinerU outputs.
        mineru_cmd: Optional command path/name. If empty, uses env MINERU_CMD or "mineru".
        timeout_sec: Subprocess timeout.
    """
    pdf_path = os.path.abspath(str(pdf_path or ""))
    output_dir = os.path.abspath(str(output_dir or ""))
    cmd = str(mineru_cmd or get_mineru_cmd()).strip()

    if not os.path.exists(pdf_path):
        return {"success": False, "error": f"pdf not found: {pdf_path}"}

    resolved = (
        _which(cmd)
        if cmd and ("/" not in cmd)
        else (cmd if os.path.exists(cmd) else "")
    )
    if not resolved:
        return {
            "success": False,
            "error": "mineru command not found. Install MinerU or set MINERU_CMD to the executable path.",
            "hint": "Example: export MINERU_CMD=/path/to/mineru",
        }

    os.makedirs(output_dir, exist_ok=True)

    # NOTE: MinerU CLI flags may differ across versions. We keep this minimal and let users override via wrapper if needed.
    argv = [resolved, pdf_path, output_dir]
    try:
        p = subprocess.run(
            argv, capture_output=True, text=True, timeout=max(1, int(timeout_sec))
        )
        return {
            "success": p.returncode == 0,
            "return_code": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "output_dir": output_dir,
            "cmd": argv,
            "message": f"mineru return_code={p.returncode}, output_dir={output_dir}",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "return_code": 124,
            "stdout": "",
            "stderr": f"timeout after {int(timeout_sec)}s",
            "cmd": argv,
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "cmd": argv}
