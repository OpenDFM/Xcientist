import os
import re
import subprocess
import time
from typing import Any, Dict

from agents import function_tool

from src.agents.paper_agent.utils.config import (
    get_openai_config,
    get_run_config_path,
    get_runtime_config,
)


def _ensure_subagent_md_dir(artifact_dir: str) -> str:
    artifact_dir = os.path.abspath(str(artifact_dir or ""))
    md_dir = os.path.join(artifact_dir, "index", "subagent_md")
    os.makedirs(md_dir, exist_ok=True)
    return md_dir


def _next_seq_for_kind(md_dir: str, run_name: str, kind: str) -> int:
    md_dir = os.path.abspath(str(md_dir or ""))
    run_name = str(run_name or "").strip() or "run"
    kind = str(kind or "").strip() or "subagent"
    max_seq = 0
    try:
        for fn in os.listdir(md_dir):
            if not fn.endswith(".md"):
                continue
            if not fn.startswith(f"{run_name}__{kind}__"):
                continue
            m = re.search(r"__([0-9]{3})\.md$", fn)
            if not m:
                continue
            try:
                max_seq = max(max_seq, int(m.group(1)))
            except Exception:
                continue
    except Exception:
        pass
    return max_seq + 1


def _alloc_output_md_path(kind: str, requested_output_path: str = "") -> str:
    cfg = get_runtime_config()
    run_name = str(getattr(cfg, "run_name", "") or "").strip() or "run"
    artifact_dir = str(getattr(cfg, "artifact_dir", "") or "")
    md_dir = _ensure_subagent_md_dir(artifact_dir=artifact_dir)

    kind = str(kind or "").strip().lower() or "subagent"
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    seq = _next_seq_for_kind(md_dir=md_dir, run_name=run_name, kind=kind)

    if str(requested_output_path or "").strip():
        base = os.path.basename(str(requested_output_path))
        base = base if base.endswith(".md") else (base + ".md")
        out = os.path.join(md_dir, base)
        if os.path.exists(out):
            root, ext = os.path.splitext(base)
            out = os.path.join(md_dir, f"{root}__{ts}__{seq:03d}{ext}")
        return os.path.abspath(out)

    name = f"{run_name}__{kind}__{ts}__{seq:03d}.md"
    return os.path.abspath(os.path.join(md_dir, name))


def _run_subagent(
    kind: str,
    request: str,
) -> Dict[str, Any]:
    if not get_run_config_path():
        return {
            "success": False,
            "error": "PAPER_AGENT_RUN_CONFIG_PATH is not set (required to locate run config).",
            "result": {},
        }
    try:
        cfg = get_runtime_config()
        cfg_models = getattr(cfg, "models", None) or {}
        fallback_model = str(getattr(cfg, "model", "") or "gpt-5.2")
        model = str(cfg_models.get(str(kind).strip().lower()) or fallback_model)
        provider = get_openai_config(model=model)
        api_key = str(provider.get("api_key", "") or "").strip()
        if not api_key:
            return {
                "success": False,
                "error": "API key is not set for this subagent model (check config.py/env defaults).",
                "result": {},
            }
    except Exception:
        pass
    output_path = _alloc_output_md_path(kind=str(kind), requested_output_path="")

    argv = [
        "python",
        "-m",
        "src.agents.paper_agent.tools.subagent_runner",
        "--kind",
        str(kind),
        "--output-path",
        os.path.abspath(str(output_path)),
    ]
    if request:
        argv += ["--request", str(request)]

    try:
        # Inherit stdout/stderr so the caller can see live subagent logs.
        p = subprocess.run(argv, timeout=1800)
        return {
            "success": p.returncode == 0,
            "return_code": p.returncode,
            "stdout": "",
            "stderr": "",
            # We already allocated the md output path deterministically.
            "result": {
                "kind": str(kind),
                "md_path": output_path,
                "output_path": output_path,
            },
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "return_code": 124,
            "stdout": "",
            "stderr": "timeout",
            "result": {},
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "result": {}}


@function_tool
def review_paper() -> dict:
    """
    Run the paper review agent as a tool and return the review report path.
    """
    res = _run_subagent(
        kind="review",
        request="",
    )
    out = res.get("result") or {}
    md_path = out.get("md_path", "") or out.get("output_path", "")
    return {
        "success": bool(res.get("success")),
        "review_md_path": md_path,
        "message": f"review report: {md_path}" if md_path else "review completed",
        "runner": res,
    }


@function_tool
def analyze_results(
    request: str,
) -> dict:
    """
    Run the analysis agent as a tool. The request will be injected into the analysis agent prompt.
    """
    res = _run_subagent(
        kind="analysis",
        request=request,
    )
    out = res.get("result") or {}
    md_path = out.get("md_path", "") or out.get("output_path", "")
    return {
        "success": bool(res.get("success")),
        "analysis_md_path": md_path,
        "message": f"analysis report: {md_path}" if md_path else "analysis completed",
        "runner": res,
    }


@function_tool
def visualize_request(
    request: str,
) -> dict:
    """
    Run the visualization agent as a tool. The request will be injected into the viz agent prompt.
    """
    res = _run_subagent(
        kind="viz",
        request=request,
    )
    out = res.get("result") or {}
    md_path = out.get("md_path", "") or out.get("output_path", "")
    return {
        "success": bool(res.get("success")),
        "viz_md_path": md_path,
        "message": f"viz report: {md_path}" if md_path else "viz completed",
        "runner": res,
    }


@function_tool
def research_literature(request: str) -> dict:
    """
    Run the literature research subagent as a tool.
    The request should specify what to search and any desired output conventions.
    """
    res = _run_subagent(
        kind="literature",
        request=request,
    )
    out = res.get("result") or {}
    md_path = out.get("md_path", "") or out.get("output_path", "")
    err = str(res.get("error") or "") or str(
        (res.get("result") or {}).get("error") or ""
    )
    return {
        "success": bool(res.get("success")),
        "literature_md_path": md_path,
        "message": (
            f"literature report: {md_path}" if md_path else "literature completed"
        ),
        "error": err if (not bool(res.get("success"))) else "",
        "runner": res,
    }
