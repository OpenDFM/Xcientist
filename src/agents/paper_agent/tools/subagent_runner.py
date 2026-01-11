import argparse

import asyncio
import os
import sys
from typing import Any, Dict


_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
if _PROJECT_ROOT and (_PROJECT_ROOT not in sys.path):
    sys.path.insert(0, _PROJECT_ROOT)

from src.agents.paper_agent.tools.core import SecurityContext
from src.agents.paper_agent.subagents.analysis import PaperAnalysisAgent
from src.agents.paper_agent.subagents.literature import PaperLiteratureAgent
from src.agents.paper_agent.subagents.reviewer import PaperReviewerAgent
from src.agents.paper_agent.subagents.viz import PaperVizAgent


def _abs(p: str) -> str:
    return os.path.abspath(os.path.expanduser(str(p or "")))


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        # User preference: avoid JSON outputs. Keep this helper for backward compatibility,
        # but write a minimal Markdown representation instead.
        f.write(str(obj or ""))
        if not str(obj or "").endswith("\n"):
            f.write("\n")


def _write_md(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text or ""))
        if not str(text or "").endswith("\n"):
            f.write("\n")


def get_args():
    parser = argparse.ArgumentParser(
        description="paper_agent subagent runner (internal)"
    )
    parser.add_argument("--kind", required=True, help="review|analysis|viz|literature")
    parser.add_argument(
        "--model",
        default="",
        help="Override model name (optional; default from run config)",
    )
    parser.add_argument(
        "--paper-dir",
        default="",
        help="Override LaTeX project directory (optional; default from run config)",
    )
    parser.add_argument(
        "--project-dir",
        default="",
        help="Override project directory (optional; default from run config)",
    )
    parser.add_argument(
        "--artifact-dir",
        default="",
        help="Override artifacts directory (optional; default from run config)",
    )
    parser.add_argument(
        "--request", default="", help="Instruction/request for analysis/viz/literature"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to write the subagent report (Markdown)",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logs")
    return parser.parse_args()


async def _run(args) -> Dict[str, Any]:
    kind = str(args.kind or "").strip().lower()
    request = str(args.request or "")
    verbose = not bool(args.quiet)

    from src.agents.paper_agent.utils.config import get_runtime_config

    cfg = get_runtime_config()
    cfg_models = getattr(cfg, "models", None) or {}
    fallback_model = str(cfg.model or "gpt-5.2")
    model = str(args.model or cfg_models.get(kind) or fallback_model)
    paper_dir = _abs(args.paper_dir) if args.paper_dir else str(cfg.paper_dir or "")
    project_dir = (
        _abs(args.project_dir) if args.project_dir else str(cfg.project_dir or "")
    )
    artifact_dir = (
        _abs(args.artifact_dir) if args.artifact_dir else str(cfg.artifact_dir or "")
    )
    specs_dir = str(getattr(cfg, "specs_dir", "") or "")
    out_path = _abs(args.output_path)

    SecurityContext.set_roots(
        project_root=artifact_dir, workspace_root=os.path.dirname(artifact_dir)
    )

    index_dir = os.path.join(artifact_dir, "index")
    os.makedirs(index_dir, exist_ok=True)

    if kind == "review":
        agent = PaperReviewerAgent(model=model, max_turns=80, verbose=verbose)
        if verbose:
            print(f"🚀 Starting PaperReviewerAgent (model={model})...", flush=True)
        res = await agent.run(
            user_prompt=agent._build_user_prompt(
                paper_dir=paper_dir, artifact_dir=artifact_dir, output_path=out_path
            ),
            system_prompt=agent._build_system_prompt(
                paper_dir=paper_dir, artifact_dir=artifact_dir, output_path=out_path, specs_dir=specs_dir
            ),
            paper_dir=paper_dir,
            artifact_dir=artifact_dir,
            output_path=out_path,
        )
        try:
            final_text = getattr(res, "final_output", None) or ""
        except Exception:
            final_text = ""
        if not os.path.exists(out_path):
            return {
                "success": False,
                "kind": "review",
                "md_path": out_path,
                "output_path": out_path,
                "error": "subagent did not write output_path via tools (required).",
            }
        return {
            "success": True,
            "kind": "review",
            "md_path": out_path,
            "output_path": out_path,
        }

    if kind == "analysis":
        agent = PaperAnalysisAgent(model=model, max_turns=200, verbose=verbose)
        res = await agent.run(
            user_prompt=request
            or agent._build_user_prompt(
                project_dir=project_dir, artifact_dir=artifact_dir
            ),
            system_prompt=agent._build_system_prompt(
                project_dir=project_dir, artifact_dir=artifact_dir, output_path=out_path
            ),
            project_dir=project_dir,
            artifact_dir=artifact_dir,
            output_path=out_path,
        )
        try:
            final_text = getattr(res, "final_output", None) or ""
        except Exception:
            final_text = ""
        if not os.path.exists(out_path):
            return {
                "success": False,
                "kind": "analysis",
                "md_path": out_path,
                "output_path": out_path,
                "error": "subagent did not write output_path via tools (required).",
            }
        return {
            "success": True,
            "kind": "analysis",
            "md_path": out_path,
            "output_path": out_path,
        }

    if kind == "viz":
        agent = PaperVizAgent(model=model, max_turns=200, verbose=verbose)
        res = await agent.run(
            user_prompt=request
            or agent._build_user_prompt(
                project_dir=project_dir, artifact_dir=artifact_dir
            ),
            system_prompt=agent._build_system_prompt(
                project_dir=project_dir,
                artifact_dir=artifact_dir,
                request=request,
                output_path=out_path,
            ),
            project_dir=project_dir,
            artifact_dir=artifact_dir,
            request=request,
        )
        try:
            final_text = getattr(res, "final_output", None) or ""
        except Exception:
            final_text = ""
        if not os.path.exists(out_path):
            return {
                "success": False,
                "kind": "viz",
                "md_path": out_path,
                "output_path": out_path,
                "error": "subagent did not write output_path via tools (required).",
            }
        return {
            "success": True,
            "kind": "viz",
            "md_path": out_path,
            "output_path": out_path,
        }

    if kind == "literature":
        agent = PaperLiteratureAgent(model=model, max_turns=200, verbose=verbose)
        res = await agent.run(
            user_prompt=agent._build_user_prompt(
                artifact_dir=artifact_dir, request=request, output_path=out_path
            ),
            system_prompt=agent._build_system_prompt(
                artifact_dir=artifact_dir, request=request, output_path=out_path
            ),
            artifact_dir=artifact_dir,
            request=request,
            output_path=out_path,
        )
        try:
            final_text = getattr(res, "final_output", None) or ""
        except Exception:
            final_text = ""
        if not os.path.exists(out_path):
            return {
                "success": False,
                "kind": "literature",
                "md_path": out_path,
                "output_path": out_path,
                "error": "subagent did not write output_path via tools (required).",
            }
        return {
            "success": True,
            "kind": "literature",
            "md_path": out_path,
            "output_path": out_path,
        }

    return {"success": False, "error": f"unknown kind: {kind}"}


def main() -> int:
    args = get_args()
    out = asyncio.run(_run(args))
    # Do not print JSON blobs (user preference). Emit a compact one-line summary.
    kind = str((out or {}).get("kind", "") or "")
    ok = bool((out or {}).get("success"))
    out_path = str(
        (out or {}).get("output_path", "") or (out or {}).get("md_path", "") or ""
    )
    err = str((out or {}).get("error", "") or "").strip()
    if ok:
        print(
            f"[paper_agent subagent_runner] kind={kind} success=true output_path={out_path}"
        )
    else:
        print(
            f"[paper_agent subagent_runner] kind={kind} success=false output_path={out_path} error={err or 'unknown'}"
        )
    return 0 if out.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
