import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agents import function_tool

from src.agents.paper_agent.tools.vlm import vlm_layout_review_from_pages
from src.agents.paper_agent.utils.config import PAPER_COMPILE_DOCKER_IMAGE


@dataclass
class CompileIssueIndex:
    missing_files: List[str]
    undefined_citations: List[str]
    undefined_references: List[str]
    overfull_hbox: List[dict]
    errors: List[str]
    warnings: List[str]


def _which(cmd: str) -> str:
    try:
        import shutil

        return shutil.which(cmd) or ""
    except Exception:
        return ""


def _read_text(path: str, max_bytes: int = 2_000_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)
    except Exception:
        return ""


def _write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, obj: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _find_main_tex(paper_dir: str, main_tex: Optional[str]) -> Tuple[str, List[str]]:
    attempts: List[str] = []

    if main_tex:
        candidate = (
            main_tex if os.path.isabs(main_tex) else os.path.join(paper_dir, main_tex)
        )
        attempts.append(candidate)
        return candidate, attempts

    for name in ("main.tex", "paper.tex", "arxiv_main.tex"):
        candidate = os.path.join(paper_dir, name)
        if os.path.exists(candidate):
            attempts.append(candidate)
            return candidate, attempts

    tex_candidates: List[str] = []
    for root, _, files in os.walk(paper_dir):
        for fn in files:
            if fn.lower().endswith(".tex"):
                tex_candidates.append(os.path.join(root, fn))

    docclass_candidates: List[str] = []
    for p in sorted(tex_candidates):
        attempts.append(p)
        txt = _read_text(p, max_bytes=200_000)
        if "\\documentclass" in txt:
            docclass_candidates.append(p)

    if docclass_candidates:
        return docclass_candidates[0], attempts

    return (os.path.join(paper_dir, "main.tex"), attempts)


def _extract_issues_from_logs(text: str) -> CompileIssueIndex:
    missing_files: List[str] = []
    undefined_citations: List[str] = []
    undefined_references: List[str] = []
    overfull_hbox: List[dict] = []
    errors: List[str] = []
    warnings: List[str] = []

    if not text:
        return CompileIssueIndex(
            missing_files=missing_files,
            undefined_citations=undefined_citations,
            undefined_references=undefined_references,
            overfull_hbox=overfull_hbox,
            errors=errors,
            warnings=warnings,
        )

    for m in re.finditer(r"LaTeX Error: File `([^']+)' not found", text):
        missing_files.append(m.group(1))
    for m in re.finditer(r"Package .* Error: File `([^']+)' not found", text):
        missing_files.append(m.group(1))
    for m in re.finditer(r"LaTeX Warning: File `([^']+)' not found", text):
        missing_files.append(m.group(1))

    for m in re.finditer(r"LaTeX Warning: Citation `([^']+)' on page", text):
        undefined_citations.append(m.group(1))
    for m in re.finditer(r"LaTeX Warning: Citation '([^']+)' on page", text):
        undefined_citations.append(m.group(1))

    for m in re.finditer(r"LaTeX Warning: Reference `([^']+)' on page", text):
        undefined_references.append(m.group(1))
    for m in re.finditer(r"LaTeX Warning: Reference '([^']+)' on page", text):
        undefined_references.append(m.group(1))

    for m in re.finditer(
        r"Overfull \\hbox \\(([^)]+)\\) in paragraph at lines ([0-9]+)--([0-9]+)", text
    ):
        overfull_hbox.append(
            {
                "detail": m.group(1),
                "start_line": int(m.group(2)),
                "end_line": int(m.group(3)),
            }
        )

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("! "):
            errors.append(s[2:].strip())
        if s.startswith("LaTeX Warning:"):
            warnings.append(s)
        if "Undefined citations" in s or "There were undefined references" in s:
            warnings.append(s)

    def _dedup(items: List[Any]) -> List[Any]:
        seen = set()
        out = []
        for x in items:
            k = (
                json.dumps(x, sort_keys=True, ensure_ascii=False)
                if isinstance(x, (dict, list))
                else str(x)
            )
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    return CompileIssueIndex(
        missing_files=_dedup([str(x) for x in missing_files if str(x).strip()]),
        undefined_citations=_dedup(
            [str(x) for x in undefined_citations if str(x).strip()]
        ),
        undefined_references=_dedup(
            [str(x) for x in undefined_references if str(x).strip()]
        ),
        overfull_hbox=_dedup(overfull_hbox),
        errors=_dedup([str(x) for x in errors if str(x).strip()]),
        warnings=_dedup([str(x) for x in warnings if str(x).strip()]),
    )


def _run_subprocess(
    cmd: List[str],
    cwd: str,
    timeout_sec: int,
) -> Tuple[int, str, str, float]:
    start = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
        )
        return p.returncode, p.stdout or "", p.stderr or "", time.time() - start
    except subprocess.TimeoutExpired as e:
        out = ""
        err = ""
        try:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        except Exception:
            out = ""
        try:
            err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        except Exception:
            err = ""
        err = (err + "\n" if err else "") + f"[compile timeout] exceeded {timeout_sec}s"
        return 124, out, err, time.time() - start
    except Exception as e:
        return 1, "", str(e), time.time() - start


def _render_pdf_pages_with_pypdfium2(
    pdf_path: str, out_dir: str, dpi: int, max_pages: int
) -> Dict[str, Any]:
    try:
        import pypdfium2 as pdfium
    except Exception as e:
        return {
            "success": False,
            "error": f"pypdfium2 not available: {type(e).__name__}: {e}",
        }

    try:
        os.makedirs(out_dir, exist_ok=True)
        pdf = pdfium.PdfDocument(pdf_path)
        n_pages = len(pdf)
        to_render = min(int(max_pages), int(n_pages))
        rendered = []

        scale = float(dpi) / 72.0
        for i in range(to_render):
            page = pdf.get_page(i)
            pil = page.render(scale=scale).to_pil()
            fn = f"page_{i + 1:03d}.png"
            out_path = os.path.join(out_dir, fn)
            pil.save(out_path)
            rendered.append(out_path)
            try:
                page.close()
            except Exception:
                pass

        try:
            pdf.close()
        except Exception:
            pass

        return {"success": True, "rendered_pages": rendered, "total_pages": n_pages}
    except Exception as e:
        return {"success": False, "error": f"render failed: {type(e).__name__}: {e}"}


def _run_docker_compile(
    docker_image: str,
    paper_dir: str,
    artifact_dir: str,
    main_tex_rel: str,
    compile_dir_rel: str,
    timeout_sec: int,
) -> Tuple[int, str, str, float]:
    uid = os.getuid()
    gid = os.getgid()

    if not _which("docker"):
        return 1, "", "Docker command not found", 0.0

    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        "--network",
        "none",
        "-v",
        f"{paper_dir}:/src",
        "-v",
        f"{artifact_dir}:/out",
        "-w",
        "/src",
        docker_image,
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-outdir=/out/{compile_dir_rel}",
        main_tex_rel,
    ]
    return _run_subprocess(cmd, cwd=paper_dir, timeout_sec=timeout_sec)


def compile_and_vlm_review_impl(
    paper_dir: str,
    artifact_dir: str,
    main_tex: Optional[str] = None,
    compile_timeout_sec: int = 600,
) -> Dict[str, Any]:
    paper_dir = os.path.abspath(str(paper_dir or ""))
    artifact_dir = os.path.abspath(str(artifact_dir or ""))

    compile_dir = os.path.join(artifact_dir, "compile")
    pages_dir = os.path.join(artifact_dir, "pdf_pages")
    index_dir = os.path.join(artifact_dir, "index")
    os.makedirs(compile_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    main_tex_path, main_attempts = _find_main_tex(paper_dir, main_tex)
    main_tex_used = (
        os.path.relpath(main_tex_path, paper_dir)
        if main_tex_path.startswith(paper_dir)
        else main_tex_path
    )

    tectonic = _which("tectonic")
    latexmk = _which("latexmk")
    pdflatex = _which("pdflatex")

    toolchain = ""
    cmd: List[str] = []
    if tectonic:
        toolchain = "tectonic"
        cmd = [
            tectonic,
            "--keep-intermediates",
            "--keep-logs",
            "--outdir",
            compile_dir,
            main_tex_used,
        ]
    elif latexmk:
        toolchain = "latexmk"
        cmd = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            "-use-biber",
            f"-outdir={compile_dir}",
            main_tex_used,
        ]
    elif pdflatex:
        toolchain = "pdflatex"
        cmd = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={compile_dir}",
            main_tex_used,
        ]
    else:
        issue = CompileIssueIndex(
            missing_files=[],
            undefined_citations=[],
            undefined_references=[],
            overfull_hbox=[],
            errors=[
                "pdflatex not found. Please install TeX Live or similar."
            ],
            warnings=[],
        )
        return {
            "compile_success": False,
            "main_tex_used": main_tex_used,
            "main_tex_attempts": main_attempts,
            "toolchain": "",
            "pdf_path": "",
            "logs": {},
            "issues": issue.__dict__,
            "vlm_layout_review": {
                "success": False,
                "skipped": True,
                "reason": "compile failed (pdflatex missing)",
            },
        }

    rc, out, err, elapsed = _run_subprocess(
        cmd, cwd=paper_dir, timeout_sec=int(compile_timeout_sec)
    )

    stdout_path = os.path.join(compile_dir, "compile.stdout.txt")
    stderr_path = os.path.join(compile_dir, "compile.stderr.txt")
    _write_text(stdout_path, out)
    _write_text(stderr_path, err)

    base = os.path.splitext(os.path.basename(main_tex_used))[0]
    pdf_path = os.path.join(compile_dir, f"{base}.pdf")
    log_path = os.path.join(compile_dir, f"{base}.log")

    log_text = _read_text(log_path) if os.path.exists(log_path) else ""
    merged_text = "\n".join([out, err, log_text])
    issues = _extract_issues_from_logs(merged_text)

    compile_success = (rc == 0) and os.path.exists(pdf_path)

    logs = {
        "stdout": stdout_path,
        "stderr": stderr_path,
    }
    if os.path.exists(log_path):
        logs["latex_log"] = log_path

    vlm_review: Dict[str, Any] = {
        "success": False,
        "skipped": True,
        "reason": "vlm_mode=compile_only",
    }

    # Always enable VLM review with fixed strategy
    if compile_success:
        dpi = 250
        first_n_pages = 3
        max_pages = 10
        render_res = _render_pdf_pages_with_pypdfium2(
            pdf_path=pdf_path,
            out_dir=pages_dir,
            dpi=dpi,
            max_pages=max_pages,
        )
        reviewed_pages = []
        if isinstance(render_res, dict) and render_res.get("success") is True:
            total_pages = int(render_res.get("total_pages", 0) or 0)
            rendered_count = min(int(max_pages), total_pages) if total_pages > 0 else 0
            reviewed_pages = list(
                range(1, min(max(int(first_n_pages), 0), rendered_count) + 1)
            )
        vlm_review = {
            "success": False,
            "reviewed_pages": reviewed_pages,
            "findings": [],
            "render": render_res,
        }
        model = ""
        try:
            from src.agents.paper_agent.utils.config import get_runtime_config

            cfg = get_runtime_config()
            cfg_models = getattr(cfg, "models", None) or {}
            model = str(cfg_models.get("vlm") or "").strip()
        except Exception:
            model = ""
        if not model:
            from src.agents.paper_agent.utils.config import PAPER_VLM_MODEL

            model = str(PAPER_VLM_MODEL or "").strip()

        from src.agents.paper_agent.utils.config import get_openai_config

        provider_cfg = get_openai_config(model=model)
        api_key = str(provider_cfg.get("api_key", "") or "").strip()
        base_url = str(provider_cfg.get("base_url", "") or "").strip()
        extra_body = provider_cfg.get("extra_body", None)
        if (
            model
            and api_key
            and isinstance(render_res, dict)
            and render_res.get("success") is True
        ):
            pngs = list(render_res.get("rendered_pages") or [])
            pngs = [p for p in pngs if isinstance(p, str) and p and os.path.exists(p)]
            pngs = pngs[: max(1, int(first_n_pages))]
            if pngs:
                vlm_review = vlm_layout_review_from_pages(
                    pngs,
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    extra_body=extra_body,
                    max_pages=len(pngs),
                )
                vlm_review["render"] = render_res
            else:
                vlm_review["error"] = "No rendered PNG pages found for VLM review."
        else:
            vlm_review["error"] = (
                "VLM not configured (set PAPER_VLM_MODEL / PAPER_AGENT_VLM_MODEL and provider API key env vars)."
            )

    result = {
        "compile_success": bool(compile_success),
        "main_tex_used": main_tex_used,
        "main_tex_attempts": main_attempts,
        "toolchain": toolchain,
        "return_code": int(rc),
        "elapsed_sec": float(elapsed),
        "pdf_path": pdf_path if os.path.exists(pdf_path) else "",
        "logs": logs,
        "issues": issues.__dict__,
        "vlm_layout_review": vlm_review,
    }

    # User preference: avoid writing JSON artifacts; write Markdown summaries instead.
    try:
        _write_text(
            os.path.join(index_dir, "compile_result.md"),
            "\n".join(
                [
                    "# Compile Result",
                    f"- compile_success: {bool(compile_success)}",
                    f"- toolchain: {toolchain}",
                    f"- return_code: {int(rc)}",
                    f"- elapsed_sec: {float(elapsed):.3f}",
                    f"- pdf_path: {pdf_path if os.path.exists(pdf_path) else ''}",
                    f"- main_tex_used: {main_tex_used}",
                    "",
                    "## Logs",
                    f"- stdout: {stdout_path}",
                    f"- stderr: {stderr_path}",
                    f"- latex_log: {logs.get('latex_log','')}",
                    "",
                    "## Issues",
                    f"- missing_files: {len(issues.missing_files)}",
                    f"- undefined_citations: {len(issues.undefined_citations)}",
                    f"- undefined_references: {len(issues.undefined_references)}",
                    f"- overfull_hbox: {len(issues.overfull_hbox)}",
                    f"- errors: {len(issues.errors)}",
                    f"- warnings: {len(issues.warnings)}",
                    "",
                ]
            )
            + "\n",
        )
    except Exception:
        pass
    try:
        _write_text(
            os.path.join(index_dir, "vlm_layout_review.md"),
            "\n".join(
                [
                    "# VLM Layout Review",
                    f"- enabled: {str(vlm_mode or '').strip().lower() == 'compile_and_review'}",
                    f"- success: {bool((vlm_review or {}).get('success')) if isinstance(vlm_review, dict) else False}",
                    f"- reviewed_pages: {len((vlm_review or {}).get('reviewed_pages', []) or []) if isinstance(vlm_review, dict) else 0}",
                    f"- findings: {len((vlm_review or {}).get('findings', []) or []) if isinstance(vlm_review, dict) else 0}",
                    f"- error: {str((vlm_review or {}).get('error', '') or '') if isinstance(vlm_review, dict) else ''}",
                    "",
                ]
            )
            + "\n",
        )
    except Exception:
        pass

    return result


@function_tool
def compile_and_vlm_review(
    paper_dir: str,
    artifact_dir: str,
    main_tex: str = "",
    compile_timeout_sec: int = 600,
) -> dict:
    """
    Compile a LaTeX project into PDF using pdflatex, then run VLM layout review after PDF is generated.

    Notes:
    - For logging/UX, this tool returns:
      - success: bool (mirrors compile_success)
      - message: short human-readable summary
    """
    try:
        res = compile_and_vlm_review_impl(
            paper_dir=paper_dir,
            artifact_dir=artifact_dir,
            main_tex=(main_tex.strip() or None),
            compile_timeout_sec=int(compile_timeout_sec),
        )
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"compile_and_vlm_review failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
    if not isinstance(res, dict):
        return {"success": False, "error": "compile tool returned non-dict result"}
    ok = bool(res.get("compile_success"))
    pdf_path = str(res.get("pdf_path") or "")
    issues = res.get("issues") if isinstance(res.get("issues"), dict) else {}
    err_n = len(issues.get("errors", []) or []) if isinstance(issues, dict) else 0
    warn_n = len(issues.get("warnings", []) or []) if isinstance(issues, dict) else 0
    vlm = (
        res.get("vlm_layout_review")
        if isinstance(res.get("vlm_layout_review"), dict)
        else {}
    )
    vlm_ok = bool(vlm.get("success")) if isinstance(vlm, dict) else False
    findings = vlm.get("findings") if isinstance(vlm.get("findings"), list) else []
    
    # Construct detailed error message
    error_details = []
    if isinstance(issues, dict):
        if issues.get("missing_files"):
            error_details.append(f"Missing files: {issues['missing_files']}")
        if issues.get("errors"):
            # Limit errors to avoid hitting context limits, but give enough info
            errors = issues['errors']
            error_details.append(f"LaTeX Errors: {errors[:10]}") 

    msg = f"compile_success={ok}, pdf={pdf_path or '(none)'}, issues: errors={err_n}, warnings={warn_n}, vlm_success={vlm_ok}, vlm_findings={len(findings)}"
    
    res["success"] = ok
    res["message"] = msg
    
    if not ok and "error" not in res:
        res["error"] = "; ".join(error_details) if error_details else "Compile failed (check logs for details)"
        # Also append to message for visibility
        if error_details:
             res["message"] += f". Details: {'; '.join(error_details)}"
             
    return res

