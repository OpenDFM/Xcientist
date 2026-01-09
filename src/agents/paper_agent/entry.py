import json
import os
import shutil
from typing import Any, Dict, List, Optional

from src.agents.paper_agent.architect import PaperArchitectAgent
from src.agents.paper_agent.tools.compile import compile_and_vlm_review_impl
from src.agents.paper_agent.tools.core import SecurityContext
from src.agents.paper_agent.writer import PaperWriterAgent


def ensure_run_dirs(output_dir: str, run_name: str) -> Dict[str, str]:
    output_dir = os.path.abspath(str(output_dir or "output"))
    run_name = str(run_name or "").strip()
    if not run_name:
        raise ValueError("run_name is required")

    run_dir = os.path.join(output_dir, run_name)
    paper_dir = os.path.join(run_dir, "paper")
    artifact_dir = os.path.join(run_dir, "artifacts")
    state_dir = os.path.join(run_dir, "state")
    specs_dir = os.path.join(run_dir, "specs")

    os.makedirs(paper_dir, exist_ok=True)
    os.makedirs(os.path.join(artifact_dir, "compile"), exist_ok=True)
    os.makedirs(os.path.join(artifact_dir, "pdf_pages"), exist_ok=True)
    os.makedirs(os.path.join(artifact_dir, "index"), exist_ok=True)
    os.makedirs(os.path.join(artifact_dir, "assets", "figures"), exist_ok=True)
    os.makedirs(os.path.join(artifact_dir, "assets", "tables"), exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(specs_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "paper_dir": paper_dir,
        "artifact_dir": artifact_dir,
        "state_dir": state_dir,
        "specs_dir": specs_dir,
        "state_path": os.path.join(state_dir, "paper_state.json"),
        "run_config_path": os.path.join(state_dir, "paper_agent_config.json"),
    }


def init_minimal_latex_template(paper_dir: str) -> None:
    paper_dir = os.path.abspath(str(paper_dir or ""))
    if not paper_dir:
        raise ValueError("paper_dir is required")
    os.makedirs(paper_dir, exist_ok=True)
    if os.listdir(paper_dir):
        return

    main_tex = os.path.join(paper_dir, "main.tex")
    with open(main_tex, "w", encoding="utf-8") as f:
        f.write(
            "\\documentclass{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{hyperref}\n"
            "\\title{Paper Draft}\n"
            "\\author{}\n"
            "\\date{}\n"
            "\\begin{document}\n"
            "\\maketitle\n"
            "\\begin{abstract}\n"
            "TODO.\n"
            "\\end{abstract}\n"
            "\\section{Introduction}\n"
            "TODO.\n"
            "\\section{Method}\n"
            "TODO.\n"
            "\\section{Experiments}\n"
            "TODO.\n"
            "\\section{Conclusion}\n"
            "TODO.\n"
            "\\end{document}\n"
        )


def copy_template_dir(template_dir: str, paper_dir: str) -> None:
    template_dir = os.path.abspath(str(template_dir or ""))
    paper_dir = os.path.abspath(str(paper_dir or ""))
    if not template_dir:
        init_minimal_latex_template(paper_dir=paper_dir)
        return
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"template_dir not found: {template_dir}")

    def _has_tex_files(d: str) -> bool:
        try:
            if not os.path.isdir(d):
                return False
            for fn in os.listdir(d):
                if str(fn).lower().endswith(".tex"):
                    return True
            return os.path.isfile(os.path.join(d, "main.tex"))
        except Exception:
            return False

    # If paper_dir already exists but does not look like a LaTeX project, it is likely polluted
    # (e.g. mistakenly contains repo code). Backup and re-initialize from template.
    if os.path.isdir(paper_dir) and os.listdir(paper_dir):
        if _has_tex_files(paper_dir):
            return
        parent = os.path.dirname(paper_dir)
        base = os.path.basename(paper_dir.rstrip(os.sep)) or "paper"
        backup_dir = os.path.join(parent, f"{base}.bak")
        idx = 1
        while os.path.exists(backup_dir):
            backup_dir = os.path.join(parent, f"{base}.bak{idx}")
            idx += 1
        shutil.move(paper_dir, backup_dir)
        os.makedirs(paper_dir, exist_ok=True)

    shutil.copytree(template_dir, paper_dir, dirs_exist_ok=True)


def _ensure_not_within(path: str, forbidden_root: str) -> None:
    ap = os.path.abspath(str(path or ""))
    fr = os.path.abspath(str(forbidden_root or ""))
    if not ap or not fr:
        return
    try:
        if os.path.commonpath([ap, fr]) == fr:
            raise ValueError(f"forbidden path (must not be within {fr}): {ap}")
    except ValueError:
        raise
    except Exception:
        return


def _ensure_nonempty_file(path: str) -> None:
    p = os.path.abspath(str(path or ""))
    if not os.path.isfile(p):
        raise FileNotFoundError(f"required file not found: {p}")
    try:
        if os.path.getsize(p) <= 0:
            raise ValueError(f"required file is empty: {p}")
    except Exception:
        pass


def write_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


async def run_paper_cycle(
    run_name: str,
    template_dir: str,
    idea_md: str,
    project_dir: str,
    output_dir: str = "output",
    model: str = "gpt-5.2",
    models: Optional[Dict[str, str]] = None,
    compile_first: bool = True,
    run_writer: bool = True,
    run_architect: bool = True,
    final_compile_with_vlm: bool = False,
    experiment_id: str = "",
    experiment_workspace_dir: str = "",
    experiment_idea_md: str = "",
    experiment_specs_dir: str = "",
    experiment_spec_files: Optional[List[str]] = None,
    experiment_result_files: Optional[List[str]] = None,
    verbose: bool = True,
    resume: bool = False,
) -> Dict[str, Any]:
    paths = ensure_run_dirs(output_dir=output_dir, run_name=run_name)
    paper_dir = paths["paper_dir"]
    artifact_dir = paths["artifact_dir"]
    specs_dir = paths["specs_dir"]
    state_path = paths["state_path"]
    run_config_path = paths["run_config_path"]

    template_dir = os.path.abspath(str(template_dir or "")) if template_dir else ""
    idea_md = os.path.abspath(str(idea_md or ""))
    project_dir = os.path.abspath(str(project_dir or ""))
    experiment_workspace_dir = (
        os.path.abspath(str(experiment_workspace_dir or ""))
        if experiment_workspace_dir
        else ""
    )
    experiment_idea_md = (
        os.path.abspath(str(experiment_idea_md or "")) if experiment_idea_md else ""
    )
    experiment_specs_dir = (
        os.path.abspath(str(experiment_specs_dir or "")) if experiment_specs_dir else ""
    )

    effective_models: Dict[str, str] = dict(models or {})
    default_model = str(model or "gpt-5.2")
    if "default" not in effective_models:
        effective_models["default"] = default_model
    for key in [
        "architect",
        "writer",
        "analysis",
        "literature",
        "review",
        "viz",
        "vlm",
    ]:
        if key not in effective_models:
            effective_models[key] = default_model

    prev_state: Dict[str, Any] = {}
    if bool(resume) and os.path.exists(state_path):
        prev_state = read_state(state_path)
        try:
            prev_paths = (
                prev_state.get("paths")
                if isinstance(prev_state.get("paths"), dict)
                else {}
            )
            prev_template = str(prev_paths.get("template_dir", "") or "").strip()
        except Exception:
            prev_template = ""
        if prev_template:
            template_dir = os.path.abspath(prev_template)
    else:
        copy_template_dir(template_dir=template_dir, paper_dir=paper_dir)

    if template_dir and experiment_workspace_dir:
        _ensure_not_within(path=template_dir, forbidden_root=experiment_workspace_dir)

    SecurityContext.set_roots(
        project_root=paper_dir, workspace_root=os.path.dirname(paper_dir)
    )
    SecurityContext.set_access(
        read_roots=[
            paper_dir,
            artifact_dir,
            specs_dir,
            project_dir,
            experiment_workspace_dir,
        ],
        write_roots=[paper_dir, artifact_dir, specs_dir],
    )

    state: Dict[str, Any] = {
        "run_name": run_name,
        "paths": {
            "template_dir": template_dir,
            "paper_dir": paper_dir,
            "artifact_dir": artifact_dir,
            "idea_md": idea_md,
            "project_dir": project_dir,
            "specs_dir": specs_dir,
            "experiment": {
                "experiment_id": str(experiment_id or ""),
                "workspace_dir": str(experiment_workspace_dir or ""),
                "idea_md": str(experiment_idea_md or ""),
                "specs_dir": str(experiment_specs_dir or ""),
                "spec_files": experiment_spec_files or [],
                "result_files": experiment_result_files or [],
            },
        },
        "status": {
            "initialized": True,
            "compiled_once": False,
            "architect_ran": False,
            "writer_ran": False,
        },
    }
    if prev_state:
        prev_status = (
            prev_state.get("status")
            if isinstance(prev_state.get("status"), dict)
            else {}
        )
        state_status = (
            state.get("status") if isinstance(state.get("status"), dict) else {}
        )
        for k in ["architect_ran", "writer_ran", "compiled_once"]:
            if k in prev_status:
                state_status[k] = prev_status.get(k)
        state["status"] = state_status
    write_state(state_path, state)

    try:
        from src.agents.paper_agent.utils.config import (
            PaperAgentRunConfig,
            write_run_config,
        )

        cfg = PaperAgentRunConfig(
            run_name=str(run_name),
            paper_dir=str(paper_dir),
            project_dir=str(project_dir),
            artifact_dir=str(artifact_dir),
            model=str(model),
            models=dict(effective_models),
            experiment_id=str(experiment_id or ""),
            experiment_workspace_dir=str(experiment_workspace_dir or ""),
            experiment_idea_md=str(experiment_idea_md or ""),
            experiment_specs_dir=str(experiment_specs_dir or ""),
            experiment_spec_files=experiment_spec_files or [],
            experiment_result_files=experiment_result_files or [],
        )
        write_run_config(run_config_path, cfg)
        os.environ["PAPER_AGENT_RUN_CONFIG_PATH"] = str(run_config_path)
    except Exception:
        pass

    if (
        bool(resume)
        and isinstance(state.get("status"), dict)
        and state["status"].get("architect_ran") is True
    ):
        run_architect = False

    if run_architect:
        spec_template_path = os.path.join(
            os.path.dirname(__file__), "templates", "spec.md"
        )
        plan_template_path = os.path.join(
            os.path.dirname(__file__), "templates", "plan.md"
        )
        spec_template = _read_text(spec_template_path)
        plan_template = _read_text(plan_template_path)

        architect = PaperArchitectAgent(
            model=str(effective_models.get("architect") or default_model),
            max_turns=999,
            verbose=verbose,
        )
        _ = await architect.run(
            user_prompt=architect._build_user_prompt(
                specs_dir=specs_dir,
                idea_path=idea_md,
                project_dir=project_dir,
                paper_dir=paper_dir,
                experiment_id=str(experiment_id or ""),
                experiment_workspace_dir=str(experiment_workspace_dir or ""),
                experiment_specs_dir=str(experiment_specs_dir or ""),
                experiment_spec_files=experiment_spec_files or [],
                experiment_result_files=experiment_result_files or [],
            ),
            system_prompt=architect._build_system_prompt(
                specs_dir=specs_dir,
                idea_path=idea_md,
                project_dir=project_dir,
                paper_dir=paper_dir,
                spec_template=spec_template,
                plan_template=plan_template,
                experiment_id=str(experiment_id or ""),
                experiment_workspace_dir=str(experiment_workspace_dir or ""),
                experiment_specs_dir=str(experiment_specs_dir or ""),
                experiment_spec_files=experiment_spec_files or [],
                experiment_result_files=experiment_result_files or [],
            ),
            specs_dir=specs_dir,
            idea_path=idea_md,
            project_dir=project_dir,
            paper_dir=paper_dir,
        )
        state["status"]["architect_ran"] = True
        write_state(state_path, state)

        _ensure_nonempty_file(os.path.join(specs_dir, "spec.md"))
        _ensure_nonempty_file(os.path.join(specs_dir, "plan.md"))

    compile_result: Optional[Dict[str, Any]] = None
    if compile_first:
        compile_result = compile_and_vlm_review_impl(
            paper_dir=paper_dir,
            artifact_dir=artifact_dir,
            main_tex=None,
            compile_timeout_sec=600,
            vlm_mode="compile_only",
            vlm_page_strategy=None,
        )
        state["status"]["compiled_once"] = bool(compile_result.get("compile_success"))
        state["last_compile"] = compile_result
        write_state(state_path, state)

    if not run_writer:
        return {"paths": paths, "compile_first": compile_result}

    SecurityContext.set_access(
        read_roots=[
            paper_dir,
            artifact_dir,
            specs_dir,
            project_dir,
            experiment_workspace_dir,
        ],
        write_roots=[paper_dir, artifact_dir],
    )

    writer = PaperWriterAgent(
        model=str(effective_models.get("writer") or default_model),
        max_turns=999,
        verbose=verbose,
    )
    _ = await writer.run(
        user_prompt=writer._build_user_prompt(),
        system_prompt=writer._build_system_prompt(
            paper_dir=paper_dir,
            artifact_dir=artifact_dir,
            specs_dir=specs_dir,
            idea_path=idea_md,
            project_dir=project_dir,
        ),
        paper_dir=paper_dir,
        artifact_dir=artifact_dir,
        specs_dir=specs_dir,
        idea_path=idea_md,
        project_dir=project_dir,
    )
    state["status"]["writer_ran"] = True
    write_state(state_path, state)

    # Compilation/VLM review is handled by reviewer sub-agent. Keep this flag for backward compatibility.
    if bool(final_compile_with_vlm):
        pass

    return {"paths": paths, "compile_first": compile_result}
