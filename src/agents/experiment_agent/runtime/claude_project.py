from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.agents.experiment_agent.agents.code.planner import EXPERIMENT_CODE_PLANNER
from src.agents.experiment_agent.agents.code.validator import CODE_VALIDATOR, code_validator_prompt
from src.agents.experiment_agent.agents.code.worker import CODE_WORKER, code_worker_prompt
from src.agents.experiment_agent.agents.prepare.validator import (
    PREPARE_VALIDATOR,
    prepare_validator_prompt,
)
from src.agents.experiment_agent.agents.prepare.worker import (
    PREPARE_DATASET_WORKER,
    PREPARE_ENV_WORKER,
    PREPARE_MODEL_WORKER,
    PREPARE_REPO_WORKER,
    PREPARE_SYNTHESIS_WORKER,
    prepare_worker_prompt,
)
from src.agents.experiment_agent.agents.reporting.integrator import (
    EXPERIMENT_ABLATION_REPORT_INTEGRATOR,
    ablation_report_integrator_prompt,
)
from src.agents.experiment_agent.agents.science.planner import (
    EXPERIMENT_ABLATION_SCIENCE_PLANNER,
    EXPERIMENT_STANDARD_SCIENCE_PLANNER,
)
from src.agents.experiment_agent.agents.science.validator import (
    ABLATION_SCIENCE_VALIDATOR,
    STANDARD_SCIENCE_VALIDATOR,
    ablation_science_validator_prompt,
    standard_science_validator_prompt,
)
from src.agents.experiment_agent.agents.science.worker import (
    ABLATION_SCIENCE_WORKER,
    STANDARD_SCIENCE_WORKER,
    ablation_science_worker_prompt,
    standard_science_worker_prompt,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _yaml_list(values: Iterable[str]) -> str:
    items = [value for value in values if value]
    if not items:
        return "[]"
    return "\n".join(f"  - {value}" for value in items)


def _agent_markdown(spec: Dict[str, Any]) -> str:
    frontmatter = [
        "---",
        f"name: {spec['name']}",
        f"description: {spec['description']}",
        f"model: {spec['model']}",
        f"permissionMode: {spec.get('permission_mode', 'bypassPermissions')}",
        f"maxTurns: {int(spec.get('max_turns', 12))}",
    ]
    tools = list(spec.get("tools") or [])
    if tools:
        frontmatter.append("tools:")
        frontmatter.append(_yaml_list(tools))
    skills = list(spec.get("skills") or [])
    if skills:
        frontmatter.append("skills:")
        frontmatter.append(_yaml_list(skills))
    mcp_servers = list(spec.get("mcp_servers") or [])
    if mcp_servers:
        frontmatter.append("mcpServers:")
        frontmatter.append(_yaml_list(mcp_servers))
    frontmatter.extend(["---", "", spec["prompt"].strip()])
    return "\n".join(frontmatter)


def _command_markdown(name: str, body: str) -> str:
    return "\n".join(
        [
            "---",
            f"description: Run the {name} workflow inside the current experiment workspace",
            "---",
            "",
            body.strip(),
        ]
    )


def _workspace_claude_md() -> str:
    return """# Experiment Workspace Rules

- The current working directory is the experiment workspace root.
- Treat `idea.json` as the only canonical structured experiment input.
- Restrict all file reads and writes to workspace-local paths such as `project/`, `agent_reports/`, `results/`, `dataset_candidate/`, `model_candidate/`, and `repos/`.
- Keep runtime code self-contained under `project/`; `repos/` is reference-only.
- Never wait for interactive approval. If an action cannot complete, return structured failure evidence instead.
- Do not write `ablation_results.json` unless you are the dedicated ablation report integrator.
"""


def _planner_prompt(label: str) -> str:
    return f"""You are the {label} planner for this experiment workspace.

Your job is to return only the structured plan requested by the runtime.

Rules:
- Work only inside the current workspace root and its declared subpaths.
- Ground every step in real workspace evidence.
- Do not execute the whole phase yourself; output contracts for the worker and validator chain.
- Do not invent paths, datasets, or model assets that are not present in the workspace.
"""


def _command_body(agent_name: str, task: str) -> str:
    return f"""Use the `@{agent_name}` subagent to {task}.

Respect the workspace contract in `CLAUDE.md` and return structured JSON when the runtime schema requires it.
"""


def _load_json_file(path: str) -> Dict[str, Any]:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _merge_dict(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _agent_specs(role_models: Dict[str, str], mcp_servers: List[str]) -> List[Dict[str, Any]]:
    planner_skills = ["prepare-planning", "code-planning", "science-planning", "component-coverage", "convergence-gate"]
    worker_skills = ["bounded-tool-use", "environment-setup", "resource-acquisition", "code-enablement", "science-execution"]
    validator_skills = ["bounded-tool-use", "component-coverage", "convergence-gate"]
    integrator_skills = ["bounded-tool-use", "component-coverage", "iteration-integration"]
    common_tools = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]
    reviewer_tools = ["Read", "Glob", "Grep", "Bash"]
    return [
        {
            "name": "prepare-planner",
            "description": "Prepare phase planner. Use for staging repo, env, dataset, model, and synthesis contracts.",
            "model": role_models["planner"],
            "tools": common_tools,
            "skills": planner_skills,
            "mcp_servers": mcp_servers,
            "prompt": _planner_prompt("prepare phase"),
        },
        {
            "name": PREPARE_REPO_WORKER.replace("_", "-"),
            "description": "Prepare repo worker. Use for executing the repos stage inside the experiment workspace.",
            "model": role_models["worker"],
            "tools": common_tools,
            "skills": worker_skills,
            "mcp_servers": mcp_servers,
            "prompt": prepare_worker_prompt("repos"),
        },
        {
            "name": PREPARE_ENV_WORKER.replace("_", "-"),
            "description": "Prepare env worker. Use for executing the env stage inside the experiment workspace.",
            "model": role_models["worker"],
            "tools": common_tools,
            "skills": worker_skills,
            "mcp_servers": mcp_servers,
            "prompt": prepare_worker_prompt("env"),
        },
        {
            "name": PREPARE_DATASET_WORKER.replace("_", "-"),
            "description": "Prepare dataset worker. Use for executing the dataset stage inside the experiment workspace.",
            "model": role_models["worker"],
            "tools": common_tools,
            "skills": worker_skills,
            "mcp_servers": mcp_servers,
            "prompt": prepare_worker_prompt("dataset"),
        },
        {
            "name": PREPARE_MODEL_WORKER.replace("_", "-"),
            "description": "Prepare model worker. Use for executing the model stage inside the experiment workspace.",
            "model": role_models["worker"],
            "tools": common_tools,
            "skills": worker_skills,
            "mcp_servers": mcp_servers,
            "prompt": prepare_worker_prompt("model"),
        },
        {
            "name": PREPARE_VALIDATOR.replace("_", "-"),
            "description": "Prepare phase validator. Use to review a single prepare stage from concrete workspace evidence.",
            "model": role_models["validator"],
            "tools": reviewer_tools,
            "skills": validator_skills,
            "prompt": prepare_validator_prompt(),
        },
        {
            "name": PREPARE_SYNTHESIS_WORKER.replace("_", "-"),
            "description": "Prepare synthesis worker. Use for executing the synthesis stage inside the experiment workspace.",
            "model": role_models["worker"],
            "tools": common_tools,
            "skills": worker_skills,
            "mcp_servers": mcp_servers,
            "prompt": prepare_worker_prompt("synthesis"),
        },
        {
            "name": EXPERIMENT_CODE_PLANNER.replace("_", "-"),
            "description": "Code phase planner. Use for building validator-backed code execution plans.",
            "model": role_models["planner"],
            "tools": common_tools,
            "skills": planner_skills,
            "mcp_servers": mcp_servers,
            "prompt": _planner_prompt("code phase"),
        },
        {
            "name": CODE_WORKER.replace("_", "-"),
            "description": "Code phase worker. Use for implementing one experiment code step.",
            "model": role_models["worker"],
            "tools": common_tools,
            "skills": worker_skills,
            "mcp_servers": mcp_servers,
            "prompt": code_worker_prompt(),
        },
        {
            "name": CODE_VALIDATOR.replace("_", "-"),
            "description": "Code phase validator. Use for reviewing one code step from local evidence.",
            "model": role_models["validator"],
            "tools": reviewer_tools,
            "skills": validator_skills,
            "prompt": code_validator_prompt(),
        },
        {
            "name": EXPERIMENT_STANDARD_SCIENCE_PLANNER.replace("_", "-"),
            "description": "Standard science planner. Use for benchmark and full-method science plans.",
            "model": role_models["planner"],
            "tools": common_tools,
            "skills": planner_skills,
            "mcp_servers": mcp_servers,
            "prompt": _planner_prompt("standard science"),
        },
        {
            "name": STANDARD_SCIENCE_WORKER.replace("_", "-"),
            "description": "Standard science worker. Use for executing one standard science step.",
            "model": role_models["worker"],
            "tools": common_tools,
            "skills": worker_skills,
            "mcp_servers": mcp_servers,
            "prompt": standard_science_worker_prompt(),
        },
        {
            "name": STANDARD_SCIENCE_VALIDATOR.replace("_", "-"),
            "description": "Standard science validator. Use for reviewing one standard science step from concrete evidence.",
            "model": role_models["validator"],
            "tools": reviewer_tools,
            "skills": validator_skills,
            "prompt": standard_science_validator_prompt(),
        },
        {
            "name": EXPERIMENT_ABLATION_SCIENCE_PLANNER.replace("_", "-"),
            "description": "Ablation science planner. Use for per-component ablation plans that preserve canonical component order.",
            "model": role_models["planner"],
            "tools": common_tools,
            "skills": planner_skills,
            "mcp_servers": mcp_servers,
            "prompt": _planner_prompt("ablation science"),
        },
        {
            "name": ABLATION_SCIENCE_WORKER.replace("_", "-"),
            "description": "Ablation science worker. Use for executing one ablation component step.",
            "model": role_models["worker"],
            "tools": common_tools,
            "skills": worker_skills,
            "mcp_servers": mcp_servers,
            "prompt": ablation_science_worker_prompt(),
        },
        {
            "name": ABLATION_SCIENCE_VALIDATOR.replace("_", "-"),
            "description": "Ablation science validator. Use for reviewing one ablation step and extracting component results.",
            "model": role_models["validator"],
            "tools": reviewer_tools,
            "skills": validator_skills,
            "prompt": ablation_science_validator_prompt(),
        },
        {
            "name": EXPERIMENT_ABLATION_REPORT_INTEGRATOR.replace("_", "-"),
            "description": "Ablation report integrator. Use only for composing the final ablation_results payload from validator-backed evidence.",
            "model": role_models["integrator"],
            "tools": reviewer_tools,
            "skills": integrator_skills,
            "prompt": ablation_report_integrator_prompt(),
        },
    ]


def _extract_mcp_servers(global_client_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract mcpServers from the global Claude Code client config."""
    return dict(global_client_cfg.get("mcpServers", {}) or {})


def build_experiment_mcp_config(
    *,
    workspace_cfg: Dict[str, Any],
    external_cfg: Dict[str, Any],
    global_client_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    _ = external_cfg
    mcp_servers = _extract_mcp_servers(global_client_cfg or {})

    # Add workspace-specific Tavily server if enabled
    tavily_enabled = bool(workspace_cfg.get("tavily_enabled"))
    tavily_api_key = str(workspace_cfg.get("tavily_api_key") or "").strip()
    if tavily_enabled and tavily_api_key:
        template = str(workspace_cfg.get("tavily_remote_url_template") or "").strip()
        if template:
            mcp_servers["tavily"] = {
                "type": "http",
                "url": template.format(api_key=tavily_api_key),
            }
    return {"mcpServers": mcp_servers}


def materialize_workspace_claude_project(
    *,
    workspace_dir: str,
    role_models: Dict[str, str],
    workspace_cfg: Dict[str, Any],
    external_cfg: Dict[str, Any],
    global_settings_path: str = "",
    global_client_config_path: str = "",
) -> Dict[str, str]:
    workspace_root = Path(workspace_dir)
    claude_dir = workspace_root / ".claude"
    agents_dir = claude_dir / "agents"
    skills_dir = claude_dir / "skills"
    commands_dir = claude_dir / "commands"

    claude_dir.mkdir(parents=True, exist_ok=True)
    _reset_dir(agents_dir)
    _reset_dir(skills_dir)
    _reset_dir(commands_dir)

    global_settings = _load_json_file(global_settings_path)
    _global_client_cfg = _load_json_file(global_client_config_path)
    settings_payload = _merge_dict(
        global_settings,
        {
        "model": role_models.get("planner", "opus"),
        "permissionMode": "bypassPermissions",
        },
    )
    _write_text(claude_dir / "settings.json", json.dumps(settings_payload, ensure_ascii=False, indent=2))
    _write_text(workspace_root / "CLAUDE.md", _workspace_claude_md())

    src_skills_dir = Path(__file__).resolve().parents[1] / "skills"
    if src_skills_dir.is_dir():
        for skill_dir in sorted(src_skills_dir.iterdir()):
            if not (skill_dir / "SKILL.md").is_file():
                continue
            shutil.copytree(skill_dir, skills_dir / skill_dir.name)

    mcp_payload = build_experiment_mcp_config(
        workspace_cfg=workspace_cfg,
        external_cfg=external_cfg,
        global_client_cfg=_global_client_cfg,
    )
    _write_text(workspace_root / ".mcp.json", json.dumps(mcp_payload, ensure_ascii=False, indent=2))
    enabled_mcp_servers = sorted(mcp_payload.get("mcpServers", {}).keys())

    for spec in _agent_specs(role_models, enabled_mcp_servers):
        _write_text(agents_dir / f"{spec['name']}.md", _agent_markdown(spec))

    command_specs = {
        "prepare-planner": _command_body("prepare-planner", "draft the prepare stage plan"),
        "code-planner": _command_body("experiment-code-planner", "draft the code enablement plan"),
        "standard-science-planner": _command_body("experiment-standard-science-planner", "draft the standard science plan"),
        "ablation-science-planner": _command_body("experiment-ablation-science-planner", "draft the ablation science plan"),
        "ablation-report-integrator": _command_body("experiment-ablation-report-integrator", "compose the final ablation results payload"),
    }
    for name, body in command_specs.items():
        _write_text(commands_dir / f"{name}.md", _command_markdown(name, body))

    return {
        "claude_dir": str(claude_dir),
        "settings_path": str(claude_dir / "settings.json"),
        "agents_dir": str(agents_dir),
        "skills_dir": str(skills_dir),
        "commands_dir": str(commands_dir),
        "mcp_config_path": str(workspace_root / ".mcp.json"),
        "claude_md_path": str(workspace_root / "CLAUDE.md"),
    }
