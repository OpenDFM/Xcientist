"""Deterministic project-code integrity checks for the code phase.

These checks are intentionally formal and conservative. They do not try to
judge whether the research idea is good; they only block project states that
make the later science phase non-reproducible or ambiguous.
"""

from __future__ import annotations

import ast
import os
import py_compile
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


CANONICAL_CWD = "workspace_root"
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "env",
    "venv",
    ".venv",
    "node_modules",
}
FORBIDDEN_DIRS = {
    ".scratch",
    "scratch",
    "tmp",
    "temp",
    "backup",
    "backups",
    "old",
}
EXPERIMENTAL_FILE_RE = re.compile(
    r"(^|[_\-.])(patched|repaired|backup|bak|old|tmp|temp|copy|fixed|final|vectorized|draft)([_\-.]|$)",
    re.IGNORECASE,
)
RUNNER_NAMES = {"run.py", "run_condition.py", "smoke_integration.py"}
RESOURCE_DEFAULT_FLAGS = {"--data", "--adjdata", "--topk-indices"}


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def _iter_files(project_root: Path) -> Iterable[Path]:
    for current_root, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in IGNORED_DIRS and name not in FORBIDDEN_DIRS
        ]
        current = Path(current_root)
        for name in filenames:
            yield current / name


def _issue(
    issues: List[Dict[str, Any]],
    *,
    rule: str,
    message: str,
    path: str = "",
    line: int | None = None,
    fix: str = "",
) -> None:
    payload: Dict[str, Any] = {
        "rule": rule,
        "message": message,
    }
    if path:
        payload["path"] = path
    if line is not None:
        payload["line"] = int(line)
    if fix:
        payload["fix"] = fix
    issues.append(payload)


def _declared_project_paths(plan_steps: Sequence[Mapping[str, Any]]) -> Set[str]:
    declared: Set[str] = set()
    for step in plan_steps:
        for path in step.get("project_target_paths") or []:
            if isinstance(path, str) and path.strip():
                declared.add(_normalize_project_rel(path))
        for item in step.get("code_artifacts") or []:
            if isinstance(item, Mapping):
                path = item.get("path")
                if isinstance(path, str) and path.strip():
                    declared.add(_normalize_project_rel(path))
        interface_contract = step.get("interface_contract")
        if isinstance(interface_contract, Mapping):
            for value in interface_contract.values():
                if isinstance(value, str):
                    for match in re.findall(r"\bproject/[^\s`'\";|&]+", value):
                        declared.add(_normalize_project_rel(match))
    return declared


def _normalize_project_rel(path: str) -> str:
    value = str(path).strip().replace("\\", "/")
    if value == "project":
        return ""
    if value.startswith("project/"):
        return value[len("project/") :]
    return value.lstrip("/")


def _check_project_tree(
    *,
    project_root: Path,
    declared_paths: Set[str],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    files: List[str] = []
    forbidden_dirs: List[str] = []
    for current_root, dirnames, _filenames in os.walk(project_root):
        rel_current = _rel(Path(current_root), project_root)
        for dirname in list(dirnames):
            if dirname in FORBIDDEN_DIRS:
                rel_dir = dirname if rel_current == "." else f"{rel_current}/{dirname}"
                forbidden_dirs.append(rel_dir)
                _issue(
                    issues,
                    rule="forbidden_project_directory",
                    path=rel_dir,
                    message=f"Project contains forbidden transient directory `{rel_dir}`.",
                    fix="Remove transient exploration files from project/ before finishing the code phase.",
                )
        dirnames[:] = [
            name
            for name in dirnames
            if name not in IGNORED_DIRS and name not in FORBIDDEN_DIRS
        ]

    for path in _iter_files(project_root):
        rel_path = _rel(path, project_root)
        files.append(rel_path)
        if path.suffix == ".py" and EXPERIMENTAL_FILE_RE.search(path.stem):
            declared = rel_path in declared_paths or f"project/{rel_path}" in declared_paths
            if not declared:
                _issue(
                    issues,
                    rule="experimental_variant_file",
                    path=rel_path,
                    message=f"Python file `{rel_path}` looks like an intermediate/variant implementation and is not declared as a final project target.",
                    fix="Keep only the canonical implementation file or declare this file as a real target with a clear entrypoint role.",
                )
    return {
        "checked_files": len(files),
        "forbidden_dirs": forbidden_dirs,
        "declared_project_paths": sorted(declared_paths),
    }


def _check_python_compiles(*, project_root: Path, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    checked = 0
    for path in _iter_files(project_root):
        if path.suffix != ".py":
            continue
        checked += 1
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            _issue(
                issues,
                rule="python_compile_error",
                path=_rel(path, project_root),
                message=str(exc.msg),
                fix="Fix the syntax error before finishing the code phase.",
            )
    return {"checked_python_files": checked}


def _command_mentions_cd_project(command: str) -> bool:
    return bool(re.search(r"(^|[;&|]\s*)cd\s+project(\s|[;&|]|$)", command))


def _check_plan_commands(
    *,
    plan_steps: Sequence[Mapping[str, Any]],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    checked: List[Dict[str, str]] = []
    for step in plan_steps:
        step_id = str(step.get("step_id") or step.get("stage_id") or "")
        for field in ("verify_command", "command"):
            command = step.get(field)
            if not isinstance(command, str) or not command.strip():
                continue
            checked.append({"step_id": step_id, "field": field, "command": command})
            if _command_mentions_cd_project(command):
                _issue(
                    issues,
                    rule="noncanonical_cwd_command",
                    path=f"plan:{step_id}.{field}",
                    message=f"`{field}` uses `cd project`, but code/science commands must run from the workspace root.",
                    fix="Rewrite commands to use workspace-root paths such as `project/.venv/bin/python project/run.py --condition <condition_id> ...`.",
                )
    return {"canonical_cwd": CANONICAL_CWD, "checked_commands": checked}


def _extract_argparse_defaults(path: Path) -> Dict[str, str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    defaults: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        flags = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        relevant = [flag for flag in flags if flag in RESOURCE_DEFAULT_FLAGS]
        if not relevant:
            continue
        default_value = None
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    default_value = keyword.value.value
        if default_value:
            for flag in relevant:
                defaults[flag] = default_value
    return defaults


def _check_runner_resource_defaults(
    *,
    workspace_root: Path,
    project_root: Path,
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    checked: Dict[str, Dict[str, str]] = {}
    for runner_name in RUNNER_NAMES:
        runner = project_root / runner_name
        if not runner.exists():
            continue
        defaults = _extract_argparse_defaults(runner)
        checked[runner_name] = defaults
        for flag, default_path in defaults.items():
            if os.path.isabs(default_path):
                resolved = Path(default_path)
            else:
                resolved = workspace_root / default_path
            if not resolved.exists():
                _issue(
                    issues,
                    rule="runner_default_resource_missing",
                    path=f"{runner_name}:{flag}",
                    message=f"Default `{flag}` path `{default_path}` does not exist from the canonical workspace-root cwd.",
                    fix="Make runner defaults resolve from the workspace root, or require the command to pass an explicit valid path.",
                )
    return {"checked_runner_defaults": checked}


def _check_silent_resource_fallbacks(
    *,
    project_root: Path,
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    checked = 0
    for path in _iter_files(project_root):
        if path.suffix != ".py":
            continue
        checked += 1
        rel_path = _rel(path, project_root)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            normalized = line.replace(" ", "")
            lowered = line.lower()
            if "topk" in lowered and "exists(" in normalized:
                window = "\n".join(lines[idx - 1 : min(len(lines), idx + 8)])
                if "raise " not in window and "sys.exit" not in window:
                    _issue(
                        issues,
                        rule="silent_required_resource_fallback",
                        path=rel_path,
                        line=idx,
                        message="Required top-k resource existence check can silently continue when the file is missing.",
                        fix="Fail fast with a clear exception when required top-k indices are missing.",
                    )
            if "A_k_indices" in line and "torch.zeros" in line:
                _issue(
                    issues,
                    rule="placeholder_topk_indices",
                    path=rel_path,
                    line=idx,
                    message="Sparse top-k indices fall back to an all-zero placeholder.",
                    fix="Require explicit top-k indices for sparse attention or raise before model construction.",
                )
    return {"checked_fallback_files": checked}


def audit_code_project_integrity(
    *,
    workspace_root: str,
    project_root: str,
    plan_steps: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Run deterministic code-phase project integrity checks."""
    workspace_dir = Path(os.path.realpath(workspace_root))
    project_dir = Path(os.path.realpath(project_root))
    issues: List[Dict[str, Any]] = []

    if not project_dir.exists():
        _issue(
            issues,
            rule="missing_project_dir",
            path=str(project_dir),
            message="Project directory does not exist.",
            fix="Create project/ and place the runnable experiment code there.",
        )
        return {
            "status": "FAIL",
            "hook": "code_project_integrity",
            "issues": issues,
            "checks": {},
            "project_root": str(project_dir),
            "canonical_cwd": CANONICAL_CWD,
        }

    declared_paths = _declared_project_paths(plan_steps)
    checks = {
        "project_tree": _check_project_tree(
            project_root=project_dir,
            declared_paths=declared_paths,
            issues=issues,
        ),
        "python_compile": _check_python_compiles(project_root=project_dir, issues=issues),
        "plan_commands": _check_plan_commands(plan_steps=plan_steps, issues=issues),
        "runner_resource_defaults": _check_runner_resource_defaults(
            workspace_root=workspace_dir,
            project_root=project_dir,
            issues=issues,
        ),
        "silent_resource_fallbacks": _check_silent_resource_fallbacks(
            project_root=project_dir,
            issues=issues,
        ),
    }
    return {
        "status": "PASS" if not issues else "FAIL",
        "hook": "code_project_integrity",
        "issues": issues,
        "checks": checks,
        "project_root": str(project_dir),
        "canonical_cwd": CANONICAL_CWD,
    }


def format_project_integrity_feedback(payload: Mapping[str, Any]) -> str:
    issues = payload.get("issues") if isinstance(payload.get("issues"), list) else []
    lines = [
        "Xcientist formal code-project integrity hook blocked worker completion.",
        "",
        "The non-formal reviewer was not called because deterministic checks failed first.",
        "",
        "Issues:",
    ]
    if not issues:
        lines.append("- Unknown project integrity failure.")
        return "\n".join(lines)
    for item in issues:
        if not isinstance(item, Mapping):
            lines.append(f"- {item}")
            continue
        location = str(item.get("path") or "").strip()
        if item.get("line") is not None:
            location = f"{location}:{item.get('line')}" if location else f"line {item.get('line')}"
        prefix = f"[{item.get('rule')}]"
        message = str(item.get("message") or "").strip()
        fix = str(item.get("fix") or "").strip()
        line = f"- {prefix} {location} {message}".strip()
        if fix:
            line += f" Fix: {fix}"
        lines.append(line)
    lines.extend(
        [
            "",
            "Fix these issues in the same worker session, update managed artifacts through artifact tools, then finish again.",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "CANONICAL_CWD",
    "audit_code_project_integrity",
    "format_project_integrity_feedback",
]
