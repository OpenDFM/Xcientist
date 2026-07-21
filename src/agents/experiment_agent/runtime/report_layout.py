"""Structured report layout for experiment-agent workspaces."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple


SCOPE_DIRS = {
    "prepare": "prepare",
    "code": "code",
    "science": "science",
    "ablation": "ablation",
    "runtime": "_runtime",
    "finalization": "ablation",
}


def scope_dir(scope: str) -> str:
    return SCOPE_DIRS.get(str(scope or "").strip(), slug(str(scope or "runtime")))


def slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return text or "item"


def reports_rel(*parts: str) -> str:
    clean = [str(part).strip("/").strip() for part in parts if str(part).strip("/").strip()]
    return os.path.join("agent_reports", *clean)


def reports_abs(workspace_root: str, *parts: str) -> str:
    return os.path.join(os.path.realpath(workspace_root), reports_rel(*parts))


def runtime_rel(*parts: str) -> str:
    return reports_rel("_runtime", *parts)


def phase_rel(scope: str, *parts: str) -> str:
    return reports_rel(scope_dir(scope), *parts)


def phase_abs(workspace_root: str, scope: str, *parts: str) -> str:
    return os.path.join(os.path.realpath(workspace_root), phase_rel(scope, *parts))


def planner_rel(scope: str, name: str) -> str:
    return phase_rel(scope, "plan", name)


def planner_abs(workspace_root: str, scope: str, name: str) -> str:
    return os.path.join(os.path.realpath(workspace_root), planner_rel(scope, name))


def step_report_paths(scope: str, step_id: str, attempt: int | None = None) -> Dict[str, str]:
    step = step_dir(scope, step_id)
    if attempt is None:
        return {
            "worker_report_path": phase_rel(scope, "worker", step, "latest.json"),
            "review_report_path": phase_rel(scope, "review", step, "latest.json"),
            "hook_report_path": phase_rel(scope, "hook", step, "latest.json"),
        }
    name = f"{int(attempt):03d}.json"
    return {
        "worker_report_path": phase_rel(scope, "worker", step, "attempts", name),
        "review_report_path": phase_rel(scope, "review", step, "attempts", name),
        "hook_report_path": phase_rel(scope, "hook", step, "attempts", name),
    }


def step_report_abs_paths(
    workspace_root: str,
    scope: str,
    step_id: str,
    attempt: int | None = None,
) -> Dict[str, str]:
    root = os.path.realpath(workspace_root)
    return {
        key: os.path.join(root, rel)
        for key, rel in step_report_paths(scope, step_id, attempt).items()
    }


def artifact_rel(scope: str, step_id: str, suffix: str, *, section: str | None = None) -> str:
    step = step_dir(scope, step_id)
    clean_suffix = str(suffix or "artifact").strip().lstrip("_")
    target_section = section
    if target_section is None:
        target_section = "evidence" if scope == "science" else "artifacts"
    if clean_suffix == "evidence.json":
        filename = f"{step}.json"
    elif clean_suffix == "handoff.json":
        filename = f"{step}.handoff.json"
    elif clean_suffix == "artifact.json":
        filename = f"{step}.json"
    else:
        filename = f"{step}_{clean_suffix}" if clean_suffix else step
    return phase_rel(scope, target_section, filename)


def step_dir(scope: str, step_id: str) -> str:
    step = slug(step_id)
    if scope == "science" and "::" in str(step_id):
        prefix, raw_step = str(step_id).split("::", 1)
        step = os.path.join(slug(prefix), slug(raw_step))
    return step


@dataclass(frozen=True)
class ReportLayout:
    workspace_root: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace_root", os.path.realpath(self.workspace_root))

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.workspace_root, "agent_reports")

    @property
    def runtime_dir(self) -> str:
        return os.path.join(self.reports_dir, "_runtime")

    @property
    def artifact_ledger(self) -> str:
        return os.path.join(self.runtime_dir, "artifact_ledger.jsonl")

    @property
    def artifact_registry(self) -> str:
        return os.path.join(self.runtime_dir, "artifact_registry.json")

    @property
    def run_timeline(self) -> str:
        return os.path.join(self.runtime_dir, "run_timeline.jsonl")

    @property
    def stray_outputs(self) -> str:
        return os.path.join(self.runtime_dir, "stray_outputs.json")

    def phase_dir(self, scope: str) -> str:
        return os.path.join(self.reports_dir, scope_dir(scope))

    def phase_file(self, scope: str, *parts: str) -> str:
        return os.path.join(self.workspace_root, phase_rel(scope, *parts))

    def planner_file(self, scope: str, name: str) -> str:
        return os.path.join(self.workspace_root, planner_rel(scope, name))

    def latest_and_attempt(
        self,
        scope: str,
        step_id: str,
        role: str,
        attempt: int,
    ) -> Tuple[str, str]:
        step = step_dir(scope, step_id)
        latest = self.phase_file(scope, role, step, "latest.json")
        history = self.phase_file(scope, role, step, "attempts", f"{int(attempt):03d}.json")
        return latest, history

    def review_latest_and_attempt(
        self,
        scope: str,
        step_id: str,
        reviewer_id: str,
        attempt: int,
    ) -> Tuple[str, str]:
        step = step_dir(scope, step_id)
        reviewer = slug(reviewer_id)
        latest = self.phase_file(scope, "review", step, reviewer, "latest.json")
        history = self.phase_file(scope, "review", step, reviewer, "attempts", f"{int(attempt):03d}.json")
        return latest, history


__all__ = [
    "ReportLayout",
    "artifact_rel",
    "phase_abs",
    "phase_rel",
    "planner_abs",
    "planner_rel",
    "reports_abs",
    "reports_rel",
    "runtime_rel",
    "scope_dir",
    "slug",
    "step_dir",
    "step_report_abs_paths",
    "step_report_paths",
]
