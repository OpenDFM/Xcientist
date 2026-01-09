"""
Science layer document management.

Goals:
- Keep existing workspace paths (e.g., workspace_root/specs/) intact for compatibility.
- Store source-of-truth spec-coding artifacts under cache_root/science/ for robust resume:
  - idea.md, spec.md (stable), plan_v###.md, tasks_v###.md, report_v###.md, feedback_v###.md
- Provide deterministic absolute paths for agent prompts and orchestration.
"""

import os
import shutil
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ScienceDocPaths:
    workspace_root: str
    cache_root: str

    def science_cache_dir(self) -> str:
        return os.path.join(self.cache_root, "science")

    def ensure_dirs(self) -> None:
        os.makedirs(self.science_cache_dir(), exist_ok=True)

    def idea_md(self) -> str:
        return os.path.join(self.science_cache_dir(), "idea.md")

    def spec_md(self) -> str:
        return os.path.join(self.science_cache_dir(), "spec.md")

    def plan_md(self, version: int) -> str:
        return os.path.join(self.science_cache_dir(), f"plan_v{int(version):03d}.md")

    def tasks_md(self, version: int) -> str:
        return os.path.join(self.science_cache_dir(), f"tasks_v{int(version):03d}.md")

    def report_md(self, version: int) -> str:
        return os.path.join(self.science_cache_dir(), f"report_v{int(version):03d}.md")

    def feedback_md(self, version: int) -> str:
        return os.path.join(
            self.science_cache_dir(), f"feedback_v{int(version):03d}.md"
        )

    def latest_aliases(self) -> Dict[str, str]:
        """
        Convenience stable filenames in cache/science/ pointing to the latest docs.
        These are optional but make manual inspection easier.
        """
        return {
            "plan": os.path.join(self.science_cache_dir(), "plan.md"),
            "tasks": os.path.join(self.science_cache_dir(), "tasks.md"),
            "report": os.path.join(self.science_cache_dir(), "report.md"),
            "feedback": os.path.join(self.science_cache_dir(), "feedback.md"),
        }

    def workspace_specs_dir(self) -> str:
        return os.path.join(self.workspace_root, "specs")

    def ensure_workspace_specs_dir(self) -> None:
        os.makedirs(self.workspace_specs_dir(), exist_ok=True)

    def workspace_specs_paths(self) -> Dict[str, str]:
        """
        Keep the legacy/visible locations stable for compatibility and ergonomics.
        """
        return {
            "spec": os.path.join(self.workspace_specs_dir(), "spec.md"),
            "plan": os.path.join(self.workspace_specs_dir(), "plan.md"),
            "tasks": os.path.join(self.workspace_specs_dir(), "tasks.md"),
            "report": os.path.join(self.workspace_specs_dir(), "report.md"),
            "feedback": os.path.join(self.workspace_specs_dir(), "feedback.md"),
        }


def _safe_copy(src: str, dst: str) -> None:
    if not src or not dst:
        return
    if not os.path.exists(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def snapshot_idea_to_cache(
    idea_input_path: Optional[str],
    proposal_text: Optional[str],
    doc_paths: ScienceDocPaths,
) -> str:
    """
    Ensure cache/science/idea.md exists.

    Priority:
    1) Copy from idea_input_path if it exists (idea.md or idea.json upstream)
    2) Fall back to proposal_text (string) written into idea.md
    """
    doc_paths.ensure_dirs()
    dst = doc_paths.idea_md()

    if idea_input_path and os.path.exists(idea_input_path):
        # Normalize to markdown for downstream prompts.
        if idea_input_path.endswith(".md"):
            _safe_copy(idea_input_path, dst)
            return dst
        # If JSON, keep it but also write a minimal markdown wrapper.
        try:
            with open(idea_input_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception:
            raw = ""
        content = (
            "# Idea (from idea.json)\n\n" "```json\n" + (raw or "").strip() + "\n```\n"
        )
        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)
        return dst

    # Proposal text fallback
    if not proposal_text:
        proposal_text = ""
    with open(dst, "w", encoding="utf-8") as f:
        f.write((proposal_text or "").strip() + "\n")
    return dst


def sync_science_docs_to_specs(
    doc_paths: ScienceDocPaths,
    version: int,
    write_latest_aliases: bool = True,
) -> None:
    """
    Mirror the canonical cache/science docs into workspace_root/specs/ for compatibility.
    This does NOT change any paths; it only copies content to the legacy visible locations.
    """
    doc_paths.ensure_dirs()
    doc_paths.ensure_workspace_specs_dir()

    cache_spec = doc_paths.spec_md()
    cache_plan = doc_paths.plan_md(version)
    cache_tasks = doc_paths.tasks_md(version)
    cache_report = doc_paths.report_md(version)
    cache_feedback = doc_paths.feedback_md(version)

    specs = doc_paths.workspace_specs_paths()
    _safe_copy(cache_spec, specs["spec"])
    _safe_copy(cache_plan, specs["plan"])
    _safe_copy(cache_tasks, specs["tasks"])
    _safe_copy(cache_report, specs["report"])
    _safe_copy(cache_feedback, specs["feedback"])

    if write_latest_aliases:
        aliases = doc_paths.latest_aliases()
        _safe_copy(cache_plan, aliases["plan"])
        _safe_copy(cache_tasks, aliases["tasks"])
        _safe_copy(cache_report, aliases["report"])
        _safe_copy(cache_feedback, aliases["feedback"])
