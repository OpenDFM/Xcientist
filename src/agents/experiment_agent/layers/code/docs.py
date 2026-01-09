"""
Code layer document management (spec-coding artifacts).

Goals:
- Mirror the Science layer's `docs.py` pattern.
- Keep existing workspace paths (workspace_root/specs/) intact for compatibility.
- Store source-of-truth spec-coding artifacts under cache_root/code/ for robust resume:
  - idea.md, spec.md, plan.md (stable, but may evolve under --fresh)
- Provide deterministic absolute paths for agent prompts and orchestration.
"""

import os
import shutil
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class CodeDocPaths:
    workspace_root: str
    cache_root: str

    def code_cache_dir(self) -> str:
        return os.path.join(self.cache_root, "code")

    def ensure_dirs(self) -> None:
        os.makedirs(self.workspace_root, exist_ok=True)
        os.makedirs(self.cache_root, exist_ok=True)
        os.makedirs(self.code_cache_dir(), exist_ok=True)
        os.makedirs(self.specs_dir(), exist_ok=True)
        os.makedirs(self.templates_dir(), exist_ok=True)

    def specs_dir(self) -> str:
        return os.path.join(self.workspace_root, "specs")

    def templates_dir(self) -> str:
        return os.path.join(self.workspace_root, "templates")

    def constitution_path(self) -> str:
        return os.path.join(self.cache_root, "constitution.md")

    def idea_md(self) -> str:
        return os.path.join(self.code_cache_dir(), "idea.md")

    def spec_md(self) -> str:
        return os.path.join(self.code_cache_dir(), "spec.md")

    def plan_md(self) -> str:
        return os.path.join(self.code_cache_dir(), "plan.md")

    def workspace_spec_path(self) -> str:
        return os.path.join(self.specs_dir(), "spec.md")

    def workspace_plan_path(self) -> str:
        return os.path.join(self.specs_dir(), "plan.md")

    def to_prompt_dict(self) -> Dict[str, str]:
        return {
            "idea_path": self.idea_md(),
            "spec_path": self.spec_md(),
            "plan_path": self.plan_md(),
            "constitution_path": self.constitution_path(),
            "templates_dir": self.templates_dir(),
            "specs_dir": self.specs_dir(),
        }


def build_code_doc_paths(
    workspace_root: str,
    cache_root: str,
    ensure: bool = True,
) -> CodeDocPaths:
    p = CodeDocPaths(workspace_root=str(workspace_root), cache_root=str(cache_root))
    if ensure:
        p.ensure_dirs()
    return p


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
    doc_paths: CodeDocPaths,
) -> str:
    """
    Ensure cache/code/idea.md exists.
    """
    doc_paths.ensure_dirs()
    dst = doc_paths.idea_md()

    if idea_input_path and os.path.exists(idea_input_path):
        if idea_input_path.endswith(".md"):
            _safe_copy(idea_input_path, dst)
            return dst
        try:
            with open(idea_input_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception:
            raw = ""
        content = (
            "# Idea (from idea.json)\n\n```json\n" + (raw or "").strip() + "\n```\n"
        )
        with open(dst, "w", encoding="utf-8") as f:
            f.write(content)
        return dst

    with open(dst, "w", encoding="utf-8") as f:
        f.write((proposal_text or "").strip() + "\n")
    return dst


def sync_code_docs_to_specs(doc_paths: CodeDocPaths) -> None:
    """
    Mirror cache/code/{spec,plan}.md into workspace_root/specs/ for compatibility.
    """
    doc_paths.ensure_dirs()
    _safe_copy(doc_paths.spec_md(), doc_paths.workspace_spec_path())
    _safe_copy(doc_paths.plan_md(), doc_paths.workspace_plan_path())
