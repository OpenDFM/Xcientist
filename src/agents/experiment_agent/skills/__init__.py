"""
Curated skill loading utilities for Experiment Agent.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Set

from openhands.sdk.context import AgentContext
from openhands.sdk.context.skills import load_project_skills, load_skills_from_dir


SKILLS_DIR = Path(__file__).parent

PREPARE_SKILLS = {
    "prepare-planning",
    "resource-acquisition",
    "benchmark-discovery",
    "component-coverage",
}
CODE_SKILLS = {
    "code-planning",
    "code-enablement",
}
SCIENCE_SKILLS = {
    "science-planning",
    "science-execution",
    "component-coverage",
}
MASTER_SKILLS = {
    "convergence-gate",
}
REPORTING_SKILLS = {
    "component-coverage",
    "science-execution",
    "convergence-gate",
}
ITERATION_REPORTER_SKILLS = {
    "iteration-integration",
}
WORKER_SKILL_MAP = {
    "prepare_worker": {
        "resource-acquisition",
        "environment-setup",
        "benchmark-discovery",
        "component-coverage",
    },
    "prepare_validator": {
        "resource-acquisition",
        "environment-setup",
        "benchmark-discovery",
        "component-coverage",
    },
    "prepare_step_executor": {
        "prepare-planning",
        "resource-acquisition",
        "environment-setup",
        "benchmark-discovery",
        "component-coverage",
    },
    "code_worker": {
        "code-planning",
        "code-enablement",
    },
    "code_validator": {
        "code-planning",
        "code-enablement",
    },
    "code_step_executor": {
        "code-planning",
        "code-enablement",
    },
    "standard_science_worker": {
        "science-planning",
        "science-execution",
        "component-coverage",
    },
    "standard_science_validator": {
        "science-planning",
        "science-execution",
        "component-coverage",
    },
    "standard_science_step_executor": {
        "science-planning",
        "science-execution",
        "component-coverage",
    },
    "ablation_science_worker": {
        "science-planning",
        "science-execution",
        "component-coverage",
    },
    "ablation_science_validator": {
        "science-planning",
        "science-execution",
        "component-coverage",
    },
    "ablation_science_step_executor": {
        "science-planning",
        "science-execution",
        "component-coverage",
    },
    "ablation_report_integrator": {
        "component-coverage",
        "science-execution",
        "convergence-gate",
    },
    "iteration_reporter": {
        "iteration-integration",
    },
}


def get_skills_dir() -> Path:
    return SKILLS_DIR


def _discover_project_skill_root() -> Optional[Path]:
    candidates = [Path.cwd().resolve(), *SKILLS_DIR.resolve().parents]
    for candidate in candidates:
        if (candidate / "AGENTS.md").is_file():
            return candidate
    return None


def _load_skill_map():
    repo_skills, knowledge_skills, agent_skills = load_skills_from_dir(str(SKILLS_DIR))
    all_skills = {}
    for source in (repo_skills, knowledge_skills, agent_skills):
        all_skills.update(source)
    return all_skills


def _load_project_skill_list() -> List:
    project_root = _discover_project_skill_root()
    if project_root is None:
        return []
    try:
        return list(load_project_skills(project_root))
    except Exception:
        return []


def _merge_skill_lists(*skill_groups: Iterable) -> List:
    merged: List = []
    seen_names: Set[str] = set()
    for group in skill_groups:
        for skill in group:
            skill_name = getattr(skill, "name", None) or f"anonymous_{len(merged)}"
            if skill_name in seen_names:
                continue
            seen_names.add(skill_name)
            merged.append(skill)
    return merged


def load_skills(skill_names: Optional[Iterable[str]] = None) -> AgentContext:
    skill_map = _load_skill_map()
    requested: Set[str] = set(skill_names or skill_map.keys())
    selected = [skill for name, skill in skill_map.items() if name in requested]
    project_skills = _load_project_skill_list()
    return AgentContext(
        skills=_merge_skill_lists(project_skills, selected),
        load_public_skills=False,
    )


def load_all_skills() -> AgentContext:
    return load_skills()


def get_prepare_agent_context() -> AgentContext:
    return load_skills(PREPARE_SKILLS)


def get_code_agent_context() -> AgentContext:
    return load_skills(CODE_SKILLS)


def get_exp_agent_context() -> AgentContext:
    return load_skills(SCIENCE_SKILLS)


def get_master_agent_context() -> AgentContext:
    return load_skills(MASTER_SKILLS)


def get_worker_agent_context(role: str) -> AgentContext:
    return load_skills(WORKER_SKILL_MAP.get(role, set()))


def print_loaded_skills():
    skill_map = _load_skill_map()
    print("Loaded skills:")
    print(f"  All skills: {list(skill_map.keys())}")
    print(f"  Prepare skills: {sorted(PREPARE_SKILLS)}")
    print(f"  Code skills: {sorted(CODE_SKILLS)}")
    print(f"  Science skills: {sorted(SCIENCE_SKILLS)}")
    print(f"  Master skills: {sorted(MASTER_SKILLS)}")
    print(f"  Worker skill roles: {sorted(WORKER_SKILL_MAP.keys())}")
