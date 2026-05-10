"""
Backward-compatible helpers for role metadata.
"""

from __future__ import annotations

from typing import Iterable, List


DEFAULT_INCLUDE_DEFAULT_TOOLS: tuple[str, ...] = ("Read", "Edit", "Bash")


def default_builtin_tool_names() -> list[str]:
    return list(DEFAULT_INCLUDE_DEFAULT_TOOLS)


def build_tool_list(tool_names: Iterable[str]) -> List[str]:
    unique_names: List[str] = []
    for name in tool_names:
        if name not in unique_names:
            unique_names.append(name)
    return unique_names


def create_phase_subagent(
    llm,
    *,
    role: str,
    tool_names: Iterable[str],
    system_prompt: str,
    extra_skills=None,
    mcp_servers=None,
    workspace_root=None,
):
    _ = llm, extra_skills, mcp_servers, workspace_root
    return {
        "role": role,
        "tool_names": build_tool_list(tool_names),
        "system_prompt": system_prompt,
    }
