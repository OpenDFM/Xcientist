"""
Tools package for experiment agents.

This package provides all tools needed by the experiment agent system,
optimized for minimal toolset strategy.
"""

from typing import List, Dict, Any

# Import all tools from modules
from src.agents.experiment_agent.tools.file_tools import (
    read_file,
    write_file,
    edit_file,
    list_directory,
)

from src.agents.experiment_agent.tools.execution_tools import (
    run_shell_command,
)

from src.agents.experiment_agent.tools.document_tools import (
    parse_latex_sections,
    extract_latex_equations,
    parse_json_file,
    extract_code_blocks,
    summarize_document,
    extract_urls,
    parse_requirements_txt,
    extract_key_terms,
)

from src.agents.experiment_agent.tools.code_analysis_tools import (
    analyze_python_file,
)

from src.agents.experiment_agent.tools.repository_tools import (
    list_papers_in_directory,
    generate_code_tree,
    analyze_repository_structure,
    get_repository_overview,
    read_pdf_paper,
)


# Tool categories for different agents
FILE_TOOLS = [
    read_file,
    write_file,
    edit_file,
    list_directory,
]

# Read-only file tools for planning/analysis phases
FILE_TOOLS_READONLY = [
    read_file,
    list_directory,
]

EXECUTION_TOOLS = [
    run_shell_command,
]

DOCUMENT_TOOLS = [
    parse_latex_sections,
    extract_latex_equations,
    parse_json_file,
    extract_code_blocks,
    summarize_document,
    extract_urls,
    parse_requirements_txt,
    extract_key_terms,
]

CODE_ANALYSIS_TOOLS = [
    analyze_python_file,
]

REPOSITORY_TOOLS = [
    list_papers_in_directory,
    generate_code_tree,
    analyze_repository_structure,
    get_repository_overview,
    read_pdf_paper,
]

# All tools combined
ALL_TOOLS = (
    FILE_TOOLS
    + EXECUTION_TOOLS
    + DOCUMENT_TOOLS
    + CODE_ANALYSIS_TOOLS
    + REPOSITORY_TOOLS
)


def get_tools_for_agent(agent_type: str) -> List:
    """
    Get recommended tools for a specific agent type.

    Args:
        agent_type: Type of agent (pre_analysis, code_plan, code_implement, etc.)

    Returns:
        List of tool functions for the agent
    """
    # Base tools for all agents (Shell/Read)
    # Actually, only "execution" agents should get run_shell_command in write mode?
    # No, per discussion:
    # - Plan/Analysis: Read-only (read_file, list_dir) + maybe shell for grep?
    # - Implement/Execute: Full Shell + Write

    # However, strict read-only agents shouldn't have run_shell_command unless we trust them
    # to only use grep. The user said "don't use run_shell_command for plan agents".

    tool_configs = {
        "pre_analysis": {
            "paper": DOCUMENT_TOOLS + FILE_TOOLS_READONLY + REPOSITORY_TOOLS,
            "idea": DOCUMENT_TOOLS + FILE_TOOLS_READONLY + REPOSITORY_TOOLS,
        },
        "code_plan": {
            "initial": FILE_TOOLS_READONLY + CODE_ANALYSIS_TOOLS + REPOSITORY_TOOLS,
            "judge_feedback": FILE_TOOLS_READONLY
            + CODE_ANALYSIS_TOOLS
            + REPOSITORY_TOOLS,
            "error_feedback": FILE_TOOLS_READONLY
            + CODE_ANALYSIS_TOOLS
            + REPOSITORY_TOOLS,
            "analysis_feedback": FILE_TOOLS_READONLY
            + CODE_ANALYSIS_TOOLS
            + REPOSITORY_TOOLS,
        },
        "code_implement": FILE_TOOLS + EXECUTION_TOOLS + CODE_ANALYSIS_TOOLS,
        "code_judge": FILE_TOOLS + EXECUTION_TOOLS + CODE_ANALYSIS_TOOLS,
        "experiment_execute": FILE_TOOLS_READONLY
        + EXECUTION_TOOLS,  # Execute needs to run scripts
        "experiment_analysis": FILE_TOOLS_READONLY
        + DOCUMENT_TOOLS
        + CODE_ANALYSIS_TOOLS
        + EXECUTION_TOOLS,
    }

    return tool_configs.get(agent_type, [])


def get_all_tool_names() -> List[str]:
    """
    Get names of all available tools.

    Returns:
        List of tool names
    """
    return [tool.__name__ for tool in ALL_TOOLS]


def get_tool_by_name(tool_name: str):
    """
    Get a tool function by its name.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool function or None if not found
    """
    for tool in ALL_TOOLS:
        if tool.__name__ == tool_name:
            return tool
    return None


__all__ = [
    "read_file",
    "write_file",
    "edit_file",
    "list_directory",
    "run_shell_command",
    # ... document/repo tools ...
    "parse_latex_sections",
    "extract_latex_equations",
    "parse_json_file",
    "extract_code_blocks",
    "summarize_document",
    "extract_urls",
    "parse_requirements_txt",
    "extract_key_terms",
    "analyze_python_file",
    "list_papers_in_directory",
    "generate_code_tree",
    "analyze_repository_structure",
    "get_repository_overview",
    "read_pdf_paper",
    # Collections
    "FILE_TOOLS",
    "FILE_TOOLS_READONLY",
    "EXECUTION_TOOLS",
    "DOCUMENT_TOOLS",
    "CODE_ANALYSIS_TOOLS",
    "REPOSITORY_TOOLS",
    "ALL_TOOLS",
    # Helpers
    "get_tools_for_agent",
    "get_all_tool_names",
    "get_tool_by_name",
]
