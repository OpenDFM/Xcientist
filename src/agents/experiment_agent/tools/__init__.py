"""
Tools package for experiment agents.

This package provides all tools needed by the experiment agent system,
organized into categories and compatible with openai-agents SDK.
"""

from typing import List, Dict, Any

# Import all tools from modules
from src.agents.experiment_agent.tools.file_tools import (
    read_file,
    write_file,
    list_directory,
    create_directory,
    delete_file,
    copy_file,
    file_exists,
    get_file_info,
)

from src.agents.experiment_agent.tools.execution_tools import (
    run_python_script,
    run_shell_command,
    run_python_code,
    install_package,
    check_python_syntax,
    get_environment_info,
    list_installed_packages,
    create_log_file,
    append_to_log,
    # Docker execution tools
    run_in_docker,
    run_python_in_docker,
    test_docker_connection,
    set_docker_client,
    get_docker_client,
    # Local execution tools (no Docker required)
    run_pytest_local,
    run_python_script_local,
    run_python_code_local,
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
    search_in_codebase,
    count_lines_of_code,
    extract_function_code,
    list_python_files,
    check_imports_available,
    get_file_dependencies,
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
    list_directory,
    create_directory,
    delete_file,
    copy_file,
    file_exists,
    get_file_info,
]

# Read-only file tools for planning/analysis phases
FILE_TOOLS_READONLY = [
    read_file,
    list_directory,
    file_exists,
    get_file_info,
]

EXECUTION_TOOLS = [
    run_python_script,
    run_shell_command,
    run_python_code,
    install_package,
    check_python_syntax,
    get_environment_info,
    list_installed_packages,
    create_log_file,
    append_to_log,
]

# Docker execution tools (separate category for flexibility)
DOCKER_TOOLS = [
    run_in_docker,
    run_python_in_docker,
    test_docker_connection,
]

# Local execution tools (no Docker required)
LOCAL_EXECUTION_TOOLS = [
    run_pytest_local,
    run_python_script_local,
    run_python_code_local,
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
    search_in_codebase,
    count_lines_of_code,
    extract_function_code,
    list_python_files,
    check_imports_available,
    get_file_dependencies,
]

REPOSITORY_TOOLS = [
    list_papers_in_directory,
    generate_code_tree,
    analyze_repository_structure,
    get_repository_overview,
    read_pdf_paper,
]

# All tools combined (Docker tools separate for flexibility - can be added when needed)
ALL_TOOLS = (
    FILE_TOOLS
    + EXECUTION_TOOLS
    + DOCUMENT_TOOLS
    + CODE_ANALYSIS_TOOLS
    + REPOSITORY_TOOLS
)
ALL_TOOLS_WITH_DOCKER = ALL_TOOLS + DOCKER_TOOLS


def get_tools_for_agent(agent_type: str) -> List:
    """
    Get recommended tools for a specific agent type.

    Args:
        agent_type: Type of agent (pre_analysis, code_plan, code_implement, etc.)

    Returns:
        List of tool functions for the agent
    """
    tool_configs = {
        "pre_analysis": {
            "paper": DOCUMENT_TOOLS
            + FILE_TOOLS[:3]
            + REPOSITORY_TOOLS,  # read, write, list + repo tools
            "idea": DOCUMENT_TOOLS + FILE_TOOLS[:3] + REPOSITORY_TOOLS,
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
        "code_implement": FILE_TOOLS + EXECUTION_TOOLS[:6] + CODE_ANALYSIS_TOOLS,
        "code_judge": FILE_TOOLS + CODE_ANALYSIS_TOOLS + LOCAL_EXECUTION_TOOLS,
        "experiment_execute": FILE_TOOLS[:3] + EXECUTION_TOOLS,
        "experiment_analysis": FILE_TOOLS + DOCUMENT_TOOLS + CODE_ANALYSIS_TOOLS[:3],
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
    # File tools
    "read_file",
    "write_file",
    "list_directory",
    "create_directory",
    "delete_file",
    "copy_file",
    "file_exists",
    "get_file_info",
    # Execution tools
    "run_python_script",
    "run_shell_command",
    "run_python_code",
    "install_package",
    "check_python_syntax",
    "get_environment_info",
    "list_installed_packages",
    "create_log_file",
    "append_to_log",
    # Docker tools
    "run_in_docker",
    "run_python_in_docker",
    "test_docker_connection",
    "set_docker_client",
    "get_docker_client",
    # Local execution tools
    "run_pytest_local",
    "run_python_script_local",
    "run_python_code_local",
    # Document tools
    "parse_latex_sections",
    "extract_latex_equations",
    "parse_json_file",
    "extract_code_blocks",
    "summarize_document",
    "extract_urls",
    "parse_requirements_txt",
    "extract_key_terms",
    # Code analysis tools
    "analyze_python_file",
    "search_in_codebase",
    "count_lines_of_code",
    "extract_function_code",
    "list_python_files",
    "check_imports_available",
    "get_file_dependencies",
    # Repository tools
    "list_papers_in_directory",
    "generate_code_tree",
    "analyze_repository_structure",
    "get_repository_overview",
    "read_pdf_paper",
    # Tool collections
    "FILE_TOOLS",
    "FILE_TOOLS_READONLY",
    "EXECUTION_TOOLS",
    "DOCKER_TOOLS",
    "LOCAL_EXECUTION_TOOLS",
    "DOCUMENT_TOOLS",
    "CODE_ANALYSIS_TOOLS",
    "REPOSITORY_TOOLS",
    "ALL_TOOLS",
    "ALL_TOOLS_WITH_DOCKER",
    # Utility functions
    "get_tools_for_agent",
    "get_all_tool_names",
    "get_tool_by_name",
]
