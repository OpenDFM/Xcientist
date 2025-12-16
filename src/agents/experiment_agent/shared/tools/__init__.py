"""
Shared tools for SuperAgent agents.

Core tools (agent-callable):
- bash: Execute shell commands
- file_viewer: View file with line numbers
- write_file: Write file to disk
- edit_file: Edit file with string replacement

Validation utilities (from validation.py):
- run_linter: Run syntax and style checks
- validate_code_against_spec: Validate code matches specification
- extract_interface_stub: Extract interface stub from Python file
- validate_file_exists: Check if file exists
- validate_python_file: Validate Python syntax
- validate_imports: Validate import statements
- count_lines_of_code: Count LOC statistics

Tool collections:
- get_architect_tools: Tools for Architect agents
- get_worker_tools: Tools for Worker agents
- get_integrator_tools: Tools for Integrator agents

Parsing utilities (from parsing.py):
- extract_json_from_llm_output: Extract JSON from LLM output
- extract_code_block: Extract code block from LLM output
- parse_to_model: Parse LLM output to Pydantic model
"""

from src.agents.experiment_agent.shared.tools.core import (
    # Security
    SecurityContext,
    # Core tools
    bash,
    file_viewer,
    write_file,
    edit_file,
    # Tool collections
    get_architect_tools,
    get_worker_tools,
    get_integrator_tools,
)

from src.agents.experiment_agent.shared.tools.validation import (
    run_linter,
    validate_code_against_spec,
    extract_interface_stub,
    validate_file_exists,
    validate_python_file,
    validate_imports,
    count_lines_of_code,
)

from src.agents.experiment_agent.shared.tools.parsing import (
    extract_json_from_llm_output,
    extract_code_block,
    parse_to_model,
    extract_verdict,
    extract_status,
    clean_llm_output,
)

__all__ = [
    # Security
    "SecurityContext",
    # Core tools
    "bash",
    "file_viewer",
    "write_file",
    "edit_file",
    # Validation
    "run_linter",
    "validate_code_against_spec",
    "extract_interface_stub",
    "validate_file_exists",
    "validate_python_file",
    "validate_imports",
    "count_lines_of_code",
    # Tool collections
    "get_architect_tools",
    "get_worker_tools",
    "get_integrator_tools",
    # Parsing
    "extract_json_from_llm_output",
    "extract_code_block",
    "parse_to_model",
    "extract_verdict",
    "extract_status",
    "clean_llm_output",
]
