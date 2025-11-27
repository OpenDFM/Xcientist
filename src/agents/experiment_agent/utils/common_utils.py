import os
from typing import Any, Dict, List, Optional


def read_file_smart(working_dir: str, file_path: str, max_lines: int = 300) -> str:
    """
    Read file content with smart truncation for context injection.
    Returns formatted string with content or skeletal structure.

    Args:
        working_dir: Workspace directory (will append 'project' subdirectory)
        file_path: Relative path to the file
        max_lines: Maximum lines to read before truncating

    Returns:
        File content or error message
    """
    # Construct full path - assuming project root is working_dir/project
    full_path = os.path.join(working_dir, "project", file_path)

    # Check if path exists
    if not os.path.exists(full_path):
        return f"[File not found: {file_path}]"

    # Check if path is a directory
    if os.path.isdir(full_path):
        return f"[Directory: {file_path} - use list_directory to see contents]"

    # Try to read the file
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) <= max_lines:
            return "".join(lines)
        else:
            # For large files, return head + summary hint
            return (
                "".join(lines[:50])
                + f"\n\n... ({len(lines)-50} more lines. Use read_file tool to see full content) ..."
            )
    except UnicodeDecodeError:
        # Handle binary files
        return f"[Binary file: {file_path} - cannot display as text]"
    except Exception as e:
        return f"[Error reading file: {str(e)}]"


def format_list(items: List[Any]) -> str:
    """Format a list into a readable string."""
    if not items:
        return "N/A"
    return "\n".join(f"- {item}" for item in items)


def format_dict(d: Dict[str, Any]) -> str:
    """Format a dictionary into a readable string."""
    if not d:
        return "N/A"
    return "\n".join(f"- {key}: {value}" for key, value in d.items())


def extract_core_plan_context(plan_output: Any) -> str:
    """
    Extract only core design context from the full code plan.
    """
    if not plan_output:
        return "No plan available."

    # Handle both dict and object formats
    is_dict = isinstance(plan_output, dict)

    def get_field(name: str, default: str = "N/A") -> str:
        if is_dict:
            return str(plan_output.get(name, default))
        return str(getattr(plan_output, name, default))

    # Extract high-level design components
    dataset_plan = get_field("dataset_plan", "See File Structure")
    model_plan = get_field("model_plan", "See File Structure")
    training_plan = get_field("training_plan", "Standard Training Loop")

    return f"""
=== GLOBAL DESIGN CONTEXT ===

[Dataset Design]
{dataset_plan}

[Model Architecture]
{model_plan}

[Training Configuration]
{training_plan}
"""


def extract_analysis_summary(analysis_output: Any) -> str:
    """
    Extract theoretically relevant parts from analysis for Code Judge.
    Focuses on Algorithms, Math, and Innovations rather than full text.
    """
    if not analysis_output:
        return "N/A"

    # Helper to get attribute from object or dict
    def get_field(name: str) -> str:
        if isinstance(analysis_output, dict):
            val = analysis_output.get(name)
        else:
            val = getattr(analysis_output, name, None)
        return str(val) if val else ""

    # Extract only what a code reviewer needs to verify correctness
    algorithms = get_field("algorithms")
    math = get_field("mathematical_formulations")
    innovations = get_field("key_innovations")
    summary = get_field("summary")

    # Fallback to summary if specific fields are missing
    if not any([algorithms, math, innovations]):
        if summary:
            return summary
        # Ultimate fallback if structure is unknown
        return str(analysis_output)[:2000] + "..."

    return f"""
[Research Summary]
{summary}

[Key Innovations]
{innovations}

[Core Algorithms]
{algorithms}

[Mathematical Formulations]
{math}
"""
