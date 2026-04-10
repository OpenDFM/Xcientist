"""
Utility functions for repository structure formatting and processing.
"""

import os
from typing import Dict, List, Tuple, Set


# Default configuration for format_repo_structure
DEFAULT_CODE_EXTENSIONS = {".py", ".ipynb", ".yaml", ".yml", ".json", ".cfg", ".toml", ".sh"}
DEFAULT_HEUR_KEYWORDS = {"train", "model", "main", "run", "inference", "predict", "forward", "pipeline"}
DEFAULT_MAX_LINES_WARNING = 500
DEFAULT_MAX_LINES_CODE_ONLY = 200


def format_repo_structure(
    structure: dict,
    code_extensions: Set[str] = None,
    heur_keywords: Set[str] = None,
    max_lines_warning: int = None,
    max_lines_code_only: int = None,
) -> str:
    """
    Format repository structure as a tree string with fallback mechanisms.
    
    Fallback levels:
    1. Full structure (if lines <= threshold)
    2. Code files only (if full structure exceeds threshold)
    3. Keyword-filtered code files (if code-only still exceeds threshold)
    4. First N lines with warning (final fallback)
    
    Args:
        structure: Repository structure dict
        code_extensions: Set of code file extensions to consider (default: DEFAULT_CODE_EXTENSIONS)
        heur_keywords: Set of keywords to filter files (default: DEFAULT_HEUR_KEYWORDS)
        max_lines_warning: Line threshold for showing full structure (default: DEFAULT_MAX_LINES_WARNING)
        max_lines_code_only: Line threshold for code-only filtering (default: DEFAULT_MAX_LINES_CODE_ONLY)
        
    Returns:
        Formatted repository structure string
    """
    if code_extensions is None:
        code_extensions = DEFAULT_CODE_EXTENSIONS
    if heur_keywords is None:
        heur_keywords = DEFAULT_HEUR_KEYWORDS
    if max_lines_warning is None:
        max_lines_warning = DEFAULT_MAX_LINES_WARNING
    if max_lines_code_only is None:
        max_lines_code_only = DEFAULT_MAX_LINES_CODE_ONLY
    
    def _build_tree_string(
        node: dict,
        prefix: str = "",
        only_code_files: bool = False,
        keyword_filter: bool = False
    ) -> List[str]:
        """Build tree string recursively."""
        lines = []
        keys = sorted(node.keys())
        total_items = len(keys)
        
        for i, key in enumerate(keys):
            value = node[key]
            is_last = (i == total_items - 1)
            connector = "└── " if is_last else "├── "
            
            if isinstance(value, dict) and value.get("_is_file"):
                # It's a file
                if only_code_files:
                    file_type = value.get("type", "").lower()
                    if file_type not in code_extensions:
                        continue  # Skip non-code files
                    
                    if keyword_filter:
                        # Check if filename contains any keyword
                        lower_key = key.lower()
                        if not any(kw in lower_key for kw in heur_keywords):
                            continue
                
                lines.append(f"{prefix}{connector}{key}")
                
                # Add annotations
                if value.get("is_core_code"):
                    lines[-1] += " (core code)"
                if value.get("is_main_code"):
                    lines[-1] += " (main code)"
            else:
                # It's a directory
                lines.append(f"{prefix}{connector}{key}")
                extension = "    " if is_last else "│   "
                sub_lines = _build_tree_string(
                    value, 
                    prefix + extension,
                    only_code_files=only_code_files,
                    keyword_filter=keyword_filter
                )
                lines.extend(sub_lines)
        
        return lines
    
    # First attempt: full structure
    full_lines = _build_tree_string(structure)
    
    # Check if we need fallback
    if len(full_lines) <= max_lines_warning:
        return "\n".join(full_lines)
    
    # Fallback 1: Code files only
    code_only_lines = _build_tree_string(structure, only_code_files=True)
    
    if len(code_only_lines) <= max_lines_code_only:
        return "[Note: Showing code files only (non-code files omitted due to large repository)]\n" + "\n".join(code_only_lines)
    
    # Fallback 2: Keyword-filtered code files
    keyword_filtered_lines = _build_tree_string(
        structure, 
        only_code_files=True, 
        keyword_filter=True
    )
    
    if len(keyword_filtered_lines) <= max_lines_code_only:
        return "[Note: Showing key code files only (filtered by keywords: {}))\n".format(
            ", ".join(sorted(heur_keywords))
        ) + "\n".join(keyword_filtered_lines)
    
    # Fallback 3: First N lines with warning (code files only)
    # Use keyword-filtered code files as the base for truncation
    truncated_lines = keyword_filtered_lines[:max_lines_code_only]
    return (
        "[WARNING: Repository structure too large. Showing first {} lines of code files only. "
        "Some files and directories are omitted.]\n".format(max_lines_code_only)
        + "\n".join(truncated_lines)
        + "\n[...]"
    )


def filter_code_files_in_structure(
    structure: dict,
    code_extensions: Set[str] = None,
) -> List[str]:
    """
    Get list of code file paths from repository structure.
    
    Args:
        structure: Repository structure dict
        code_extensions: Set of code file extensions to consider
        
    Returns:
        List of file paths (relative)
    """
    if code_extensions is None:
        code_extensions = DEFAULT_CODE_EXTENSIONS
    
    code_files = []
    
    def _collect_files(node: dict, current_path: str = ""):
        for key, value in node.items():
            file_path = os.path.join(current_path, key) if current_path else key
            if isinstance(value, dict) and value.get("_is_file"):
                file_type = value.get("type", "").lower()
                if file_type in code_extensions:
                    code_files.append(file_path)
            elif isinstance(value, dict):
                _collect_files(value, file_path)
    
    _collect_files(structure)
    return code_files
