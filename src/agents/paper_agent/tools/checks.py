import os
import re
from typing import Dict, List, Set, Any

from agents import function_tool


def _extract_citations_from_tex(tex_content: str) -> Set[str]:
    """
    Extract citation keys from LaTeX content.
    Matches \cite{...}, \citep{...}, \citet{...}, \citeauthor{...}, etc.
    Handles multiple keys in one cite command (e.g. \cite{key1,key2}).
    """
    citations = set()
    # Pattern matches \cite*{key1,key2}
    # exclude comments (%)
    lines = tex_content.splitlines()
    clean_lines = []
    for line in lines:
        if "%" in line:
            line = line.split("%")[0]
        clean_lines.append(line)
    text = "\n".join(clean_lines)

    # Regex for \cite command families
    # matches \cite followed by optional [*] or [opt] then {keys}
    pattern = r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}"
    
    for match in re.finditer(pattern, text):
        keys_str = match.group(1)
        for key in keys_str.split(","):
            k = key.strip()
            if k:
                citations.add(k)
    return citations


def _extract_keys_from_bib(bib_content: str) -> Set[str]:
    """
    Extract keys from BibTeX content.
    Matches @type{key, ...
    """
    keys = set()
    # Simple regex for bibtex keys: @type{key,
    pattern = r"@[a-zA-Z]+\s*\{\s*([^,]+),"
    for match in re.finditer(pattern, bib_content):
        k = match.group(1).strip()
        if k:
            keys.add(k)
    return keys


@function_tool
def check_citations(paper_dir: str, bib_file: str = "references.bib") -> Dict[str, Any]:
    """
    Static analysis to validate that all citations in .tex files exist in the .bib file.
    
    Args:
        paper_dir: Directory containing .tex files
        bib_file: Path to the .bib file (relative to paper_dir or absolute)
        
    Returns:
        Dict with:
        - valid: bool (True if all citations are found)
        - missing_keys: List[str] (citations used in tex but missing in bib)
        - unused_keys: List[str] (citations in bib but not used in tex)
        - stats: Dict (counts)
    """
    try:
        paper_dir = os.path.abspath(paper_dir)
        
        # Locate bib file
        bib_path = bib_file
        if not os.path.isabs(bib_path):
            bib_path = os.path.join(paper_dir, bib_path)
        
        # Read BibTeX keys
        defined_keys = set()
        if os.path.exists(bib_path):
            try:
                with open(bib_path, "r", encoding="utf-8", errors="ignore") as f:
                    bib_content = f.read()
                defined_keys = _extract_keys_from_bib(bib_content)
            except Exception as e:
                return {"valid": False, "error": f"Failed to read bib file: {e}"}
        else:
            # If bib file doesn't exist, all citations are undefined (unless none are used)
            pass

        # Scan all .tex files
        used_keys = set()
        scanned_files = []
        
        if not os.path.isdir(paper_dir):
             return {"valid": False, "error": f"paper_dir not found: {paper_dir}"}

        for root, _, files in os.walk(paper_dir):
            for file in files:
                if file.endswith(".tex"):
                    path = os.path.join(root, file)
                    scanned_files.append(os.path.relpath(path, paper_dir))
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        keys = _extract_citations_from_tex(content)
                        used_keys.update(keys)
                    except Exception:
                        pass
        
        missing_keys = sorted(list(used_keys - defined_keys))
        unused_keys = sorted(list(defined_keys - used_keys))
        
        return {
            "valid": len(missing_keys) == 0,
            "missing_keys": missing_keys,
            "unused_keys": unused_keys,
            "stats": {
                "n_tex_files": len(scanned_files),
                "n_citations_used": len(used_keys),
                "n_citations_defined": len(defined_keys),
                "bib_path": bib_path
            }
        }
    except Exception as e:
        import traceback
        return {
            "valid": False,
            "error": f"Unexpected error in check_citations: {str(e)}",
            "traceback": traceback.format_exc()
        }

