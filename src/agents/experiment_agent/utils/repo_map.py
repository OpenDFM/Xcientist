"""
Repo Map Generator - Generate compressed code skeleton using Tree-sitter.

Similar to Aider's RepoMap, this module generates a condensed view of the codebase
showing only class/function signatures without implementation details.
"""

import os
import re
from typing import Optional
from pathlib import Path

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


class RepoMapGenerator:
    """Generate repository map (code skeleton) for a project."""
    
    SKIP_DIRS = {
        "__pycache__", ".git", ".venv", "venv", "env",
        "node_modules", ".idea", ".vscode", "build", "dist",
        "egg-info", ".eggs", ".tox", ".pytest_cache",
        "logs", "results", "checkpoints", "data", "datasets",
    }
    
    SKIP_FILES = {"__init__.py", "setup.py", "conftest.py"}
    
    def __init__(self, max_depth: int = 5, max_files: int = 50):
        self.max_depth = max_depth
        self.max_files = max_files
        
        if TREE_SITTER_AVAILABLE:
            self.parser = Parser(Language(tspython.language()))
        else:
            self.parser = None
    
    def generate(self, project_dir: str, relative_to: Optional[str] = None) -> str:
        """Generate repo map for the given project directory."""
        if not TREE_SITTER_AVAILABLE:
            return self._fallback_generate(project_dir, relative_to)
        
        if relative_to is None:
            relative_to = project_dir
        
        project_path = Path(project_dir)
        if not project_path.exists():
            return f"[Project directory not found: {project_dir}]"
        
        py_files = self._collect_files(project_path, relative_to)
        
        if not py_files:
            return "[No Python files found in project]"
        
        lines = ["=== REPO MAP (Code Skeleton) ===", ""]
        
        file_count = 0
        for rel_path, full_path in sorted(py_files):
            if file_count >= self.max_files:
                lines.append(f"... ({len(py_files) - file_count} more files)")
                break
            
            skeleton = self._extract_skeleton(full_path)
            if skeleton:
                lines.append(f"📄 {rel_path}")
                lines.append(skeleton)
                lines.append("")
                file_count += 1
        
        lines.append("=== END REPO MAP ===")
        return "\n".join(lines)
    
    def _collect_files(self, project_path: Path, relative_to: str) -> list:
        """Collect all Python files in the project."""
        files = []
        relative_base = Path(relative_to)
        
        def _walk(path: Path, depth: int = 0):
            if depth > self.max_depth:
                return
            
            try:
                entries = sorted(path.iterdir())
            except PermissionError:
                return
            
            for entry in entries:
                if entry.is_dir():
                    if entry.name not in self.SKIP_DIRS and not entry.name.startswith("."):
                        _walk(entry, depth + 1)
                elif entry.is_file():
                    if entry.suffix == ".py" and entry.name not in self.SKIP_FILES:
                        try:
                            rel_path = entry.relative_to(relative_base)
                            files.append((str(rel_path), str(entry)))
                        except ValueError:
                            files.append((entry.name, str(entry)))
        
        _walk(project_path)
        return files
    
    def _extract_skeleton(self, file_path: str) -> str:
        """Extract code skeleton from a Python file using AST traversal."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception as e:
            return f"  [Error reading file: {e}]"
        
        if not code.strip():
            return "  [Empty file]"
        
        tree = self.parser.parse(code.encode())
        
        if tree.root_node.has_error:
            return self._simple_extract(code)
        
        return self._traverse_ast(tree.root_node, code)
    
    def _traverse_ast(self, root, code: str) -> str:
        """Traverse AST and extract skeleton using direct node inspection."""
        lines = []
        
        # Extract imports
        imports = self._get_imports(root)
        if imports:
            display = imports[:5]
            suffix = f" (+{len(imports)-5} more)" if len(imports) > 5 else ""
            lines.append(f"  # imports: {', '.join(display)}{suffix}")
        
        # Extract top-level definitions
        for child in root.children:
            if child.type == "class_definition":
                lines.extend(self._extract_class(child))
            elif child.type == "function_definition":
                lines.append(self._extract_function(child, indent=2))
            elif child.type == "decorated_definition":
                for subchild in child.children:
                    if subchild.type == "class_definition":
                        lines.extend(self._extract_class(subchild))
                    elif subchild.type == "function_definition":
                        lines.append(self._extract_function(subchild, indent=2))
        
        return "\n".join(lines) if lines else "  [No classes or functions]"
    
    def _get_imports(self, root) -> list:
        """Extract import module names by traversing AST."""
        imports = []
        
        for child in root.children:
            if child.type == "import_statement":
                # import foo, bar
                for subchild in child.children:
                    if subchild.type == "dotted_name":
                        imports.append(subchild.text.decode())
                    elif subchild.type == "aliased_import":
                        for name in subchild.children:
                            if name.type == "dotted_name":
                                imports.append(name.text.decode())
                                break
            elif child.type == "import_from_statement":
                # from foo import bar
                for subchild in child.children:
                    if subchild.type == "dotted_name":
                        imports.append(subchild.text.decode())
                        break
        
        return imports
    
    def _extract_class(self, node, indent: int = 2) -> list:
        """Extract class definition with methods."""
        lines = []
        prefix = " " * indent
        
        class_name = "?"
        bases = []
        
        for child in node.children:
            if child.type == "identifier":
                class_name = child.text.decode()
            elif child.type == "argument_list":
                # Base classes
                for arg in child.children:
                    if arg.type == "identifier" or arg.type == "attribute":
                        bases.append(arg.text.decode())
        
        base_str = f"({', '.join(bases)})" if bases else ""
        lines.append(f"{prefix}class {class_name}{base_str}:")
        
        # Find body block and extract methods
        body = None
        for child in node.children:
            if child.type == "block":
                body = child
                break
        
        if body:
            method_count = 0
            for child in body.children:
                if child.type == "function_definition":
                    lines.append(self._extract_function(child, indent=indent+4))
                    method_count += 1
                elif child.type == "decorated_definition":
                    for subchild in child.children:
                        if subchild.type == "function_definition":
                            lines.append(self._extract_function(subchild, indent=indent+4))
                            method_count += 1
            
            if method_count == 0:
                lines.append(f"{prefix}    ...")
        
        return lines
    
    def _extract_function(self, node, indent: int = 2) -> str:
        """Extract function signature."""
        prefix = " " * indent
        
        func_name = "?"
        params = "()"
        return_type = ""
        
        for child in node.children:
            if child.type == "identifier":
                func_name = child.text.decode()
            elif child.type == "parameters":
                params = child.text.decode()
            elif child.type == "type":
                return_type = f" -> {child.text.decode()}"
        
        return f"{prefix}def {func_name}{params}{return_type}: ..."
    
    def _simple_extract(self, code: str) -> str:
        """Simple regex-based extraction for files with syntax errors."""
        lines = []
        
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("class "):
                match = re.match(r"class\s+(\w+).*:", stripped)
                if match:
                    lines.append(f"  class {match.group(1)}: ...")
            elif stripped.startswith("def "):
                match = re.match(r"def\s+(\w+)\s*\([^)]*\).*:", stripped)
                if match:
                    indent_level = len(line) - len(line.lstrip())
                    prefix = "  " if indent_level == 0 else "      "
                    lines.append(f"{prefix}def {match.group(1)}(...): ...")
        
        return "\n".join(lines) if lines else "  [Could not parse]"
    
    def _fallback_generate(self, project_dir: str, relative_to: Optional[str]) -> str:
        """Fallback when tree-sitter is not available."""
        if relative_to is None:
            relative_to = project_dir
        
        project_path = Path(project_dir)
        if not project_path.exists():
            return f"[Project directory not found: {project_dir}]"
        
        lines = ["=== REPO MAP (Fallback Mode) ===", ""]
        
        py_files = self._collect_files(project_path, relative_to)
        
        for rel_path, full_path in sorted(py_files)[:self.max_files]:
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()
                
                skeleton = self._simple_extract(code)
                if skeleton:
                    lines.append(f"📄 {rel_path}")
                    lines.append(skeleton)
                    lines.append("")
            except Exception:
                continue
        
        lines.append("=== END REPO MAP ===")
        return "\n".join(lines)


def generate_repo_map(project_dir: str, max_files: int = 50) -> str:
    """
    Convenience function to generate repo map.
    
    Args:
        project_dir: Path to project directory
        max_files: Maximum number of files to include
        
    Returns:
        Formatted repo map string
    """
    generator = RepoMapGenerator(max_files=max_files)
    return generator.generate(project_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(generate_repo_map(sys.argv[1]))
    else:
        print("Usage: python repo_map.py <project_dir>")
