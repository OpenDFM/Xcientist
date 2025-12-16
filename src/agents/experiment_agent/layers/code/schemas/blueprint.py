from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FunctionSignature(BaseModel):
    """Function signature specification - all fields required."""

    name: str = Field(description="Name of the function")
    args: str = Field(description="Arguments string, e.g. 'x: int, y: int'")
    return_type: str = Field(description="Return type annotation")
    docstring: str = Field(description="Docstring describing the function")


class ClassSignature(BaseModel):
    """Class signature specification - all fields required."""

    name: str = Field(description="Name of the class")
    methods: List[FunctionSignature] = Field(description="List of methods in the class")
    docstring: str = Field(description="Docstring describing the class")
    attributes: Optional[Dict[str, str]] = Field(
        default=None, description="Class attributes and their types"
    )


class FileSpec(BaseModel):
    """File specification - required fields for code generation."""

    file_path: str = Field(description="Relative path to the file")
    description: str = Field(description="Description of the file's purpose")
    dependencies: List[str] = Field(
        description="List of file paths this file depends on"
    )
    classes: List[ClassSignature] = Field(
        default_factory=list, description="Classes defined in this file"
    )
    functions: List[FunctionSignature] = Field(
        default_factory=list, description="Top-level functions defined in this file"
    )
    data_structures: Optional[List[str]] = Field(
        default=None, description="Names of data structures defined here"
    )
    test_file: Optional[str] = Field(
        default=None,
        description="Path to corresponding test file. If set, test file will be generated BEFORE this file.",
    )
    is_test: bool = Field(default=False, description="Whether this is a test file")


class BlueprintHandover(BaseModel):
    """
    Optional handover hints for building CodeManifest (CHP).

    The goal is to avoid guessing in the orchestrator: Architect should provide these explicitly.
    """

    entry_points: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of script name -> runnable command, e.g. {'train': 'python train.py --config config.yaml'}",
    )
    config_file: Optional[str] = Field(
        default=None,
        description="Path to main configuration file (relative to project root)",
    )
    config_format: Optional[str] = Field(
        default=None, description="Config format hint: 'yaml', 'json', 'py'"
    )
    metrics_log_file: Optional[str] = Field(
        default=None, description="Path to metrics log file (relative to project root)"
    )
    metrics_log_format: Optional[str] = Field(
        default=None, description="Metrics log format: 'json', 'csv', 'tensorboard'"
    )
    metrics_keys: List[str] = Field(
        default_factory=list, description="Metric keys to extract from metrics_log_file"
    )
    primary_metric: Optional[str] = Field(
        default=None, description="Primary metric name for optimization"
    )
    higher_is_better: Optional[bool] = Field(
        default=None, description="Whether higher values are better for primary_metric"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Reserved for future extensions (do not rely on arbitrary keys)",
    )


class Blueprint(BaseModel):
    """Project blueprint specification."""

    file_tree: List[str] = Field(description="List of all file paths in the project")
    files: List[FileSpec] = Field(description="Detailed specifications for each file")
    shared_data_structures: Dict[str, str] = Field(
        description="Definitions of shared data structures"
    )
    entry_point: str = Field(description="Path to the entry point file (e.g., main.py)")
    handover: Optional[BlueprintHandover] = Field(
        default=None,
        description="Optional CodeManifest (CHP) handover hints to avoid guessing",
    )

    def validate_dag(self) -> bool:
        """
        Validate that the dependency graph is a valid DAG (no cycles).
        Raises ValueError if a cycle is detected or if dependencies are missing.
        """
        # Consistency guard: every path in file_tree should have a corresponding FileSpec.
        # This is important because execution tasks are derived from `files`, while many other
        # parts of the system assume `file_tree` represents the complete project.
        file_tree_set = set(self.file_tree or [])
        spec_set = {f.file_path for f in self.files}

        missing_specs = sorted(list(file_tree_set - spec_set))
        extra_specs = sorted(list(spec_set - file_tree_set))

        if missing_specs or extra_specs:
            missing_preview = ", ".join(missing_specs[:20])
            extra_preview = ", ".join(extra_specs[:20])
            raise ValueError(
                "Blueprint inconsistency: `file_tree` and `files[*].file_path` must match. "
                f"missing_specs={len(missing_specs)} [{missing_preview}] "
                f"extra_specs={len(extra_specs)} [{extra_preview}]"
            )

        graph = {}
        all_files = {f.file_path for f in self.files}

        for file_spec in self.files:
            deps = set(file_spec.dependencies) & all_files
            graph[file_spec.file_path] = deps

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in graph}

        def dfs(node: str, path: List[str]):
            color[node] = GRAY
            path.append(node)

            for dep in graph.get(node, set()):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    cycle_start = path.index(dep)
                    cycle = path[cycle_start:] + [dep]
                    raise ValueError(
                        f"Circular dependency detected: {' -> '.join(cycle)}"
                    )
                if color[dep] == WHITE:
                    dfs(dep, path)

            path.pop()
            color[node] = BLACK

        for node in graph:
            if color[node] == WHITE:
                dfs(node, [])

        return True
