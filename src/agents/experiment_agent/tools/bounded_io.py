"""Bounded file and terminal tool overrides for experiment agents."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Literal, Sequence

from pydantic import Field

from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.file_editor import FileEditorTool as UpstreamFileEditorTool
from openhands.tools.file_editor.definition import (
    FileEditorAction as UpstreamFileEditorAction,
)
from openhands.tools.file_editor.impl import FileEditorExecutor as UpstreamFileEditorExecutor
from openhands.tools.terminal import TerminalTool as UpstreamTerminalTool
from openhands.tools.terminal.definition import (
    TerminalAction,
    TerminalObservation,
    TOOL_DESCRIPTION as UPSTREAM_TERMINAL_DESCRIPTION,
)
from openhands.tools.terminal.impl import TerminalExecutor as UpstreamTerminalExecutor

from src.agents.experiment_agent.runtime.self_contained import VALID_ENV_NAME
from src.agents.experiment_agent.tools.openhands import SecurityValidator
from src.agents.experiment_agent.tools.resource_tools import enable_resource_tools

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation


FILE_VIEW_MAX_LINES = int(os.environ.get("EXPERIMENT_AGENT_FILE_VIEW_MAX_LINES", "160"))
TOOL_MAX_CHARS = int(os.environ.get("EXPERIMENT_AGENT_TOOL_MAX_CHARS", "12000"))
SEARCH_MAX_MATCHES = int(os.environ.get("EXPERIMENT_AGENT_SEARCH_MAX_MATCHES", "8"))
DIR_VIEW_MAX_ENTRIES = int(os.environ.get("EXPERIMENT_AGENT_DIR_VIEW_MAX_ENTRIES", "80"))
TERMINAL_MAX_LINES = int(os.environ.get("EXPERIMENT_AGENT_TERMINAL_MAX_LINES", "160"))
LARGE_FILE_BYTES = int(os.environ.get("EXPERIMENT_AGENT_LARGE_FILE_BYTES", "524288"))
SEARCH_CONTEXT_BEFORE = int(os.environ.get("EXPERIMENT_AGENT_SEARCH_CONTEXT_BEFORE", "3"))
SEARCH_CONTEXT_AFTER = int(os.environ.get("EXPERIMENT_AGENT_SEARCH_CONTEXT_AFTER", "3"))
TERMINAL_HEAD_LINES = max(1, TERMINAL_MAX_LINES // 2)
TERMINAL_TAIL_LINES = TERMINAL_MAX_LINES - TERMINAL_HEAD_LINES


def _truncate_text(text: str, *, max_chars: int = TOOL_MAX_CHARS) -> tuple[str, bool]:
    if max_chars < 32 or len(text) <= max_chars:
        return text, False
    notice = "\n... <truncated>\n"
    available = max_chars - len(notice)
    if available <= 0:
        return text[:max_chars], True
    head = available // 2
    tail = available - head
    return text[:head] + notice + text[-tail:], True


def _ensure_abs_path(path: str, workspace_root: str) -> Path:
    if os.path.isabs(path):
        abs_path = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    else:
        abs_path = os.path.realpath(os.path.abspath(os.path.join(workspace_root, path)))
    SecurityValidator.validate_or_raise(abs_path, workspace_root, "access")
    return Path(abs_path)


def _save_text_artifact(base_dir: str | None, prefix: str, content: str) -> str | None:
    if not base_dir:
        return None
    artifact_dir = Path(base_dir) / "tool_artifacts" / prefix
    artifact_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()[:12]
    artifact_path = artifact_dir / f"{prefix}_{digest}.txt"
    if not artifact_path.exists():
        artifact_path.write_text(content, encoding="utf-8")
    return str(artifact_path)


def _iter_visible_entries(root: Path, *, max_depth: int, glob_pattern: str | None) -> Iterable[Path]:
    root_depth = len(root.parts)
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        rel_depth = len(current_path.parts) - root_depth
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        if rel_depth >= max_depth:
            dirnames[:] = []
        for name in sorted(dirnames):
            candidate = current_path / name
            rel_name = str(candidate.relative_to(root))
            if glob_pattern and not fnmatch.fnmatch(rel_name, glob_pattern):
                continue
            yield candidate
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            candidate = current_path / name
            rel_name = str(candidate.relative_to(root))
            if glob_pattern and not fnmatch.fnmatch(rel_name, glob_pattern):
                continue
            yield candidate


def _render_directory_listing(
    path: Path,
    *,
    max_depth: int,
    max_entries: int,
    glob_pattern: str | None,
) -> tuple[str, bool, int]:
    entries = list(_iter_visible_entries(path, max_depth=max_depth, glob_pattern=glob_pattern))
    omitted = max(0, len(entries) - max_entries)
    visible_entries = entries[:max_entries]
    lines = [
        f"[file_editor:list]",
        f"path: {path}",
        f"max_depth: {max_depth}",
        f"returned_entries: {len(visible_entries)}",
        f"omitted_entries: {omitted}",
    ]
    if glob_pattern:
        lines.append(f"glob: {glob_pattern}")
    lines.append("entries:")
    for entry in visible_entries:
        entry_type = "dir" if entry.is_dir() else "file"
        size = entry.stat().st_size if entry.is_file() else 0
        lines.append(f"- {entry_type} {entry} size={size}")
    text, clipped = _truncate_text("\n".join(lines))
    return text, (omitted > 0) or clipped, omitted


def _read_text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _is_binary_file(path: Path) -> bool:
    try:
        sample = path.read_bytes()[:4096]
    except Exception:
        return False
    return b"\x00" in sample


def _format_line_block(path: Path, start_line: int, end_line: int, lines: list[str]) -> str:
    rendered = [
        f"{line_no:>6}\t{line}"
        for line_no, line in zip(range(start_line, end_line + 1), lines, strict=False)
    ]
    return "\n".join(
        [
            f"[file_editor:view]",
            f"path: {path}",
            f"line_range: {start_line}-{end_line}",
            "content:",
            *rendered,
        ]
    )


def _json_path_tokens(selector: str) -> list[str | int]:
    normalized = selector.strip()
    if normalized.startswith("$."):
        normalized = normalized[2:]
    elif normalized.startswith("$"):
        normalized = normalized[1:]
    normalized = normalized.lstrip(".")
    if not normalized:
        return []
    pattern = re.compile(r"([^.\\[\\]]+)|\\[(\\d+)\\]")
    tokens: list[str | int] = []
    for match in pattern.finditer(normalized):
        key_token, index_token = match.groups()
        if key_token is not None:
            tokens.append(key_token)
        elif index_token is not None:
            tokens.append(int(index_token))
    return tokens


def _extract_json_value(payload: Any, selector: str) -> Any:
    current = payload
    for token in _json_path_tokens(selector):
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise KeyError(selector)
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                raise KeyError(selector)
            current = current[token]
    return current


class BoundedFileEditorAction(Action):
    """Schema for bounded file editor operations."""

    command: Literal[
        "view",
        "list",
        "search",
        "read_json",
        "stat",
        "create",
        "str_replace",
        "insert",
        "undo_edit",
    ] = Field(
        description=(
            "Bounded file command to run. Prefer `list`, `stat`, `read_json`, or "
            "`search` before `view`. `view` returns only a bounded window and "
            "never returns a whole long file."
        )
    )
    path: str = Field(description="Absolute or workspace-relative file or directory path.")
    view_range: list[int] | None = Field(
        default=None,
        description=(
            "Optional line range for `view` on text files, inclusive and 1-based. "
            "If omitted, only the first bounded window is returned."
        ),
    )
    file_text: str | None = Field(
        default=None,
        description="Required for `create`; plain-text content to write.",
    )
    old_str: str | None = Field(
        default=None,
        description="Required for `str_replace`; exact old string to replace.",
    )
    new_str: str | None = Field(
        default=None,
        description="Replacement text for `str_replace` or inserted text for `insert`.",
    )
    insert_line: int | None = Field(
        default=None,
        ge=0,
        description="Required for `insert`; insert after this 1-based line number.",
    )
    query: str | None = Field(
        default=None,
        description="Required for `search`; literal query or regex pattern.",
    )
    regex: bool = Field(
        default=False,
        description="If true, interpret `query` as a regex for `search`.",
    )
    context_before: int = Field(
        default=SEARCH_CONTEXT_BEFORE,
        ge=0,
        le=20,
        description="Number of context lines to show before each `search` match.",
    )
    context_after: int = Field(
        default=SEARCH_CONTEXT_AFTER,
        ge=0,
        le=20,
        description="Number of context lines to show after each `search` match.",
    )
    max_matches: int = Field(
        default=SEARCH_MAX_MATCHES,
        ge=1,
        le=50,
        description="Maximum number of `search` matches to return.",
    )
    json_paths: list[str] | None = Field(
        default=None,
        description=(
            "Required for `read_json`; JSON selectors such as `$.status` or "
            "`findings[0].check`."
        ),
    )
    max_depth: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum directory depth for `list` or directory `view`.",
    )
    max_entries: int = Field(
        default=DIR_VIEW_MAX_ENTRIES,
        ge=1,
        le=200,
        description="Maximum entries returned for `list` or directory `view`.",
    )
    glob: str | None = Field(
        default=None,
        description="Optional glob filter for `list`.",
    )


class BoundedFileEditorObservation(Observation):
    """Bounded observation returned to the model."""

    command: str = Field(description="The bounded file editor command that ran.")
    path: str | None = Field(default=None, description="The file or directory path.")
    prev_exist: bool = Field(default=True)
    old_content: str | None = Field(default=None)
    new_content: str | None = Field(default=None)
    truncated: bool = Field(default=False)
    returned_line_start: int | None = Field(default=None)
    returned_line_end: int | None = Field(default=None)
    total_lines: int | None = Field(default=None)
    next_view_range: list[int] | None = Field(default=None)
    omitted_entries: int = Field(default=0)


class BoundedFileEditorExecutor(ToolExecutor[BoundedFileEditorAction, BoundedFileEditorObservation]):
    """Executor enforcing bounded file access patterns."""

    def __init__(self, workspace_root: str):
        self.workspace_root = os.path.realpath(workspace_root)
        self._upstream = UpstreamFileEditorExecutor(workspace_root=self.workspace_root)

    def _observe_text(
        self,
        *,
        text: str,
        command: str,
        path: Path,
        truncated: bool = False,
        returned_line_start: int | None = None,
        returned_line_end: int | None = None,
        total_lines: int | None = None,
        next_view_range: list[int] | None = None,
        omitted_entries: int = 0,
        is_error: bool = False,
    ) -> BoundedFileEditorObservation:
        final_text, clipped = _truncate_text(text)
        return BoundedFileEditorObservation.from_text(
            text=final_text,
            is_error=is_error,
            command=command,
            path=str(path),
            truncated=truncated or clipped,
            returned_line_start=returned_line_start,
            returned_line_end=returned_line_end,
            total_lines=total_lines,
            next_view_range=next_view_range,
            omitted_entries=omitted_entries,
        )

    def _is_model_share_path(self, path: Path) -> bool:
        model_share_root = Path(self.workspace_root) / "model_candidate" / "model_share"
        model_share_abs = os.path.abspath(str(model_share_root))
        path_abs = os.path.abspath(str(path))
        return path_abs == model_share_abs or path_abs.startswith(model_share_abs + os.sep)

    def _run_edit(self, action: BoundedFileEditorAction, path: Path) -> BoundedFileEditorObservation:
        if self._is_model_share_path(path):
            return self._observe_text(
                text=f"Editing is not allowed under read-only shared model mount: {path}",
                command=action.command,
                path=path,
                is_error=True,
            )
        upstream_action = UpstreamFileEditorAction(
            command=action.command,  # type: ignore[arg-type]
            path=str(path),
            file_text=action.file_text,
            old_str=action.old_str,
            new_str=action.new_str,
            insert_line=action.insert_line,
            view_range=action.view_range,
        )
        result = self._upstream(upstream_action)
        text, clipped = _truncate_text(result.text)
        return BoundedFileEditorObservation(
            content=[TextContent(text=text)],
            is_error=result.is_error,
            command=result.command,
            path=result.path,
            prev_exist=result.prev_exist,
            old_content=result.old_content,
            new_content=result.new_content,
            truncated=clipped,
        )

    def _view_file(self, action: BoundedFileEditorAction, path: Path) -> BoundedFileEditorObservation:
        if _is_binary_file(path):
            return self._observe_text(
                text=(
                    f"[file_editor:view]\npath: {path}\n"
                    "This file appears to be binary. Use `stat` instead of viewing raw bytes."
                ),
                command=action.command,
                path=path,
                is_error=True,
            )
        if path.stat().st_size > LARGE_FILE_BYTES and not action.view_range:
            return self._observe_text(
                text=(
                    f"[file_editor:view]\npath: {path}\n"
                    f"file_size_bytes: {path.stat().st_size}\n"
                    "This file is large. Use `stat`, `search`, or an explicit "
                    "`view_range` for targeted access."
                ),
                command=action.command,
                path=path,
                truncated=True,
            )

        lines = _read_text_lines(path)
        total_lines = len(lines)
        if total_lines == 0:
            return self._observe_text(
                text=f"[file_editor:view]\npath: {path}\nline_range: 0-0\ncontent:\n<empty file>",
                command=action.command,
                path=path,
                total_lines=0,
            )

        if action.view_range:
            if len(action.view_range) != 2 or not all(isinstance(i, int) for i in action.view_range):
                return self._observe_text(
                    text="`view_range` must be a two-element integer list.",
                    command=action.command,
                    path=path,
                    is_error=True,
                )
            start_line = max(1, action.view_range[0])
            end_line = total_lines if action.view_range[1] == -1 else min(total_lines, action.view_range[1])
        else:
            start_line = 1
            end_line = min(total_lines, FILE_VIEW_MAX_LINES)

        if start_line > total_lines:
            return self._observe_text(
                text=f"`view_range` starts at line {start_line}, but file only has {total_lines} lines.",
                command=action.command,
                path=path,
                total_lines=total_lines,
                is_error=True,
            )
        if end_line < start_line:
            return self._observe_text(
                text=f"`view_range` end line {end_line} must be >= start line {start_line}.",
                command=action.command,
                path=path,
                total_lines=total_lines,
                is_error=True,
            )

        bounded_end = min(end_line, start_line + FILE_VIEW_MAX_LINES - 1)
        next_view_range = None
        truncated = bounded_end < end_line
        if bounded_end < total_lines:
            next_view_range = [bounded_end + 1, min(total_lines, bounded_end + FILE_VIEW_MAX_LINES)]
        text = _format_line_block(path, start_line, bounded_end, lines[start_line - 1 : bounded_end])
        if truncated and next_view_range:
            text += f"\nnext_view_range: {next_view_range}"
        return self._observe_text(
            text=text,
            command=action.command,
            path=path,
            truncated=truncated,
            returned_line_start=start_line,
            returned_line_end=bounded_end,
            total_lines=total_lines,
            next_view_range=next_view_range,
        )

    def _search_file(self, action: BoundedFileEditorAction, path: Path) -> BoundedFileEditorObservation:
        if not action.query:
            return self._observe_text(
                text="`search` requires `query`.",
                command=action.command,
                path=path,
                is_error=True,
            )
        if _is_binary_file(path):
            return self._observe_text(
                text="`search` is only available for text files. Use `stat` for binary files.",
                command=action.command,
                path=path,
                is_error=True,
            )
        lines = _read_text_lines(path)
        snippets: list[str] = []
        match_count = 0
        compiled = re.compile(action.query) if action.regex else None
        for idx, line in enumerate(lines, start=1):
            matched = bool(compiled.search(line)) if compiled else action.query in line
            if not matched:
                continue
            match_count += 1
            if len(snippets) >= action.max_matches:
                continue
            start = max(1, idx - action.context_before)
            end = min(len(lines), idx + action.context_after)
            block = _format_line_block(path, start, end, lines[start - 1 : end])
            snippets.append(f"match {len(snippets)+1}:\n{block}")
        text_lines = [
            "[file_editor:search]",
            f"path: {path}",
            f"query: {action.query}",
            f"regex: {action.regex}",
            f"returned_matches: {min(match_count, action.max_matches)}",
            f"total_matches: {match_count}",
        ]
        if not snippets:
            text_lines.append("matches:\n<no matches>")
        else:
            text_lines.append("matches:")
            text_lines.extend(snippets)
        return self._observe_text(
            text="\n".join(text_lines),
            command=action.command,
            path=path,
            truncated=match_count > action.max_matches,
        )

    def _read_json(self, action: BoundedFileEditorAction, path: Path) -> BoundedFileEditorObservation:
        selectors = action.json_paths or []
        if not selectors:
            return self._observe_text(
                text="`read_json` requires `json_paths`.",
                command=action.command,
                path=path,
                is_error=True,
            )
        if _is_binary_file(path):
            return self._observe_text(
                text="`read_json` is only available for UTF-8 JSON text files.",
                command=action.command,
                path=path,
                is_error=True,
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return self._observe_text(
                text=f"Failed to parse JSON from {path}: {exc}",
                command=action.command,
                path=path,
                is_error=True,
            )
        values: dict[str, Any] = {}
        missing: list[str] = []
        for selector in selectors:
            try:
                values[selector] = _extract_json_value(payload, selector)
            except KeyError:
                missing.append(selector)
        text_lines = [
            "[file_editor:read_json]",
            f"path: {path}",
            "values:",
            json.dumps(values, ensure_ascii=False, indent=2),
        ]
        if missing:
            text_lines.append(f"missing_paths: {missing}")
        return self._observe_text(
            text="\n".join(text_lines),
            command=action.command,
            path=path,
            truncated=len(missing) > 0 and len(values) == 0,
        )

    def _stat_path(self, action: BoundedFileEditorAction, path: Path) -> BoundedFileEditorObservation:
        stat = path.stat()
        is_dir = path.is_dir()
        is_binary = False
        line_count = None
        if path.is_file():
            is_binary = _is_binary_file(path)
            if not is_binary:
                try:
                    line_count = len(_read_text_lines(path))
                except Exception:
                    line_count = None
        payload = {
            "path": str(path),
            "kind": "directory" if is_dir else "file",
            "size_bytes": stat.st_size,
            "mtime": int(stat.st_mtime),
            "is_binary": is_binary,
            "line_count": line_count,
            "sha256_short": hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12],
        }
        return self._observe_text(
            text="[file_editor:stat]\n" + json.dumps(payload, ensure_ascii=False, indent=2),
            command=action.command,
            path=path,
        )

    def __call__(
        self,
        action: BoundedFileEditorAction,
        conversation: "LocalConversation | None" = None,
    ) -> BoundedFileEditorObservation:
        try:
            path = _ensure_abs_path(action.path, self.workspace_root)
        except Exception as exc:
            return BoundedFileEditorObservation.from_text(
                text=str(exc),
                is_error=True,
                command=action.command,
                path=action.path,
            )

        if action.command in {"create", "str_replace", "insert", "undo_edit"}:
            return self._run_edit(action, path)

        if action.command == "list" or (action.command == "view" and path.is_dir()):
            text, truncated, omitted_entries = _render_directory_listing(
                path,
                max_depth=min(action.max_depth, 5),
                max_entries=min(action.max_entries, DIR_VIEW_MAX_ENTRIES),
                glob_pattern=action.glob,
            )
            return self._observe_text(
                text=text,
                command=action.command,
                path=path,
                truncated=truncated,
                omitted_entries=omitted_entries,
            )

        if not path.exists():
            return self._observe_text(
                text=f"Path does not exist: {path}",
                command=action.command,
                path=path,
                is_error=True,
            )

        if action.command == "stat":
            return self._stat_path(action, path)

        if not path.is_file():
            return self._observe_text(
                text=f"Command `{action.command}` requires a file path, got directory: {path}",
                command=action.command,
                path=path,
                is_error=True,
            )

        try:
            if action.command == "search":
                return self._search_file(action, path)
            if action.command == "read_json":
                return self._read_json(action, path)
            if action.command == "view":
                return self._view_file(action, path)
        except Exception as exc:
            return self._observe_text(
                text=f"{action.command} failed for {path}: {exc}",
                command=action.command,
                path=path,
                is_error=True,
            )

        return self._observe_text(
            text=f"Unsupported file_editor command: {action.command}",
            command=action.command,
            path=path,
            is_error=True,
        )


FILE_EDITOR_DESCRIPTION = """Bounded workspace file tool for targeted inspection and editing.

Use `list`, `stat`, `read_json`, or `search` before `view` whenever possible.
`view` returns a bounded line window only; if `view_range` is omitted, it returns the first window.
Directory listing returns a bounded number of entries.
Large or binary files will not be dumped wholesale into the model context.
Editing commands (`create`, `str_replace`, `insert`, `undo_edit`) behave like the standard editor and still require exact path and exact string matching.
"""


class BoundedFileEditorTool(ToolDefinition[BoundedFileEditorAction, BoundedFileEditorObservation]):
    """Experiment-specific bounded file editor."""

    name: ClassVar[str] = UpstreamFileEditorTool.name

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["BoundedFileEditorTool"]:
        executor = BoundedFileEditorExecutor(workspace_root=conv_state.workspace.working_dir)
        return [
            cls(
                action_type=BoundedFileEditorAction,
                observation_type=BoundedFileEditorObservation,
                description=(
                    FILE_EDITOR_DESCRIPTION
                    + f"\n\nCurrent working directory: {conv_state.workspace.working_dir}"
                ),
                annotations=ToolAnnotations(
                    title="file_editor",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


class BoundedTerminalObservation(TerminalObservation):
    """Terminal observation with bounded preview metadata."""

    output_truncated: bool = Field(default=False)
    raw_output_path: str | None = Field(default=None)

    @property
    def to_llm_content(self) -> Sequence[TextContent]:
        preview = self.text
        lines = [
            preview,
            f"[Command finished with exit code {self.exit_code}]",
        ]
        if self.metadata.working_dir:
            lines.append(f"[Current working directory: {self.metadata.working_dir}]")
        if self.raw_output_path:
            lines.append(f"[Raw output saved to: {self.raw_output_path}]")
        if self.output_truncated:
            lines.append("[Output truncated to fit the experiment-agent context budget.]")
        return [TextContent(text="\n".join(part for part in lines if part))]


def _parent_env_export_chunks() -> list[str]:
    exports: list[str] = []
    for key, value in os.environ.items():
        if not VALID_ENV_NAME.match(key):
            continue
        exports.append(f"export {key}={shlex.quote(value)}")
    chunk_size = 64
    return [" && ".join(exports[index : index + chunk_size]) for index in range(0, len(exports), chunk_size)]


class ExperimentTerminalExecutor(UpstreamTerminalExecutor):
    """Upstream terminal executor with explicit parent-env propagation."""

    def __init__(
        self,
        working_dir: str,
        username: str | None = None,
        no_change_timeout_seconds: int | None = None,
        terminal_type: Literal["tmux", "subprocess"] | None = None,
        shell_path: str | None = None,
        full_output_save_dir: str | None = None,
    ):
        super().__init__(
            working_dir=working_dir,
            username=username,
            no_change_timeout_seconds=no_change_timeout_seconds,
            terminal_type=terminal_type,
            shell_path=shell_path,
            full_output_save_dir=full_output_save_dir,
        )
        self._export_parent_envs()

    def _export_parent_envs(self) -> None:
        if not getattr(self, "session", None):
            return
        for command in _parent_env_export_chunks():
            if not command:
                continue
            _ = self.session.execute(
                TerminalAction(
                    command=command,
                    is_input=False,
                    timeout=10,
                )
            )

    def reset(self) -> TerminalObservation:
        observation = super().reset()
        self._export_parent_envs()
        return observation


class BoundedTerminalExecutor(ToolExecutor[TerminalAction, BoundedTerminalObservation]):
    """Wrap the upstream terminal executor with bounded observations."""

    def __init__(
        self,
        *,
        working_dir: str,
        username: str | None = None,
        no_change_timeout_seconds: int | None = None,
        terminal_type: Literal["tmux", "subprocess"] | None = None,
        shell_path: str | None = None,
        full_output_save_dir: str | None = None,
    ):
        self.full_output_save_dir = full_output_save_dir
        self._upstream = ExperimentTerminalExecutor(
            working_dir=working_dir,
            username=username,
            no_change_timeout_seconds=no_change_timeout_seconds,
            terminal_type=terminal_type,
            shell_path=shell_path,
            full_output_save_dir=full_output_save_dir,
        )

    def _preview_text(self, text: str, exit_code: int | None) -> tuple[str, bool]:
        lines = text.splitlines()
        truncated = False
        if len(lines) > TERMINAL_MAX_LINES:
            truncated = True
            if exit_code == 0:
                head = lines[:TERMINAL_HEAD_LINES]
                tail = lines[-TERMINAL_TAIL_LINES:] if TERMINAL_TAIL_LINES else []
                lines = [
                    *head,
                    "... <truncated middle output>",
                    *tail,
                ]
            else:
                lines = lines[-TERMINAL_MAX_LINES:]
                lines.insert(0, "... <truncated leading output>")
        rendered, clipped = _truncate_text("\n".join(lines))
        return rendered, truncated or clipped

    def __call__(
        self,
        action: TerminalAction,
        conversation: "LocalConversation | None" = None,
    ) -> BoundedTerminalObservation:
        upstream = self._upstream(action, conversation)
        raw_text = upstream.text or ""
        raw_path = _save_text_artifact(self.full_output_save_dir, "terminal", raw_text) if raw_text else None
        preview, truncated = self._preview_text(raw_text, upstream.exit_code)
        return BoundedTerminalObservation.from_text(
            text=preview,
            command=upstream.command,
            exit_code=upstream.exit_code,
            timeout=upstream.timeout,
            metadata=upstream.metadata,
            full_output_save_dir=upstream.full_output_save_dir,
            output_truncated=truncated,
            raw_output_path=raw_path,
            is_error=upstream.is_error,
        )

    def close(self) -> None:
        self._upstream.close()


TERMINAL_DESCRIPTION = """Execute one bash command in a persistent shell session with bounded observation previews.

The model receives only a capped preview of command output plus the saved raw output path.
Prefer targeted commands such as `rg`, `head`, `tail`, `jq`, or focused test invocations over broad `cat`, recursive dumps, or full-log prints.
If output is long, only a bounded preview is returned and the full raw output is saved on disk.

""" + UPSTREAM_TERMINAL_DESCRIPTION


class BoundedTerminalTool(ToolDefinition[TerminalAction, BoundedTerminalObservation]):
    """Experiment-specific bounded terminal tool."""

    name: ClassVar[str] = UpstreamTerminalTool.name

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        username: str | None = None,
        no_change_timeout_seconds: int | None = None,
        terminal_type: Literal["tmux", "subprocess"] | None = None,
        shell_path: str | None = None,
    ) -> Sequence["BoundedTerminalTool"]:
        executor = BoundedTerminalExecutor(
            working_dir=conv_state.workspace.working_dir,
            username=username,
            no_change_timeout_seconds=no_change_timeout_seconds,
            terminal_type=terminal_type,
            shell_path=shell_path,
            full_output_save_dir=conv_state.env_observation_persistence_dir,
        )
        return [
            cls(
                action_type=TerminalAction,
                observation_type=BoundedTerminalObservation,
                description=TERMINAL_DESCRIPTION,
                annotations=ToolAnnotations(
                    title="terminal",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


_OVERRIDES_ENABLED = False


def enable_experiment_tool_overrides() -> None:
    """Override OpenHands tool registrations with bounded experiment variants."""
    global _OVERRIDES_ENABLED
    if _OVERRIDES_ENABLED:
        return
    register_tool(BoundedFileEditorTool.name, BoundedFileEditorTool)
    register_tool(BoundedTerminalTool.name, BoundedTerminalTool)
    enable_resource_tools()
    _OVERRIDES_ENABLED = True


__all__ = [
    "BoundedFileEditorAction",
    "BoundedFileEditorObservation",
    "BoundedFileEditorTool",
    "BoundedTerminalObservation",
    "BoundedTerminalTool",
    "enable_experiment_tool_overrides",
]
