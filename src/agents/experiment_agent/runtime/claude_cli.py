from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.experiment_agent.config import get_claude_code_config
from src.agents.experiment_agent.runtime.manifests import ensure_claude_trace_paths


class ClaudeCodeError(RuntimeError):
    pass


_STREAM_SENTINEL_TYPES = {"result", "error", "exception"}


def _parse_stream_jsonl(stream_text: str) -> tuple[str, list[Dict[str, Any]]]:
    """Parse stream-json output into (final_result_json, all_events).

    Returns the last 'result' or 'error' event JSON as the final result,
    and the complete list of all events for replay/debugging.
    """
    events: list[Dict[str, Any]] = []
    final_result = ""
    for line in stream_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(event)
        event_type = str(event.get("type", ""))
        if event_type in _STREAM_SENTINEL_TYPES:
            final_result = line
    return final_result, events


def _claude_failure_summary(
    stdout_text: str,
    stderr_text: str,
    events: list[Dict[str, Any]],
) -> str:
    for event in reversed(events):
        for key in ("result", "error", "message"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    text = (stderr_text or stdout_text or "").strip()
    return text[-2000:] if text else ""


def _is_retryable_transport_failure(summary: str) -> bool:
    lowered = str(summary or "").lower()
    retryable_markers = (
        "api error:",
        "socket connection was closed",
        "fetch()",
        "fetch failed",
        "connection reset",
        "econnreset",
        "etimedout",
        "network error",
        "upstream",
    )
    return any(marker in lowered for marker in retryable_markers)


@dataclass
class ClaudeInvocation:
    argv: List[str]
    cwd: str
    agent_name: str


class ClaudeStreamPrettyRenderer:
    """Render Claude Code stream-json events as readable terminal progress."""

    def __init__(
        self,
        *,
        agent_name: str,
        mode: str = "off",
        output: str = "stderr",
        detail: str = "compact",
    ) -> None:
        self.agent_name = str(agent_name or "default")
        self.mode = str(mode or "off").strip().lower()
        self.detail = str(detail or "compact").strip().lower()
        self.file = sys.stdout if str(output).strip().lower() == "stdout" else sys.stderr
        self._rich_console = None
        self._rich_escape = None
        if self.mode == "rich":
            try:
                from rich.console import Console
                from rich.markup import escape
                from rich.theme import Theme

                theme = Theme(
                    {
                        "agent": "bold cyan",
                        "thinking": "dim italic",
                        "tool": "bold blue",
                        "tool.ok": "green",
                        "tool.err": "bold red",
                        "path": "magenta",
                        "cmd": "yellow",
                        "meta": "dim",
                        "done": "bold green",
                        "error": "bold red",
                    }
                )
                self._rich_console = Console(
                    file=self.file,
                    force_terminal=True,
                    color_system="auto",
                    theme=theme,
                )
                self._rich_escape = escape
            except Exception:
                self.mode = "plain"

    @property
    def enabled(self) -> bool:
        return self.mode in {"plain", "rich"}

    def render(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        event_type = str(event.get("type") or "")
        if event_type == "system":
            self._render_system(event)
        elif event_type == "assistant":
            self._render_assistant(event.get("message") if isinstance(event.get("message"), dict) else {})
        elif event_type == "user":
            self._render_user(event.get("message") if isinstance(event.get("message"), dict) else {})
        elif event_type == "result":
            status = "failed" if event.get("is_error") else "done"
            style = "error" if event.get("is_error") else "done"
            self._markup(
                f"[{style}]{self._esc(self.agent_name)} Claude Code {status}[/{style}]"
            )
        elif event_type in {"error", "exception"} or event.get("error"):
            message = str(event.get("error") or event.get("message") or "")[:800]
            self._markup(
                f"[error]{self._esc(self.agent_name)} ERROR:[/error] {self._esc(message)}"
            )

    def _render_system(self, event: Dict[str, Any]) -> None:
        subtype = str(event.get("subtype") or "")
        if subtype == "init":
            self._markup(
                f"[agent]{self._esc(self.agent_name)}[/agent] "
                f"[meta]started model={self._esc(event.get('model'))} "
                f"cwd={self._esc(event.get('cwd'))}[/meta]"
            )
        elif subtype == "api_retry":
            self._markup(
                f"[agent]{self._esc(self.agent_name)}[/agent] "
                f"[error]API retry[/error] "
                f"[meta]{self._esc(event.get('attempt'))}/{self._esc(event.get('max_retries'))}[/meta] "
                f"{self._esc(event.get('error'))}"
            )

    def _render_assistant(self, message: Dict[str, Any]) -> None:
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type == "thinking":
                text = str(block.get("thinking") or "").strip()
                if text:
                    self._markup(f"[thinking]thinking: {self._esc(self._short(text, 500))}[/thinking]")
            elif block_type == "text":
                text = str(block.get("text") or "").rstrip()
                if text:
                    self._text(text)
            elif block_type == "tool_use":
                self._render_tool_use(block)

    def _render_user(self, message: Dict[str, Any]) -> None:
        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            label = "tool_error" if block.get("is_error") else "tool_result"
            style = "tool.err" if block.get("is_error") else "tool.ok"
            tool_id = str(block.get("tool_use_id") or "")
            content = self._short(block.get("content"), 1600 if self.detail == "full" else 700)
            if self.mode == "rich":
                self._rich_panel(content, title=f"[{style}]{label}[/] [meta]{self._esc(tool_id)}[/meta]", error=bool(block.get("is_error")))
            else:
                self._plain(f"[{self.agent_name}] {label} {tool_id}: {content}")

    def _render_tool_use(self, block: Dict[str, Any]) -> None:
        name = str(block.get("name") or "tool")
        tool_id = str(block.get("id") or "")
        tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
        if name == "Bash":
            command = str(tool_input.get("command") or "")
            description = str(tool_input.get("description") or "").strip()
            suffix = f" {description}" if description else ""
            self._markup(
                f"[tool]Bash[/tool] [meta]{self._esc(tool_id)}[/meta]{self._esc(suffix)}"
            )
            if command:
                self._syntax(command, "bash")
            return
        if name in {"Read", "Write", "Edit"}:
            path = str(tool_input.get("file_path") or tool_input.get("path") or "")
            self._markup(
                f"[tool]{self._esc(name)}[/tool] "
                f"[path]{self._esc(path)}[/path] [meta]{self._esc(tool_id)}[/meta]"
            )
            return
        preview = self._short(tool_input, 700 if self.detail == "full" else 260)
        self._markup(
            f"[tool]{self._esc(name)}[/tool] [meta]{self._esc(tool_id)}[/meta] "
            f"{self._esc(preview)}"
        )

    def _short(self, value: Any, limit: int = 300) -> str:
        if isinstance(value, str):
            text = value.strip()
        else:
            try:
                text = json.dumps(value, ensure_ascii=False)
            except TypeError:
                text = str(value)
        if self.detail != "full":
            text = " ".join(text.split())
        return text[:limit] + ("..." if len(text) > limit else "")

    def _esc(self, value: Any) -> str:
        text = "" if value is None else str(value)
        if self._rich_escape:
            return self._rich_escape(text)
        return text

    def _text(self, text: str) -> None:
        if self.mode == "rich" and self._rich_console is not None:
            self._rich_console.print(text, markup=False)
        else:
            self._plain(text)

    def _markup(self, markup: str) -> None:
        if self.mode == "rich" and self._rich_console is not None:
            self._rich_console.print(markup)
        else:
            clean = re.sub(r"\[/?[A-Za-z0-9_. ]+\]", "", markup)
            self._plain(clean)

    def _plain(self, text: str) -> None:
        print(text, file=self.file, flush=True)

    def _syntax(self, code: str, lexer: str) -> None:
        if self.mode == "rich" and self._rich_console is not None:
            try:
                from rich.syntax import Syntax

                self._rich_console.print(
                    Syntax(code, lexer, theme="ansi_dark", word_wrap=True)
                )
                return
            except Exception:
                pass
        self._plain(code)

    def _rich_panel(self, content: str, *, title: str, error: bool = False) -> None:
        if self._rich_console is None:
            self._plain(content)
            return
        try:
            from rich.panel import Panel
            from rich.text import Text

            self._rich_console.print(
                Panel(
                    Text(content),
                    title=title,
                    border_style="red" if error else "green",
                )
            )
        except Exception:
            self._plain(content)


_HELP_CACHE: Dict[str, str] = {}


def _read_settings_env(settings_path: str) -> Dict[str, str]:
    path = os.path.expanduser(str(settings_path or ""))
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    env = payload.get("env") if isinstance(payload, dict) else None
    if not isinstance(env, dict):
        return {}
    return {str(key): str(value) for key, value in env.items() if str(value)}


def _claude_subprocess_env(settings_paths: Optional[List[str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    for settings_path in settings_paths or []:
        for key, value in _read_settings_env(settings_path).items():
            if key.startswith("ANTHROPIC_"):
                env[key] = value
    api_key = str(env.get("ANTHROPIC_API_KEY") or "").strip()
    auth_token = str(env.get("ANTHROPIC_AUTH_TOKEN") or "").strip()
    if api_key and not auth_token:
        env["ANTHROPIC_AUTH_TOKEN"] = api_key
    elif auth_token and not api_key:
        env["ANTHROPIC_API_KEY"] = auth_token
    return env


def _extract_json_object_text(text: str) -> str:
    stripped = (text or "").strip()
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL):
        candidate = match.group(1).strip()
        if candidate.startswith("{"):
            return candidate
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(stripped)):
            char = stripped[idx]
            if escape:
                escape = False
                continue
            if in_string and char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return stripped[start : idx + 1]
    return stripped


def _coerce_json_output(stdout: str) -> Dict[str, Any]:
    def _coerce_payload(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, dict) and "structured_output" in payload:
            structured = payload["structured_output"]
            if isinstance(structured, dict):
                return structured
        if isinstance(payload, dict) and "result" in payload:
            result_value = payload["result"]
            if isinstance(result_value, dict):
                return result_value
            if isinstance(result_value, str):
                stripped = _extract_json_object_text(result_value)
                if stripped.startswith("{"):
                    return _coerce_json_output(stripped)
        if isinstance(payload, dict) and "content" in payload and isinstance(payload["content"], dict):
            return payload["content"]
        if isinstance(payload, dict):
            return payload
        raise ClaudeCodeError("Claude Code JSON output must be an object.")

    text = (stdout or "").strip()
    if not text:
        raise ClaudeCodeError("Claude Code returned empty JSON output.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ClaudeCodeError(f"Claude Code returned invalid JSON: {text[:500]}") from exc
    return _coerce_payload(payload)


class ClaudeCodeRunner:
    def __init__(
        self,
        *,
        model: str,
        workspace_root: str,
        verbose: bool = False,
        binary: Optional[str] = None,
    ):
        cfg = get_claude_code_config()
        self.model = model
        self.workspace_root = os.path.realpath(workspace_root)
        self.verbose = verbose
        self.binary = binary or cfg["binary"]
        self.permission_mode = str(cfg.get("permission_mode") or "bypassPermissions")
        self.dangerously_skip_permissions = bool(
            cfg.get("dangerously_skip_permissions", True)
        )
        self.use_bare = bool(cfg.get("use_bare", False))
        self.timeout_seconds = int(cfg.get("timeout_seconds") or 1800)
        self.mcp_config_path = str(cfg.get("mcp_config_path") or "").strip()
        self.settings_sources = str(cfg.get("settings_sources") or "project").strip()
        self.global_settings_path = str(cfg.get("global_settings_path") or "").strip()
        self.strict_mcp_config = bool(cfg.get("strict_mcp_config", False))
        self.no_session_persistence = bool(cfg.get("no_session_persistence", True))
        self.debug_filter = str(cfg.get("debug_filter") or "").strip()
        self.debug_file = str(cfg.get("debug_file") or "").strip()
        self.stream_renderer = str(cfg.get("stream_renderer") or "off").strip().lower()
        self.stream_renderer_output = str(
            cfg.get("stream_renderer_output") or "stderr"
        ).strip().lower()
        self.stream_renderer_detail = str(
            cfg.get("stream_renderer_detail") or "compact"
        ).strip().lower()
        trace_paths = ensure_claude_trace_paths(self.workspace_root)
        self.trace_dir = trace_paths["trace_dir"]
        self.trace_latest_path = trace_paths["latest_path"]
        self.trace_index_path = trace_paths["index_path"]
        self.trace_errors_path = trace_paths["errors_path"]
        self.trace_report_path = trace_paths["report_path"]
        if self.verbose:
            print(f"[claude-trace] dir={self.trace_dir}")

    def _supports_option(self, option_name: str) -> bool:
        if self.binary not in _HELP_CACHE:
            try:
                proc = subprocess.run(
                    [self.binary, "--help"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=10,
                )
                _HELP_CACHE[self.binary] = (proc.stdout or "") + "\n" + (proc.stderr or "")
            except Exception:
                _HELP_CACHE[self.binary] = ""
        return option_name in _HELP_CACHE[self.binary]

    def _resolve_mcp_config_path(self) -> str:
        if self.mcp_config_path:
            return self.mcp_config_path
        workspace_mcp = os.path.join(self.workspace_root, ".mcp.json")
        if os.path.exists(workspace_mcp):
            return workspace_mcp
        return ""

    def _normalize_agent_name(self, agent_name: Optional[str]) -> str:
        return str(agent_name or "").strip().replace("_", "-")

    def _base_argv(
        self,
        *,
        agent_name: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> ClaudeInvocation:
        _ = cwd
        workdir = self.workspace_root
        argv = [
            self.binary,
            "-p",
            "--model",
            self.model,
            "--permission-mode",
            self.permission_mode,
            "--output-format",
            "text",
            "--add-dir",
            self.workspace_root,
        ]
        normalized_agent = self._normalize_agent_name(agent_name)
        if normalized_agent and self._supports_option("--agent"):
            argv.extend(["--agent", normalized_agent])
        if self.use_bare and self._supports_option("--bare"):
            argv.insert(2, "--bare")
        if (
            self.dangerously_skip_permissions
            and self._supports_option("--dangerously-skip-permissions")
        ):
            argv.extend(["--dangerously-skip-permissions"])
        if self.no_session_persistence and self._supports_option("--no-session-persistence"):
            argv.append("--no-session-persistence")
        if self.settings_sources and self._supports_option("--setting-sources"):
            argv.extend(["--setting-sources", self.settings_sources])
        if self.debug_filter and self._supports_option("--debug"):
            argv.extend(["--debug", self.debug_filter])
        if self.debug_file and self._supports_option("--debug-file"):
            debug_file = self.debug_file
            if not os.path.isabs(debug_file):
                debug_file = os.path.join(self.workspace_root, debug_file)
            os.makedirs(os.path.dirname(debug_file), exist_ok=True)
            argv.extend(["--debug-file", debug_file])
        mcp_config_path = self._resolve_mcp_config_path()
        if mcp_config_path:
            argv.extend(["--mcp-config", mcp_config_path])
            if self.strict_mcp_config and self._supports_option("--strict-mcp-config"):
                argv.append("--strict-mcp-config")
        return ClaudeInvocation(argv=argv, cwd=workdir, agent_name=normalized_agent or "default")

    def _argv_with_prompt(self, invocation: ClaudeInvocation, prompt: str) -> List[str]:
        argv = list(invocation.argv)
        try:
            prompt_index = argv.index("-p") + 1
        except ValueError:
            prompt_index = len(argv)
        argv.insert(prompt_index, prompt)
        return argv

    def _trace_base_path(self, invocation: ClaudeInvocation) -> str:
        os.makedirs(self.trace_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_agent = invocation.agent_name.replace("/", "-")
        return os.path.join(self.trace_dir, f"{stamp}-{safe_agent}-{uuid.uuid4().hex[:8]}")

    def _append_trace_index(self, meta: Dict[str, Any]) -> None:
        os.makedirs(self.trace_dir, exist_ok=True)
        with open(self.trace_index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        with open(self.trace_latest_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        report_payload = {
            "trace_dir": self.trace_dir,
            "latest_path": self.trace_latest_path,
            "index_path": self.trace_index_path,
            "errors_path": self.trace_errors_path,
            "latest": meta,
        }
        with open(self.trace_report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)

    def _record_trace_error(self, message: str) -> None:
        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            with open(self.trace_errors_path, "a", encoding="utf-8") as f:
                f.write(message.rstrip() + "\n")
        except Exception:
            return

    def _maybe_log_stream_event(self, line: str) -> None:
        if not self.verbose:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        event_type = str(event.get("type") or "")
        subtype = str(event.get("subtype") or "")
        if event_type == "system" and subtype == "api_retry":
            attempt = event.get("attempt")
            max_retries = event.get("max_retries")
            status = event.get("error_status")
            error = event.get("error")
            delay_ms = event.get("retry_delay_ms")
            print(
                "[claude-api-retry] "
                f"attempt={attempt}/{max_retries} status={status} "
                f"error={error} delay_ms={int(delay_ms or 0)}",
                flush=True,
            )
        elif event_type in {"error", "exception"}:
            message = str(event.get("error") or event.get("message") or "")[:500]
            print(f"[claude-error] {message}", flush=True)
        elif event.get("error"):
            message = str(event.get("error") or "")[:500]
            print(f"[claude-error] type={event_type} {message}", flush=True)
        elif event_type == "result":
            subtype_note = f" subtype={subtype}" if subtype else ""
            error_note = " is_error=true" if event.get("is_error") else ""
            print(f"[claude-result]{subtype_note}{error_note}", flush=True)

    async def _read_stream(
        self,
        stream: Optional[asyncio.StreamReader],
        chunks: List[str],
        *,
        log_events: bool = False,
        renderer: Optional[ClaudeStreamPrettyRenderer] = None,
        raw_passthrough: bool = False,
        raw_output: str = "stdout",
    ) -> None:
        if stream is None:
            return
        pending = ""
        while True:
            data = await stream.read(65536)
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            chunks.append(text)
            if raw_passthrough:
                raw_file = sys.stderr if raw_output == "stderr" else sys.stdout
                print(text, end="", file=raw_file, flush=True)
            if log_events:
                pending += text
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    stripped = line.strip()
                    self._maybe_log_stream_event(stripped)
                    if renderer and renderer.enabled:
                        try:
                            event = json.loads(stripped)
                        except json.JSONDecodeError:
                            continue
                        renderer.render(event)
        if log_events and pending.strip():
            stripped = pending.strip()
            self._maybe_log_stream_event(stripped)
            if renderer and renderer.enabled:
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    return
                renderer.render(event)

    def _write_trace(
        self,
        *,
        base_path: str,
        invocation: ClaudeInvocation,
        prompt: str,
        stdout_text: str,
        stderr_text: str,
        returncode: int,
        duration_ms: int,
        stream_events: Optional[list[Dict[str, Any]]] = None,
    ) -> None:
        os.makedirs(self.trace_dir, exist_ok=True)
        meta = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "agent_name": invocation.agent_name,
            "argv": invocation.argv,
            "cwd": invocation.cwd,
            "returncode": returncode,
            "duration_ms": duration_ms,
            "meta_path": f"{base_path}.meta.json",
            "prompt_path": f"{base_path}.prompt.txt",
            "stdout_path": f"{base_path}.stdout.txt",
            "stderr_path": f"{base_path}.stderr.txt",
        }
        if stream_events:
            meta["stream_path"] = f"{base_path}.stream.jsonl"
            meta["stream_event_count"] = len(stream_events)
        try:
            with open(f"{base_path}.meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            with open(f"{base_path}.prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt)
            with open(f"{base_path}.stdout.txt", "w", encoding="utf-8") as f:
                f.write(stdout_text)
            with open(f"{base_path}.stderr.txt", "w", encoding="utf-8") as f:
                f.write(stderr_text)
            if stream_events:
                stream_path = f"{base_path}.stream.jsonl"
                with open(stream_path, "w", encoding="utf-8") as f:
                    for event in stream_events:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._append_trace_index(meta)
        except Exception as exc:
            self._record_trace_error(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} trace_write_failed agent={invocation.agent_name} error={exc}"
            )
            if self.verbose:
                print(f"[claude-trace-warning] agent={invocation.agent_name} error={exc}")
        if self.verbose:
            stream_note = f" events={len(stream_events)}" if stream_events else ""
            print(
                f"[claude-trace] agent={invocation.agent_name} trace={base_path}.meta.json returncode={returncode} duration_ms={duration_ms}{stream_note}"
            )

    async def _run_streaming_once(
        self,
        invocation: ClaudeInvocation,
        prompt: str,
    ) -> tuple[str, list[Dict[str, Any]], str, str, int]:
        """Run Claude Code with --output-format stream-json and capture full conversation.

        Returns (final_result_json, all_events, stdout_text, stderr_text, returncode).
        Also writes the stream to a .stream.jsonl trace file.
        """
        trace_base = self._trace_base_path(invocation)
        started = time.time()

        # Build stream-json argv from the invocation
        stream_argv = list(invocation.argv)
        # Replace or add output-format
        try:
            oi = stream_argv.index("--output-format")
            stream_argv[oi + 1] = "stream-json"
        except ValueError:
            stream_argv.extend(["--output-format", "stream-json"])

        # stream-json requires --verbose in --print mode
        if self._supports_option("--verbose") and "--verbose" not in stream_argv:
            stream_argv.append("--verbose")

        # Add json-schema if provided
        trace_invocation = ClaudeInvocation(
            argv=stream_argv,
            cwd=invocation.cwd,
            agent_name=invocation.agent_name,
        )
        exec_argv = self._argv_with_prompt(trace_invocation, prompt)

        proc = await asyncio.create_subprocess_exec(
            *exec_argv,
            cwd=invocation.cwd,
            env=_claude_subprocess_env(
                [
                    self.global_settings_path,
                    os.path.join(self.workspace_root, ".claude", "settings.json"),
                ]
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []
        renderer = ClaudeStreamPrettyRenderer(
            agent_name=invocation.agent_name,
            mode=self.stream_renderer,
            output=self.stream_renderer_output,
            detail=self.stream_renderer_detail,
        )
        raw_passthrough = self.stream_renderer == "raw"
        stdout_task = asyncio.create_task(
            self._read_stream(
                proc.stdout,
                stdout_chunks,
                log_events=True,
                renderer=renderer,
                raw_passthrough=raw_passthrough,
                raw_output=self.stream_renderer_output,
            )
        )
        stderr_task = asyncio.create_task(
            self._read_stream(
                proc.stderr,
                stderr_chunks,
                raw_passthrough=raw_passthrough,
                raw_output=self.stream_renderer_output,
            )
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.timeout_seconds)
            await asyncio.gather(stdout_task, stderr_task)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            duration_ms = int((time.time() - started) * 1000)
            stdout_text = "".join(stdout_chunks)
            stderr_text = "".join(stderr_chunks)
            self._write_trace(
                base_path=trace_base,
                invocation=trace_invocation,
                prompt=prompt,
                stdout_text=stdout_text,
                stderr_text=stderr_text
                + f"\nTIMEOUT after {self.timeout_seconds} seconds",
                returncode=-1,
                duration_ms=duration_ms,
                stream_events=[],
            )
            raise ClaudeCodeError(
                f"Claude Code timed out after {self.timeout_seconds} seconds. Trace: {trace_base}.meta.json"
            ) from exc

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)
        returncode = proc.returncode
        duration_ms = int((time.time() - started) * 1000)

        # Parse stream into events
        final_result_json, events = _parse_stream_jsonl(stdout_text)

        self._write_trace(
            base_path=trace_base,
            invocation=trace_invocation,
            prompt=prompt,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            returncode=returncode,
            duration_ms=duration_ms,
            stream_events=events,
        )

        if returncode != 0:
            summary = _claude_failure_summary(stdout_text, stderr_text, events)
            raise ClaudeCodeError(
                f"Claude Code exited with {returncode}.\n"
                f"Reason: {summary[:2000]}\n"
                f"STDOUT tail:\n{stdout_text[-2000:]}\n"
                f"STDERR:\n{stderr_text}\n"
                f"Trace: {trace_base}.meta.json"
            )

        return final_result_json, events, stdout_text, stderr_text, returncode

    async def _run_streaming(
        self,
        invocation: ClaudeInvocation,
        prompt: str,
    ) -> tuple[str, list[Dict[str, Any]], str, str, int]:
        max_attempts = max(1, int(os.environ.get("CLAUDE_CODE_TRANSPORT_ATTEMPTS", "3") or "3"))
        last_error: Optional[ClaudeCodeError] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return await self._run_streaming_once(invocation, prompt)
            except ClaudeCodeError as exc:
                last_error = exc
                summary = str(exc)
                if attempt >= max_attempts or not _is_retryable_transport_failure(summary):
                    raise
                delay_seconds = min(30, 2 ** attempt)
                print(
                    f"[claude-transport-retry] agent={invocation.agent_name} "
                    f"attempt={attempt}/{max_attempts} delay_s={delay_seconds} "
                    f"reason={summary.splitlines()[1][:240] if len(summary.splitlines()) > 1 else summary[:240]}",
                    flush=True,
                )
                await asyncio.sleep(delay_seconds)
        raise last_error or ClaudeCodeError("Claude Code failed without an error detail.")

    async def _run(self, invocation: ClaudeInvocation, prompt: str) -> str:
        """Legacy text mode: run and return the text result."""
        final_result_json, events, stdout_text, _stderr, _rc = await self._run_streaming(
            invocation, prompt
        )
        if final_result_json:
            try:
                result_event = json.loads(final_result_json)
                result = result_event.get("result", "")
                if isinstance(result, str):
                    return result.strip()
            except json.JSONDecodeError:
                pass
        # Fallback: use raw stdout
        return stdout_text.strip()

    async def run_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> str:
        invocation = self._base_argv(agent_name=agent_name, cwd=cwd)
        prompt = "\n\n".join(
            part for part in [f"System instructions:\n{system_prompt}".strip(), user_prompt.strip()] if part
        )
        return (await self._run(invocation, prompt)).strip()

    async def run_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: Optional[str] = None,
        output_schema: Dict[str, Any] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        invocation = self._base_argv(agent_name=agent_name, cwd=cwd)
        if output_schema:
            invocation.argv.extend(["--json-schema", json.dumps(output_schema, ensure_ascii=False)])
        prompt = "\n\n".join(
            part for part in [f"System instructions:\n{system_prompt}".strip(), user_prompt.strip()] if part
        )
        final_result_json, events, _stdout_text, _stderr, _rc = await self._run_streaming(
            invocation, prompt
        )
        if final_result_json:
            result_event = json.loads(final_result_json)
            return _coerce_json_output(json.dumps(result_event))
        # Fallback: parse raw stdout
        return _coerce_json_output(_stdout_text)
