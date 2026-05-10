from __future__ import annotations

import asyncio
import json
import os
import subprocess
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


@dataclass
class ClaudeInvocation:
    argv: List[str]
    cwd: str
    agent_name: str


_HELP_CACHE: Dict[str, str] = {}


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
                stripped = result_value.strip()
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
        self.strict_mcp_config = bool(cfg.get("strict_mcp_config", False))
        self.no_session_persistence = bool(cfg.get("no_session_persistence", True))
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

    async def _run_streaming(
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
        exec_argv = self._argv_with_prompt(
            ClaudeInvocation(argv=stream_argv, cwd=invocation.cwd, agent_name=invocation.agent_name),
            prompt,
        )

        proc = await asyncio.create_subprocess_exec(
            *exec_argv,
            cwd=invocation.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            duration_ms = int((time.time() - started) * 1000)
            self._write_trace(
                base_path=trace_base,
                invocation=invocation,
                prompt=prompt,
                stdout_text="",
                stderr_text=f"TIMEOUT after {self.timeout_seconds} seconds",
                returncode=-1,
                duration_ms=duration_ms,
                stream_events=[],
            )
            raise ClaudeCodeError(
                f"Claude Code timed out after {self.timeout_seconds} seconds. Trace: {trace_base}.meta.json"
            ) from exc

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        returncode = proc.returncode
        duration_ms = int((time.time() - started) * 1000)

        # Parse stream into events
        final_result_json, events = _parse_stream_jsonl(stdout_text)

        self._write_trace(
            base_path=trace_base,
            invocation=invocation,
            prompt=prompt,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            returncode=returncode,
            duration_ms=duration_ms,
            stream_events=events,
        )

        if returncode != 0:
            raise ClaudeCodeError(
                f"Claude Code exited with {returncode}.\nSTDOUT:\n{stdout_text[:2000]}\nSTDERR:\n{stderr_text}\nTrace: {trace_base}.meta.json"
            )

        return final_result_json, events, stdout_text, stderr_text, returncode

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
            try:
                result_event = json.loads(final_result_json)
                return _coerce_json_output(json.dumps(result_event))
            except (json.JSONDecodeError, ClaudeCodeError):
                pass
        # Fallback: parse raw stdout
        return _coerce_json_output(_stdout_text)
