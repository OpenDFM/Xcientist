"""OpenHarness-backed runner for experiment-agent workers and reviewers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.agents.experiment_agent.runtime.openharness_vendor import ensure_vendored_openharness_path


ensure_vendored_openharness_path()

from openharness.hooks.types import HookResult


_STRUCTURED_OUTPUT_BLOCK_MESSAGE = "Xcientist structured-output hook blocked completion"
_DEFAULT_STRUCTURED_OUTPUT_MAX_HOOK_BLOCKS = 6


def _load_openharness_symbols():
    from openharness.api.openai_client import OpenAICompatibleClient
    from openharness.config.settings import (
        MemorySettings,
        PermissionSettings,
        Settings,
    )
    from openharness.engine.query_engine import QueryEngine
    from openharness.engine.stream_events import (
        AssistantTextDelta,
        AssistantTurnComplete,
        ErrorEvent,
        StatusEvent,
        ToolExecutionCompleted,
        ToolExecutionStarted,
    )
    from openharness.permissions.checker import PermissionChecker
    from openharness.permissions.modes import PermissionMode
    from openharness.mcp.client import McpClientManager
    from openharness.mcp.types import McpHttpServerConfig
    from openharness.tools import create_default_tool_registry
    from openharness.tools.base import ToolRegistry
    from openharness.tools.bash_tool import BashTool
    from openharness.tools.file_edit_tool import FileEditTool
    from openharness.tools.file_read_tool import FileReadTool
    from openharness.tools.file_write_tool import FileWriteTool
    from openharness.tools.glob_tool import GlobTool
    from openharness.tools.grep_tool import GrepTool

    return {
        "OpenAICompatibleClient": OpenAICompatibleClient,
        "MemorySettings": MemorySettings,
        "PermissionSettings": PermissionSettings,
        "Settings": Settings,
        "QueryEngine": QueryEngine,
        "AssistantTextDelta": AssistantTextDelta,
        "AssistantTurnComplete": AssistantTurnComplete,
        "ErrorEvent": ErrorEvent,
        "StatusEvent": StatusEvent,
        "ToolExecutionCompleted": ToolExecutionCompleted,
        "ToolExecutionStarted": ToolExecutionStarted,
        "PermissionChecker": PermissionChecker,
        "PermissionMode": PermissionMode,
        "McpClientManager": McpClientManager,
        "McpHttpServerConfig": McpHttpServerConfig,
        "create_default_tool_registry": create_default_tool_registry,
        "ToolRegistry": ToolRegistry,
        "BashTool": BashTool,
        "FileEditTool": FileEditTool,
        "FileReadTool": FileReadTool,
        "FileWriteTool": FileWriteTool,
        "GlobTool": GlobTool,
        "GrepTool": GrepTool,
    }


def ensure_openharness_runtime_env(workspace_root: str) -> Dict[str, str]:
    """Pin OpenHarness runtime state to the experiment workspace."""
    root = Path(workspace_root).expanduser().resolve() / ".openharness_runtime"
    config_dir = root / "config"
    data_dir = root / "data"
    logs_dir = root / "logs"
    for path in (config_dir, data_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    updates = {
        "OPENHARNESS_CONFIG_DIR": str(config_dir),
        "OPENHARNESS_DATA_DIR": str(data_dir),
        "OPENHARNESS_LOGS_DIR": str(logs_dir),
    }
    os.environ.update(updates)
    return updates


def _write_mcp_status(workspace_root: str, payload: Dict[str, Any]) -> str:
    path = Path(workspace_root).expanduser().resolve() / "agent_reports" / "_runtime" / "mcp_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    status = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    }
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        candidates = _extract_json_object_candidates(stripped)
        if len(candidates) == 1:
            return candidates[0][2]
        if len(candidates) > 1:
            raise ValueError(
                "OpenHarness JSON response must contain exactly one unambiguous JSON object; "
                f"found {len(candidates)} top-level JSON object candidates."
            ) from exc
        raise ValueError(
            "OpenHarness JSON response must contain exactly one JSON object. "
            "Markdown fences or prose are accepted only when they wrap a single unambiguous JSON object."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("OpenHarness JSON response must be an object.")
    return payload


def _extract_json_object_candidates(text: str) -> list[tuple[int, int, Dict[str, Any]]]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, int, Dict[str, Any]]] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, relative_end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            candidates.append((start, start + int(relative_end), payload))

    outer_candidates: list[tuple[int, int, Dict[str, Any]]] = []
    seen_ranges: set[tuple[int, int]] = set()
    for index, candidate in enumerate(candidates):
        start, end, _ = candidate
        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))
        contained = False
        for other_index, other in enumerate(candidates):
            if other_index == index:
                continue
            other_start, other_end, _ = other
            if other_start <= start and end <= other_end and (other_start, other_end) != (start, end):
                contained = True
                break
        if not contained:
            outer_candidates.append(candidate)
    return outer_candidates


def extract_json_object(text: str) -> Dict[str, Any]:
    return _extract_json_object(text)


def _schema_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_json_schema_fragment(value: Any, schema: Dict[str, Any], *, path: str = "$") -> list[str]:
    issues: list[str] = []
    expected_type = schema.get("type")
    expected_types = expected_type if isinstance(expected_type, list) else [expected_type] if expected_type else []
    if expected_types and not any(_json_type_matches(value, str(item)) for item in expected_types):
        rendered = " or ".join(str(item) for item in expected_types)
        issues.append(f"{path} must be {rendered}, got {_schema_type_name(value)}.")
        return issues

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        issues.append(f"{path} must be one of {enum_values!r}, got {value!r}.")

    if isinstance(value, dict):
        required = schema.get("required") or []
        if isinstance(required, list):
            for field in required:
                if field not in value:
                    issues.append(f"{path}.{field} is required.")
        properties = schema.get("properties") or {}
        if isinstance(properties, dict):
            if schema.get("additionalProperties") is False:
                extras = sorted(str(field) for field in value if field not in properties)
                if extras:
                    issues.append(f"{path} contains unsupported fields: {', '.join(extras)}.")
            for field, child_schema in properties.items():
                if field in value and isinstance(child_schema, dict):
                    issues.extend(
                        _validate_json_schema_fragment(
                            value[field],
                            child_schema,
                            path=f"{path}.{field}",
                        )
                    )

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                issues.extend(_validate_json_schema_fragment(item, item_schema, path=f"{path}[{index}]"))
    return issues


def validate_json_schema_fragment(value: Any, schema: Dict[str, Any], *, path: str = "$") -> list[str]:
    return _validate_json_schema_fragment(value, schema, path=path)


def _build_structured_output_prefinish_gate(
    output_schema: Dict[str, Any],
    *,
    expected_reviewer_id: str = "",
) -> Any:
    schema_text = json.dumps(output_schema, ensure_ascii=False, indent=2)
    template_text = json.dumps(
        _structured_output_repair_template(
            output_schema,
            expected_reviewer_id=expected_reviewer_id,
        ),
        ensure_ascii=False,
        indent=2,
    )

    async def _gate(stop_payload: Dict[str, Any]) -> HookResult:
        text = str(stop_payload.get("assistant_text") or "").strip()
        try:
            payload = _extract_json_object(text)
        except Exception as exc:
            return HookResult(
                hook_type="xcientist_structured_output_schema",
                success=False,
                blocked=True,
                reason=(
                    "Xcientist structured-output hook blocked completion because the final "
                    "response does not contain exactly one unambiguous JSON object.\n\n"
                    f"Error: {exc}\n\n"
                    "Return JSON shaped like this, replacing placeholder text with the actual review:\n"
                    "```json\n"
                    f"{template_text}\n"
                    "```\n\n"
                    "Expected structured output schema:\n"
                    "```json\n"
                    f"{schema_text}\n"
                    "```"
                ),
            )
        issues = _validate_json_schema_fragment(payload, output_schema)
        expected = str(expected_reviewer_id or "").strip()
        if expected and payload.get("reviewer_id") != expected:
            issues.append(
                f"$.reviewer_id must be exactly {expected!r} for this reviewer, got {payload.get('reviewer_id')!r}."
            )
        if issues:
            return HookResult(
                hook_type="xcientist_structured_output_schema",
                success=False,
                blocked=True,
                reason=(
                    "Xcientist structured-output hook blocked completion because the final "
                    "JSON does not satisfy the required schema.\n\n"
                    "Schema issues:\n"
                    + "\n".join(f"- {issue}" for issue in issues)
                    + "\n\nReturn JSON shaped like this, replacing placeholder text with the actual review:\n"
                    "```json\n"
                    f"{template_text}\n"
                    "```\n\nExpected structured output schema:\n"
                    "```json\n"
                    f"{schema_text}\n"
                    "```"
                ),
                metadata={"schema_issues": issues},
            )
        return HookResult(
            hook_type="xcientist_structured_output_schema",
            success=True,
            blocked=False,
        )

    return _gate


def _first_enum_value(schema: Dict[str, Any], preferred: str = "") -> Any:
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        if preferred and preferred in enum_values:
            return preferred
        return enum_values[0]
    return preferred


def _minimal_json_value_for_schema(
    schema: Dict[str, Any],
    *,
    field_name: str = "",
    expected_reviewer_id: str = "",
) -> Any:
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        expected_type = next((item for item in expected_type if item != "null"), expected_type[0] if expected_type else None)
    if "enum" in schema:
        preferred = ""
        if field_name == "status":
            preferred = "FAIL"
        elif field_name == "reviewer_kind":
            preferred = "agent"
        return _first_enum_value(schema, preferred=preferred)
    if expected_type == "object":
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        payload: Dict[str, Any] = {}
        if isinstance(properties, dict):
            fields = required if isinstance(required, list) and required else properties.keys()
            for child_name in fields:
                child_schema = properties.get(child_name)
                if isinstance(child_schema, dict):
                    payload[str(child_name)] = _minimal_json_value_for_schema(
                        child_schema,
                        field_name=str(child_name),
                        expected_reviewer_id=expected_reviewer_id,
                    )
        return payload
    if expected_type == "array":
        if field_name == "issues":
            return [
                {
                    "code": "review_issue_code",
                    "message": "Describe the issue, or use an empty issues array when status is PASS.",
                    "required_fix": "Describe the exact repair required.",
                    "evidence": [],
                }
            ]
        return []
    if expected_type == "string":
        if field_name == "reviewer_id":
            return expected_reviewer_id or "reviewer_id"
        if field_name == "summary":
            return "one concise sentence"
        return ""
    if expected_type == "boolean":
        return True if field_name == "blocking" else False
    if expected_type in {"integer", "number"}:
        return 0
    return None


def _structured_output_repair_template(
    output_schema: Dict[str, Any],
    *,
    expected_reviewer_id: str = "",
) -> Dict[str, Any]:
    payload = _minimal_json_value_for_schema(
        output_schema,
        expected_reviewer_id=expected_reviewer_id,
    )
    return payload if isinstance(payload, dict) else {}


def _structured_output_fallback_payload(
    output_schema: Dict[str, Any],
    *,
    expected_reviewer_id: str = "",
    reason: str = "",
) -> Dict[str, Any]:
    payload = _structured_output_repair_template(
        output_schema,
        expected_reviewer_id=expected_reviewer_id,
    )
    properties = output_schema.get("properties") or {}
    if "reviewer_id" in properties:
        payload["reviewer_id"] = expected_reviewer_id or str(payload.get("reviewer_id") or "reviewer")
        if "reviewer_kind" in properties:
            payload["reviewer_kind"] = "agent"
        if "status" in properties:
            payload["status"] = "FAIL"
        if "blocking" in properties:
            payload["blocking"] = True
        if "summary" in properties:
            payload["summary"] = (
                "Reviewer did not produce valid structured JSON after repeated hook repair attempts."
            )
        if "checked_artifacts" in properties:
            payload["checked_artifacts"] = []
        if "issues" in properties:
            payload["issues"] = [
                {
                    "code": "reviewer_output_schema_invalid",
                    "message": (
                        "The reviewer subagent repeatedly ended with a response that was not "
                        "exactly one JSON object satisfying the unified review schema."
                    ),
                    "required_fix": (
                        "Finish the worker step again without changing completed experiment "
                        "artifacts unless a real review issue is reported. This reruns the "
                        "prefinish reviewers; each reviewer final response must be exactly the "
                        "unified review JSON object, with no markdown fences or prose."
                    ),
                    "evidence": [reason] if reason else [],
                }
            ]
        if "structured_findings" in properties:
            payload["structured_findings"] = {}
    return payload


def _structured_output_fallback_text_from_metadata(
    metadata: Dict[str, Any],
    *,
    block_count: int,
) -> str | None:
    max_blocks = (
        _optional_positive_int(metadata.get("xcientist_structured_output_max_hook_blocks"))
        or _DEFAULT_STRUCTURED_OUTPUT_MAX_HOOK_BLOCKS
    )
    fallback = metadata.get("xcientist_structured_output_fallback_json")
    if block_count < max_blocks or not isinstance(fallback, dict):
        return None
    return json.dumps(fallback, ensure_ascii=False)


def _json_schema_instruction(output_schema: Dict[str, Any]) -> str:
    return (
        "\n\n# Structured Output Contract\n"
        "Return exactly one JSON object and no markdown fences or prose. "
        "The object must satisfy this JSON schema:\n"
        "```json\n"
        f"{json.dumps(output_schema, ensure_ascii=False, indent=2)}\n"
        "```"
    )


def _api_key_for_config(cfg: Dict[str, Any]) -> str:
    return str(
        cfg.get("api_key")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENHARNESS_OPENAI_API_KEY")
        or ""
    ).strip()


def _optional_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _optional_positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


_END_OF_STREAM = object()


def _redact_secrets(text: str) -> str:
    def _mask(match: re.Match[str]) -> str:
        value = match.group(0)
        return value[:3] + "**********..." + value[-4:]

    redacted = re.sub(r"\b(?:sk|tp)-[A-Za-z0-9_-]{12,}\b", _mask, str(text or ""))
    redacted = re.sub(
        r"(?i)(api[_-]?key|apikey|token|secret|password)=([^\s;&]+)",
        lambda m: f"{m.group(1)}=**********...{m.group(2)[-4:]}",
        redacted,
    )
    return redacted


def _shorten(value: Any, limit: int = 96) -> str:
    text = re.sub(r"\s+", " ", _redact_secrets(str(value or ""))).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _tool_input_summary(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict) or not payload:
        return ""
    for key in ("artifact_id", "path", "file_path", "command", "pattern", "query"):
        value = payload.get(key)
        if value:
            return f"{key}={_shorten(value)}"
    keys = ", ".join(str(key) for key in list(payload.keys())[:4])
    return f"keys={keys}" if keys else ""


def _usage_summary(usage: Any) -> str:
    if usage is None:
        return ""
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    parts = []
    if input_tokens is not None:
        parts.append(f"in={input_tokens}")
    if output_tokens is not None:
        parts.append(f"out={output_tokens}")
    if total_tokens is not None:
        parts.append(f"total={total_tokens}")
    return "tokens " + ", ".join(parts) if parts else ""


class OpenHarnessAgentRunner:
    """Thin adapter around vendored OpenHarness QueryEngine."""

    def __init__(
        self,
        *,
        model: str,
        workspace_root: str,
        verbose: bool = True,
        max_turns: Optional[int] = None,
        reviewer_mode: bool = False,
        artifact_context: Optional[Dict[str, Any]] = None,
        extra_tool_metadata: Optional[Dict[str, Any]] = None,
        enable_mcp: bool = False,
    ) -> None:
        from src.agents.experiment_agent.config import (
            get_openharness_config,
            get_worker_max_turns,
        )

        self.model = str(model or "").strip()
        self.workspace_root = os.path.realpath(workspace_root)
        self.verbose = bool(verbose)
        self.reviewer_mode = bool(reviewer_mode)
        self.enable_mcp = bool(enable_mcp)
        self.artifact_context = dict(artifact_context or {})
        self.extra_tool_metadata = dict(extra_tool_metadata or {})
        self._cfg = get_openharness_config()
        requested_max_turns = max_turns if max_turns is not None else self._cfg.get("max_turns")
        if requested_max_turns is None:
            requested_max_turns = get_worker_max_turns()
        self.max_turns = _optional_positive_int(requested_max_turns)
        ensure_openharness_runtime_env(self.workspace_root)

    def _build_tavily_mcp_servers(self) -> Dict[str, Any]:
        if self.reviewer_mode or not self.enable_mcp:
            return {}
        from src.agents.experiment_agent.config import get_workspace_config

        workspace_cfg = get_workspace_config()
        if not bool(workspace_cfg.get("tavily_enabled")):
            return {}
        api_key = str(workspace_cfg.get("tavily_api_key") or os.environ.get("TAVILY_API_KEY") or "").strip()
        if not api_key:
            return {}
        template = str(
            workspace_cfg.get("tavily_remote_url_template")
            or "https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
        )
        symbols = _load_openharness_symbols()
        return {
            "tavily": symbols["McpHttpServerConfig"](
                url=template.format(api_key=api_key),
                headers={},
            )
        }

    async def _build_mcp_manager(self):
        if not self.enable_mcp:
            return None
        if self.reviewer_mode:
            _write_mcp_status(
                self.workspace_root,
                {
                    "enabled": True,
                    "connected": False,
                    "status": "disabled_for_reviewer",
                    "reason": "Reviewer sessions are read-only and do not receive MCP tools.",
                    "servers": [],
                },
            )
            return None
        servers = self._build_tavily_mcp_servers()
        if not servers:
            _write_mcp_status(
                self.workspace_root,
                {
                    "enabled": True,
                    "connected": False,
                    "status": "unavailable",
                    "reason": "No MCP servers were configured or Tavily is disabled/missing an API key.",
                    "servers": [],
                },
            )
            return None
        from src.agents.experiment_agent.config import get_execution_config

        symbols = _load_openharness_symbols()
        manager = symbols["McpClientManager"](servers)
        timeout_seconds = _optional_positive_float(get_execution_config().get("mcp_timeout_seconds")) or 120.0
        try:
            await asyncio.wait_for(manager.connect_all(), timeout=timeout_seconds)
        except Exception as exc:
            await manager.close()
            _write_mcp_status(
                self.workspace_root,
                {
                    "enabled": True,
                    "connected": False,
                    "status": "connect_failed",
                    "reason": "MCP server connection failed.",
                    "error": _redact_secrets(str(exc)),
                    "error_type": type(exc).__name__,
                    "servers": sorted(servers.keys()),
                    "timeout_seconds": timeout_seconds,
                },
            )
            return None
        _write_mcp_status(
            self.workspace_root,
            {
                "enabled": True,
                "connected": True,
                "status": "connected",
                "reason": "",
                "servers": sorted(servers.keys()),
                "timeout_seconds": timeout_seconds,
            },
        )
        return manager

    def _build_registry(self, mcp_manager=None):
        symbols = _load_openharness_symbols()
        registry = symbols["ToolRegistry"]()
        registry.register(symbols["FileReadTool"]())
        registry.register(symbols["GlobTool"]())
        registry.register(symbols["GrepTool"]())
        if not self.reviewer_mode:
            from src.agents.experiment_agent.runtime.artifacts import artifact_tools

            for tool in artifact_tools():
                registry.register(tool)
            registry.register(symbols["BashTool"]())
            registry.register(symbols["FileWriteTool"]())
            registry.register(symbols["FileEditTool"]())
            if mcp_manager is not None:
                mcp_registry = symbols["create_default_tool_registry"](mcp_manager)
                for tool in mcp_registry.list_tools():
                    if tool.name.startswith("mcp__") or tool.name in {"list_mcp_resources", "read_mcp_resource"}:
                        registry.register(tool)
        return registry

    def _build_settings(self):
        symbols = _load_openharness_symbols()
        PermissionMode = symbols["PermissionMode"]
        mode = PermissionMode.PLAN if self.reviewer_mode else PermissionMode.FULL_AUTO
        denied_tools = ["bash", "write_file", "edit_file"] if self.reviewer_mode else []
        return symbols["Settings"](
            api_key=_api_key_for_config(self._cfg),
            model=self.model,
            max_tokens=int(self._cfg.get("max_tokens") or 16384),
            base_url=str(self._cfg.get("base_url") or "").strip() or None,
            timeout=_optional_positive_float(self._cfg.get("timeout_seconds")) or 0,
            api_format="openai",
            provider="openai",
            active_profile="openai-compatible",
            max_turns=self.max_turns if self.max_turns is not None else 0,
            permission=symbols["PermissionSettings"](
                mode=mode,
                denied_tools=denied_tools,
            ),
            memory=symbols["MemorySettings"](
                enabled=False,
                session_memory_enabled=False,
                auto_extract_enabled=False,
                auto_dream_enabled=False,
            ),
            allow_project_plugins=False,
            allow_project_skills=False,
            mcp_servers=self._build_tavily_mcp_servers(),
            verbose=self.verbose,
        )

    def _build_engine(self, *, system_prompt: str, cwd: Optional[str] = None, mcp_manager=None):
        symbols = _load_openharness_symbols()
        settings = self._build_settings()
        api_key = _api_key_for_config(self._cfg)
        if not api_key:
            raise RuntimeError(
                "OpenHarness runner requires api.openai.api_key or OPENAI_API_KEY."
            )
        api_client = symbols["OpenAICompatibleClient"](
            api_key,
            base_url=str(self._cfg.get("base_url") or "").strip() or None,
            timeout=_optional_positive_float(self._cfg.get("timeout_seconds")),
            max_retries=_optional_positive_int(self._cfg.get("request_max_retries")),
            retry_base_delay=_optional_positive_float(self._cfg.get("request_retry_base_delay")),
            retry_max_delay=_optional_positive_float(self._cfg.get("request_retry_max_delay")),
        )
        tool_metadata = {
            "session_id": f"xcientist-{os.path.basename(self.workspace_root)}",
            "openharness_runtime_env": ensure_openharness_runtime_env(self.workspace_root),
            "reviewer_mode": self.reviewer_mode,
            "xcientist_artifact_context": self.artifact_context,
            "mcp_manager": mcp_manager,
            "mcp_enabled": bool(self.enable_mcp),
            "mcp_connected": bool(mcp_manager is not None),
            **self.extra_tool_metadata,
        }
        from src.agents.experiment_agent.runtime.artifacts import build_xcientist_hook_executor

        return symbols["QueryEngine"](
            api_client=api_client,
            tool_registry=self._build_registry(mcp_manager),
            permission_checker=symbols["PermissionChecker"](settings.permission),
            cwd=os.path.realpath(cwd or self.workspace_root),
            model=self.model,
            system_prompt=system_prompt,
            max_tokens=int(self._cfg.get("max_tokens") or 16384),
            context_window_tokens=self._cfg.get("context_window_tokens"),
            auto_compact_threshold_tokens=self._cfg.get("auto_compact_threshold_tokens"),
            max_turns=self.max_turns,
            settings=settings,
            hook_executor=build_xcientist_hook_executor(tool_metadata),
            tool_metadata=tool_metadata,
        )

    async def run_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        agent_name: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> str:
        from src.agents.experiment_agent.telemetry import (
            Colors,
            format_seconds,
            print_activity,
        )

        role_prompt = f"You are `{agent_name}`.\n\n" if agent_name else ""
        mcp_manager = await self._build_mcp_manager()
        engine = self._build_engine(system_prompt=role_prompt + (system_prompt or ""), cwd=cwd, mcp_manager=mcp_manager)
        pending_final_text = ""
        symbols = _load_openharness_symbols()
        agent_label = str(agent_name or "openharness").replace("_", " ")
        mode = "review" if self.reviewer_mode else "worker"
        started_at = time.monotonic()
        last_event_at = started_at
        tool_count = 0
        heartbeat_count = 0
        structured_output_block_count = 0
        try:
            heartbeat_seconds = float(
                os.environ.get("XCIENTIST_OPENHARNESS_HEARTBEAT_SECONDS") or "30"
            )
        except ValueError:
            heartbeat_seconds = 30.0
        heartbeat_seconds = max(0.05, heartbeat_seconds)
        queue: asyncio.Queue[Any] = asyncio.Queue()

        async def _produce_events() -> None:
            try:
                async for item in engine.submit_message(user_prompt):
                    await queue.put(item)
            except BaseException as exc:
                await queue.put(exc)
            finally:
                await queue.put(_END_OF_STREAM)

        producer = asyncio.create_task(_produce_events())
        print_activity(
            "model",
            "start",
            f"{agent_label} | {mode} | {self.model}",
            color=Colors.OKBLUE,
        )

        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
                except asyncio.TimeoutError:
                    heartbeat_count += 1
                    elapsed = time.monotonic() - started_at
                    idle = time.monotonic() - last_event_at
                    print_activity(
                        "model",
                        "waiting",
                        (
                            f"{agent_label} | elapsed {format_seconds(elapsed)} | "
                            f"idle {format_seconds(idle)} | tools {tool_count}"
                        ),
                        color=Colors.WARNING,
                    )
                    continue

                if event is _END_OF_STREAM:
                    break
                if isinstance(event, BaseException):
                    raise event
                last_event_at = time.monotonic()

                if isinstance(event, symbols["AssistantTextDelta"]):
                    continue
                elif isinstance(event, symbols["AssistantTurnComplete"]):
                    text = event.message.text
                    pending_final_text = text or ""
                    detail = _usage_summary(getattr(event, "usage", None))
                    if detail:
                        print_activity("model", "turn", detail, color=Colors.OKGREEN)
                elif isinstance(event, symbols["ErrorEvent"]):
                    raise RuntimeError(event.message)
                elif isinstance(event, symbols["StatusEvent"]):
                    message = str(event.message or "")
                    if "Fix the reported issues with the available tools" in message:
                        pending_final_text = ""
                    if _STRUCTURED_OUTPUT_BLOCK_MESSAGE in message:
                        structured_output_block_count += 1
                        fallback_text = _structured_output_fallback_text_from_metadata(
                            self.extra_tool_metadata,
                            block_count=structured_output_block_count,
                        )
                        if fallback_text is not None:
                            pending_final_text = fallback_text
                            print_activity(
                                "openharness",
                                "status",
                                (
                                    "Structured-output hook did not converge after "
                                    f"{structured_output_block_count} blocked STOP attempts; "
                                    "returning a schema-valid FAIL reviewer report to the outer gate."
                                ),
                                color=Colors.WARNING,
                            )
                            break
                    if self.verbose:
                        print_activity("openharness", "status", message, color=Colors.OKBLUE)
                elif isinstance(event, symbols["ToolExecutionStarted"]):
                    pending_final_text = ""
                    tool_count += 1
                    print_activity(
                        "tool",
                        "start",
                        f"{event.tool_name} {_tool_input_summary(event.tool_input)}".strip(),
                        color=Colors.OKCYAN,
                    )
                elif isinstance(event, symbols["ToolExecutionCompleted"]):
                    status = "error" if event.is_error else "done"
                    color = Colors.FAIL if event.is_error else Colors.OKGREEN
                    output = _shorten(event.output, 140)
                    detail = f"{event.tool_name}"
                    if output:
                        detail = f"{detail} | {output}"
                    print_activity("tool", status, detail, color=color)
        finally:
            if not producer.done():
                producer.cancel()
                try:
                    await producer
                except asyncio.CancelledError:
                    pass
            if mcp_manager is not None:
                await mcp_manager.close()

        elapsed_total = time.monotonic() - started_at
        output_chars = len(pending_final_text)
        detail = (
            f"{agent_label} | {format_seconds(elapsed_total)} | "
            f"tools {tool_count} | chars {output_chars}"
        )
        if heartbeat_count:
            detail += f" | waits {heartbeat_count}"
        print_activity("model", "done", detail, color=Colors.OKGREEN)
        return pending_final_text.strip()

    async def run_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_schema: Dict[str, Any],
        agent_name: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        full_prompt = user_prompt + _json_schema_instruction(output_schema)
        last_error: Exception | None = None
        previous_metadata = dict(self.extra_tool_metadata)
        if "xcientist_prefinish_gate" not in self.extra_tool_metadata:
            expected_reviewer_id = str(
                self.extra_tool_metadata.get("xcientist_expected_reviewer_id") or ""
            ).strip()
            max_hook_blocks = (
                _optional_positive_int(self._cfg.get("structured_output_max_hook_blocks"))
                or _DEFAULT_STRUCTURED_OUTPUT_MAX_HOOK_BLOCKS
            )
            fallback_payload = (
                _structured_output_fallback_payload(
                    output_schema,
                    expected_reviewer_id=expected_reviewer_id,
                    reason=(
                        "Structured-output STOP hook exceeded "
                        f"{max_hook_blocks} repair attempts."
                    ),
                )
                if expected_reviewer_id
                else None
            )
            self.extra_tool_metadata = {
                **self.extra_tool_metadata,
                "xcientist_prefinish_gate": _build_structured_output_prefinish_gate(
                    output_schema,
                    expected_reviewer_id=expected_reviewer_id,
                ),
                "xcientist_structured_output_max_hook_blocks": max_hook_blocks,
            }
            if fallback_payload is not None:
                self.extra_tool_metadata["xcientist_structured_output_fallback_json"] = fallback_payload
        try:
            for attempt in range(1, 4):
                try:
                    text = await self.run_text(
                        system_prompt=system_prompt,
                        user_prompt=full_prompt,
                        agent_name=agent_name,
                        cwd=cwd,
                    )
                    if not text.strip():
                        raise ValueError("OpenHarness JSON response was empty.")
                    return _extract_json_object(text)
                except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
                    last_error = exc
                    message = str(exc)
                    retryable = (
                        "empty assistant message" in message.lower()
                        or "response was empty" in message.lower()
                    )
                    if not retryable or attempt >= 3:
                        raise
                    if self.verbose:
                        from src.agents.experiment_agent.telemetry import Colors, print_activity

                        print_activity(
                            "model",
                            "retry",
                            f"empty structured response {attempt}/3: {_shorten(message, 140)}",
                            color=Colors.WARNING,
                        )
                    await asyncio.sleep(min(2 * attempt, 5))
        finally:
            self.extra_tool_metadata = previous_metadata
        assert last_error is not None
        raise last_error


__all__ = [
    "OpenHarnessAgentRunner",
    "extract_json_object",
    "ensure_openharness_runtime_env",
    "validate_json_schema_fragment",
]
