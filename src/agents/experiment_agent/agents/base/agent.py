"""
Base Agent - OpenHands SDK Implementation

This module provides the base agent class using OpenHands SDK:
- Agent, Conversation, LLM from openhands.sdk
- Skills system for dynamic instruction loading
- Built-in memory and state persistence
- Tool management using openhands.tools
- Context condenser for managing long conversations

Key differences from openai-agents:
- Tools must be registered or created with executors
- Conversation manages the agent lifecycle
- Workspace abstraction for file operations
"""

import asyncio
import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Iterable
from urllib.parse import quote
from uuid import uuid4

from pydantic import SecretStr

from openhands.sdk import Agent, Conversation, LLM
from openhands.sdk.tool import Tool
from openhands.sdk import get_logger
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.llm import LLMStreamChunk

# Import built-in tools
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.delegate.visualizer import DelegationVisualizer
from openhands.tools.task import TaskToolSet

# Import skills loader
from src.agents.experiment_agent.skills import load_all_skills

from src.agents.experiment_agent.telemetry.hooks import OHColors
from src.agents.experiment_agent.tools import enable_experiment_tool_overrides
from src.agents.experiment_agent.tools.parsing import (
    extract_json_from_llm_output,
)
from src.agents.experiment_agent.config import (
    get_llm_config,
    is_minimax_model,
    setup_openai_api,
    get_api_config,
    get_workspace_config,
)
from src.agents.experiment_agent.runtime.checkpoint import (
    get_checkpoint_manager,
)
from src.agents.experiment_agent.agents.base.subagents import (
    default_builtin_tool_names,
)


logger = get_logger(__name__)
logging.getLogger("openhands.sdk.conversation.state").setLevel(logging.WARNING)
logging.getLogger("openhands.sdk.conversation.impl.local_conversation").setLevel(logging.WARNING)
logging.getLogger("openhands.sdk.conversation.local_conversation").setLevel(logging.WARNING)
NON_LOCAL_MCP_COMMANDS = {"npx", "uvx"}
GLOBAL_MCP_PREWARM_CACHE = set()
MCP_WRAPPER_FILENAMES = {
    "filesystem": "mcp-filesystem",
    "fetch": "mcp-fetch",
    "github": "mcp-github",
    "MiniMax": "mcp-minimax",
    "context7": "mcp-context7",
}
LOCAL_MCP_BINARIES = {
    "filesystem": "~/.cache/researchagent_mcp/npm/node_modules/.bin/mcp-server-filesystem",
    "fetch": "~/.cache/researchagent_mcp/npm/node_modules/.bin/mcp-fetch",
    "github": "~/.cache/researchagent_mcp/npm/node_modules/.bin/mcp-server-github",
    "context7": "~/.cache/researchagent_mcp/npm/node_modules/.bin/context7-mcp",
    "MiniMax": "~/.local/bin/minimax-coding-plan-mcp",
}
MCP_COMMAND_REWRITE_MAP = {
    "mcp-fetch": ("npx", ["-y", "@kazuph/mcp-fetch"]),
    "mcp-github": ("npx", ["-y", "@modelcontextprotocol/server-github"]),
    "context7-mcp": ("npx", ["-y", "@upstash/context7-mcp"]),
}
SECRET_ENV_KEYWORDS = (
    "TOKEN",
    "KEY",
    "SECRET",
    "PASSWORD",
    "AUTH",
    "SESSION",
    "COOKIE",
)
DEFAULT_SECRET_ENV_NAMES = (
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_PERSONAL_ACCESS_TOKEN",
    "GH_TOKEN",
    "GITHUB_AI_TOKEN",
    "MINIMAX_API_KEY",
    "MINIMAX_API_HOST",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
)
PROXY_ENV_NAMES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


def _get_mcp_wrapper_dir() -> str:
    return os.path.realpath(
        os.path.expanduser(
            os.environ.get(
                "EXPERIMENT_AGENT_MCP_WRAPPER_DIR",
                "~/.cache/researchagent_mcp/bin",
            )
        )
    )


def _resolve_mcp_wrapper_command(name: str) -> str:
    local_binary = LOCAL_MCP_BINARIES.get(name, "")
    if local_binary:
        local_path = os.path.realpath(os.path.expanduser(local_binary))
        if os.path.isfile(local_path) and os.access(local_path, os.X_OK):
            return local_path
    if os.environ.get("EXPERIMENT_AGENT_MCP_USE_WRAPPERS", "1") == "0":
        return ""
    filename = MCP_WRAPPER_FILENAMES.get(name, "")
    if not filename:
        return ""
    wrapper_path = os.path.realpath(os.path.join(_get_mcp_wrapper_dir(), filename))
    if os.path.isfile(wrapper_path) and os.access(wrapper_path, os.X_OK):
        return wrapper_path
    return ""


def _is_allowed_local_mcp_command(command: str) -> bool:
    if not command:
        return False
    command_path = os.path.realpath(os.path.expanduser(command))
    wrapper_dir = _get_mcp_wrapper_dir()
    local_binary_paths = {
        os.path.realpath(os.path.expanduser(path))
        for path in LOCAL_MCP_BINARIES.values()
        if path
    }
    if not os.path.isfile(command_path):
        return False
    if not os.access(command_path, os.X_OK):
        return False
    return command_path.startswith(f"{wrapper_dir}{os.sep}") or command_path in local_binary_paths


def _inherit_proxy_env(env: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(env or {})
    has_proxy = False
    for name in PROXY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            merged.setdefault(name, value)
            if "proxy" in name.lower() and "no_proxy" not in name.lower():
                has_proxy = True
    if has_proxy:
        merged.setdefault("NODE_USE_ENV_PROXY", "1")
    return merged


def _default_mcp_servers() -> Dict[str, Dict[str, Any]]:
    workspace_cfg = get_workspace_config()
    defaults = {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem"],
            "env": {},
        },
        "fetch": {
            "command": "npx",
            "args": ["-y", "@kazuph/mcp-fetch"],
            "env": {},
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": (
                    os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
                    or os.environ.get("GITHUB_TOKEN")
                    or os.environ.get("GH_TOKEN")
                    or os.environ.get("GITHUB_AI_TOKEN")
                    or ""
                ),
            },
        },
        "playwright": {
            "command": "npx",
            "args": ["-y", "@playwright/mcp"],
            "env": {},
        },
        "memory": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "env": {},
        },
        "MiniMax": {
            "command": "uvx",
            "args": ["minimax-coding-plan-mcp", "-y"],
            "env": {
                "MINIMAX_API_KEY": os.environ.get("MINIMAX_API_KEY", ""),
                "MINIMAX_API_HOST": os.environ.get(
                    "MINIMAX_API_HOST", "https://api.minimaxi.com"
                ),
            },
        },
        "context7": {
            "command": "npx",
            "args": ["-y", "@upstash/context7-mcp"],
            "env": {},
        },
    }
    tavily_enabled = bool(workspace_cfg.get("tavily_enabled"))
    tavily_api_key = str(workspace_cfg.get("tavily_api_key") or "").strip()
    tavily_url_template = str(
        workspace_cfg.get("tavily_remote_url_template")
        or "https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"
    )
    if tavily_enabled and tavily_api_key:
        tavily_url = tavily_url_template.format(api_key=quote(tavily_api_key, safe=""))
        defaults["tavily"] = {
            "command": "npx",
            "args": ["-y", "mcp-remote", tavily_url],
            "env": {},
        }
    for name, server in defaults.items():
        wrapper_command = _resolve_mcp_wrapper_command(name)
        if wrapper_command:
            server["command"] = wrapper_command
            server["args"] = []
    return defaults


def build_experiment_mcp_config(
    *,
    workspace_root: str,
    allowed_servers: Iterable[str] | None = None,
) -> Dict[str, Any] | None:
    source_servers = _default_mcp_servers()
    allowed = [name for name in (allowed_servers or []) if isinstance(name, str) and name]
    if not allowed:
        return None
    normalized_servers: Dict[str, Any] = {}
    for name in allowed:
        server = source_servers.get(name)
        if not isinstance(server, dict):
            continue
        try:
            normalized_servers[name] = _normalize_mcp_server(
                name=name,
                server=server,
                workspace_root=os.path.realpath(workspace_root),
            )
        except ValueError as exc:
            logger.warning("Skipping MCP server '%s': %s", name, exc)
    if not normalized_servers:
        return None
    return {"mcpServers": normalized_servers}


def _normalize_mcp_server(
    name: str, server: Dict[str, Any], workspace_root: str
) -> Dict[str, Any]:
    normalized = dict(server or {})
    command = str(normalized.get("command") or "").strip()
    args = normalized.get("args") or []
    if not isinstance(args, list):
        args = []
    env = normalized.get("env") or {}
    if not isinstance(env, dict):
        env = {}
    preferred_local_command = _resolve_mcp_wrapper_command(name)
    if preferred_local_command:
        command = preferred_local_command
        args = []
    elif (
        os.environ.get("EXPERIMENT_AGENT_MCP_USE_WRAPPERS", "1") != "0"
        and name in MCP_WRAPPER_FILENAMES
    ):
        logger.warning(
            "Local MCP wrapper requested for server '%s' but no local binary was found; falling back to configured command '%s'",
            name,
            command or "<empty>",
        )
    if command in MCP_COMMAND_REWRITE_MAP:
        command, args = MCP_COMMAND_REWRITE_MAP[command]
    if name == "fetch" and command in NON_LOCAL_MCP_COMMANDS:
        command, args = "npx", ["-y", "@kazuph/mcp-fetch"]
    if name == "github" and command in NON_LOCAL_MCP_COMMANDS:
        command, args = "npx", ["-y", "@modelcontextprotocol/server-github"]
    if name == "context7" and command in {"context7-mcp", "context7"}:
        command, args = "npx", ["-y", "@upstash/context7-mcp"]
    is_non_local = command in NON_LOCAL_MCP_COMMANDS
    if not is_non_local and not _is_allowed_local_mcp_command(command):
        raise ValueError(
            f"Unsupported local MCP command for server '{name}': {command}"
        )
    if name == "filesystem":
        if is_non_local:
            args = ["-y", "@modelcontextprotocol/server-filesystem", workspace_root]
        else:
            args = [workspace_root]
    if name in {"fetch", "context7"}:
        env = _inherit_proxy_env(env)
    if name == "github":
        merged_env = dict(env)
        token = (
            os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or os.environ.get("GITHUB_AI_TOKEN")
        )
        if token:
            merged_env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token
        env = merged_env
    if name == "MiniMax":
        merged_env = dict(env)
        if os.environ.get("MINIMAX_API_KEY"):
            merged_env["MINIMAX_API_KEY"] = os.environ["MINIMAX_API_KEY"]
        if os.environ.get("MINIMAX_API_HOST"):
            merged_env["MINIMAX_API_HOST"] = os.environ["MINIMAX_API_HOST"]
        env = merged_env
    normalized["command"] = command
    normalized["args"] = args
    normalized["env"] = env
    normalized["type"] = "stdio"
    return normalized


def create_oh_llm(model: str, usage_id: str = "agent", stream: bool = False) -> LLM:
    """
    Create an OpenHands LLM instance from config.

    Args:
        model: Model name (e.g., "MiniMax-M2.1", "gpt-4.1")
        usage_id: Usage identifier for metrics
        stream: Whether to enable streaming

    Returns:
        Configured LLM instance
    """
    config = get_llm_config(model)
    if is_minimax_model(model):
        # Keep MiniMax LLM traffic off proxy paths while leaving MCP networking unchanged.
        from src.agents.experiment_agent.config import ensure_minimax_no_proxy_env

        ensure_minimax_no_proxy_env(config.get("base_url"))

    # Add provider prefix for litellm compatibility.
    model_name = config["model_name"]
    provider_prefix = str(config.get("provider_prefix") or "").strip()
    if provider_prefix:
        model_name = f"{provider_prefix}/{model_name}"

    return LLM(
        model=model_name,
        api_key=SecretStr(config["api_key"]),
        base_url=config.get("base_url"),
        stream=stream,
        temperature=config.get("temperature", 0.7),
    )


def get_default_tools() -> List[Tool]:
    return [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]


def get_search_tools() -> List[Tool]:
    return []


def get_all_tools() -> List[Tool]:
    return get_default_tools() + get_search_tools()


class OpenHandsBaseAgent(ABC):
    """
    Base agent class using OpenHands SDK.

    Features:
    - Built-in conversation persistence
    - Tool management via OpenHands registry
    - Streaming support
    - Automatic retry on network errors
    """

    def __init__(
        self,
        agent_type: str,
        model: str,
        max_turns: int = 10000,
        verbose: bool = True,
        workspace_root: str = None,
        persistence_dir: str = None,
        enable_condenser: bool = True,
        condenser_max_size: int = 20,
        condenser_keep_first: int = 2,
        resume: bool = False,
    ):
        self.agent_type = agent_type
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose
        self.resume = resume

        self.workspace_root = os.path.realpath(workspace_root or os.getcwd())
        enable_experiment_tool_overrides()
        
        # Only create .conversations in workspace subdirectories, not in root
        if persistence_dir:
            self.persistence_dir = persistence_dir
        elif "/workspace/" in self.workspace_root or self.workspace_root.endswith("/workspace"):
            # Only create .conversations if workspace_root is a workspace subdirectory
            self.persistence_dir = os.path.join(self.workspace_root, ".conversations")
        else:
            # For root or non-workspace directories, don't create .conversations
            self.persistence_dir = None

        self.llm = create_oh_llm(model, usage_id=agent_type, stream=False)

        # Context condenser for long conversations
        self.enable_condenser = enable_condenser
        self.condenser_max_size = condenser_max_size
        self.condenser_keep_first = condenser_keep_first
        self._condenser = None

        # Try to load conversation_id from checkpoint for resume
        if resume:
            self.conversation_id = self._load_conversation_id()
        else:
            self.conversation_id = uuid4()

        # Create hooks for verbose output
        from src.agents.experiment_agent.telemetry.hooks import create_hooks

        self.hooks = create_hooks(
            agent_type=agent_type,
            show_llm_responses=verbose,
            show_tools=verbose,
            show_tool_args=verbose,
        )

        # Streaming state
        self._current_stream_state: Optional[
            Literal["thinking", "content", "tool_name", "tool_args"]
        ] = None
        self._mcp_prewarmed = False
        self._last_tool_usage_summary: Dict[str, Any] = {
            "tool_counts": {},
            "task_subagent_types": [],
            "delegate_agent_types": [],
        }

        logger.debug(f"{agent_type} initialized with model={model}")

    def on_token(self, chunk: LLMStreamChunk) -> None:
        """
        Handle all types of streaming tokens including content,
        tool calls, and thinking blocks with dynamic boundary detection.
        """
        if not self.verbose:
            return

        choices = chunk.choices
        for choice in choices:
            delta = choice.delta
            if delta is not None:
                # Handle thinking blocks (reasoning content)
                reasoning_content = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning_content, str) and reasoning_content:
                    if self._current_stream_state != "thinking":
                        if self._current_stream_state is not None:
                            sys.stdout.write("\n")
                        sys.stdout.write(f"{OHColors.HEADER}THINKING:{OHColors.ENDC} ")
                        self._current_stream_state = "thinking"
                    sys.stdout.write(
                        f"{OHColors.OKBLUE}{reasoning_content}{OHColors.ENDC}"
                    )
                    sys.stdout.flush()

                # Handle regular content
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    if self._current_stream_state != "content":
                        if self._current_stream_state is not None:
                            sys.stdout.write("\n")
                        sys.stdout.write(f"{OHColors.OKCYAN}CONTENT:{OHColors.ENDC} ")
                        self._current_stream_state = "content"
                    sys.stdout.write(content)
                    sys.stdout.flush()

                # Handle tool calls
                tool_calls = getattr(delta, "tool_calls", None)
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = (
                            tool_call.function.name if tool_call.function.name else ""
                        )
                        tool_args = (
                            tool_call.function.arguments
                            if tool_call.function.arguments
                            else ""
                        )
                        if tool_name:
                            if self._current_stream_state != "tool_name":
                                if self._current_stream_state is not None:
                                    sys.stdout.write("\n")
                                sys.stdout.write(
                                    f"{OHColors.WARNING}TOOL NAME:{OHColors.ENDC} "
                                )
                                self._current_stream_state = "tool_name"
                            sys.stdout.write(tool_name)
                            sys.stdout.flush()
                        if tool_args:
                            if self._current_stream_state != "tool_args":
                                if self._current_stream_state is not None:
                                    sys.stdout.write("\n")
                                sys.stdout.write(
                                    f"{OHColors.WARNING}TOOL ARGS:{OHColors.ENDC} "
                                )
                                self._current_stream_state = "tool_args"
                            sys.stdout.write(tool_args)
                            sys.stdout.flush()

    @abstractmethod
    def _build_system_prompt(self, **kwargs) -> str:
        """Build the system prompt for the agent."""
        pass

    @abstractmethod
    def _build_user_prompt(self, **kwargs) -> str:
        """Build the user prompt for the agent."""
        pass

    def _get_tools(self) -> List[Tool]:
        """Get tools available to this agent. Override in subclasses."""
        return get_default_tools()

    def _get_agent_context(self):
        """Get AgentContext with skills. Override in subclasses for custom skills."""
        return load_all_skills()

    def _get_filter_tools_regex(self, tools: List[Tool]) -> Optional[str]:
        """Optional OpenHands tool visibility filter."""
        env_key = f"EXPERIMENT_AGENT_{self.agent_type.upper()}_FILTER_TOOLS_REGEX"
        return os.environ.get(env_key) or os.environ.get("EXPERIMENT_AGENT_FILTER_TOOLS_REGEX")

    def _collect_conversation_secrets(self) -> Dict[str, str]:
        """Register env-backed secrets with the OpenHands secret registry."""
        names = set(DEFAULT_SECRET_ENV_NAMES)
        names.update(
            name
            for name in os.environ
            if any(keyword in name.upper() for keyword in SECRET_ENV_KEYWORDS)
        )
        secrets: Dict[str, str] = {}
        for name in sorted(names):
            value = os.environ.get(name)
            if value:
                secrets[name] = value
        return secrets

    def _get_runtime_artifact_dir(self) -> str:
        return os.path.join(self.workspace_root, ".openhands")

    def _safe_model_dump(self, payload: Any) -> Any:
        if payload is None:
            return None
        if hasattr(payload, "model_dump"):
            try:
                return payload.model_dump(mode="json")
            except TypeError:
                return payload.model_dump()
        return payload

    def _maybe_generate_conversation_title(self, conversation: Conversation) -> Optional[str]:
        if os.environ.get("EXPERIMENT_AGENT_GENERATE_TITLES", "0") != "1":
            return None
        title_llm = self.llm.model_copy(
            update={"usage_id": f"{self.agent_type}_title", "stream": False}
        )
        return conversation.generate_title(llm=title_llm, max_length=64)

    def _persist_conversation_metadata(self, conversation: Conversation) -> None:
        artifact_dir = self._get_runtime_artifact_dir()
        os.makedirs(artifact_dir, exist_ok=True)

        conversation_stats = conversation.conversation_stats
        observation_metrics = self._summarize_observation_metrics(conversation)
        mcp_config = self._build_mcp_config()
        mcp_servers = mcp_config.get("mcpServers") if isinstance(mcp_config, dict) else {}
        payload = {
            "agent_type": self.agent_type,
            "conversation_id": str(self.conversation_id),
            "workspace_root": self.workspace_root,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stats": self._safe_model_dump(conversation_stats),
            "combined_metrics": self._safe_model_dump(
                conversation_stats.get_combined_metrics()
            ),
            "secret_names": sorted(self._collect_conversation_secrets().keys()),
            "tool_observation_metrics": observation_metrics,
            "filesystem_mcp_enabled": isinstance(mcp_servers, dict)
            and "filesystem" in mcp_servers,
        }
        try:
            title = self._maybe_generate_conversation_title(conversation)
            if title:
                payload["title"] = title
        except Exception as exc:
            logger.warning(f"Failed to generate conversation title: {exc}")

        stats_path = os.path.join(
            artifact_dir, f"{self.agent_type.lower()}_conversation_stats.json"
        )
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _persist_runtime_config_snapshot(self) -> None:
        artifact_dir = self._get_runtime_artifact_dir()
        os.makedirs(artifact_dir, exist_ok=True)
        llm_cfg = get_llm_config(self.model)
        api_cfg = get_api_config()
        mcp_config = self._build_mcp_config()
        mcp_servers = mcp_config.get("mcpServers") if isinstance(mcp_config, dict) else {}
        payload = {
            "agent_type": self.agent_type,
            "workspace_root": self.workspace_root,
            "persistence_dir": self.persistence_dir,
            "model": self.model,
            "max_turns": self.max_turns,
            "resume": self.resume,
            "terminal_env_inheritance": "full_parent_env",
            "self_contained_policy_enabled": True,
            "llm": {
                "model_name": llm_cfg.get("model_name"),
                "base_url": llm_cfg.get("base_url"),
                "provider_prefix": llm_cfg.get("provider_prefix"),
                "is_minimax": bool(llm_cfg.get("is_minimax")),
            },
            "api_presence": {
                "openai_api_key": bool(api_cfg.get("openai_api_key")),
                "minimax_api_key": bool(api_cfg.get("minimax_api_key")),
                "serper_api_key": bool(api_cfg.get("serper_api_key")),
                "jina_api_key": bool(api_cfg.get("jina_api_key")),
                "github_ai_token": bool(api_cfg.get("github_ai_token")),
                "tavily_api_key": bool(get_workspace_config().get("tavily_api_key")),
                "hf_token": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
            },
            "mcp_servers": sorted(mcp_servers.keys()) if isinstance(mcp_servers, dict) else [],
        }
        output_path = os.path.join(
            artifact_dir, f"{self.agent_type.lower()}_runtime_config.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _serialize_event_payload(self, payload: Any) -> str:
        if payload is None:
            return ""
        try:
            if hasattr(payload, "model_dump_json"):
                return payload.model_dump_json()
            if hasattr(payload, "model_dump"):
                return json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, default=str)
            if isinstance(payload, (dict, list, tuple)):
                return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            pass
        return repr(payload)

    def _summarize_tool_usage(self, conversation: Conversation) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "tool_counts": {},
            "task_subagent_types": [],
            "delegate_agent_types": [],
            "delegate_commands": [],
            "delegate_ids": [],
        }
        events = getattr(getattr(conversation, "state", None), "events", []) or []
        seen_task_types: set[str] = set()
        seen_delegate_types: set[str] = set()
        seen_delegate_commands: set[str] = set()
        seen_delegate_ids: set[str] = set()

        def _walk_payload(payload: Any):
            if isinstance(payload, dict):
                yield payload
                for value in payload.values():
                    yield from _walk_payload(value)
            elif isinstance(payload, (list, tuple)):
                for item in payload:
                    yield from _walk_payload(item)

        for event in events:
            action_obj = getattr(event, "action", None)
            if action_obj is None:
                continue
            structured_payload = None
            if hasattr(action_obj, "model_dump"):
                try:
                    structured_payload = action_obj.model_dump(mode="json")
                except TypeError:
                    structured_payload = action_obj.model_dump()
                except Exception:
                    structured_payload = None

            found_tool_name = None
            if structured_payload is not None:
                for node in _walk_payload(structured_payload):
                    if not isinstance(node, dict):
                        continue
                    for key in ("tool_name", "name"):
                        value = node.get(key)
                        if isinstance(value, str) and value in {
                            "task",
                            "terminal",
                            "file_editor",
                            "task_tracker",
                        }:
                            found_tool_name = value
                            break
                    if found_tool_name:
                        break
                    command = node.get("command")
                    if isinstance(command, str) and command == "spawn":
                        if command not in seen_delegate_commands:
                            summary["delegate_commands"].append(command)
                            seen_delegate_commands.add(command)
                    ids = node.get("ids")
                    if isinstance(ids, list):
                        for item in ids:
                            if isinstance(item, str) and item and item not in seen_delegate_ids:
                                summary["delegate_ids"].append(item)
                                seen_delegate_ids.add(item)
                    agent_types = node.get("agent_types")
                    if isinstance(agent_types, list):
                        for item in agent_types:
                            if (
                                isinstance(item, str)
                                and item
                                and item not in seen_delegate_types
                            ):
                                summary["delegate_agent_types"].append(item)
                                seen_delegate_types.add(item)
                    tasks = node.get("tasks")
                    if isinstance(tasks, dict):
                        for item in tasks:
                            if isinstance(item, str) and item and item not in seen_delegate_ids:
                                summary["delegate_ids"].append(item)
                                seen_delegate_ids.add(item)
            if found_tool_name:
                summary["tool_counts"][found_tool_name] = (
                    summary["tool_counts"].get(found_tool_name, 0) + 1
                )

            action_text = self._serialize_event_payload(action_obj)
            lowered = action_text.lower()

            if found_tool_name is None:
                for tool_name in ("task", "terminal", "file_editor", "task_tracker"):
                    patterns = [
                        f'"tool_name":"{tool_name}"',
                        f'"tool_name": "{tool_name}"',
                        f'"name":"{tool_name}"',
                        f'"name": "{tool_name}"',
                        f"'{tool_name}'",
                    ]
                    if any(pattern in lowered for pattern in patterns):
                        summary["tool_counts"][tool_name] = summary["tool_counts"].get(tool_name, 0) + 1
                        break

            for command in ("spawn",):
                command_patterns = [
                    f'"command":"{command}"',
                    f'"command": "{command}"',
                    f"'command': '{command}'",
                ]
                if any(pattern in lowered for pattern in command_patterns):
                    if command not in seen_delegate_commands:
                        summary["delegate_commands"].append(command)
                        seen_delegate_commands.add(command)

            for field_name, bucket, seen in (
                ("subagent_type", "task_subagent_types", seen_task_types),
                ("agent_type", "delegate_agent_types", seen_delegate_types),
            ):
                matches = re.findall(
                    rf'"{field_name}"\s*:\s*"([^"]+)"',
                    action_text,
                )
                for match in matches:
                    if match and match not in seen:
                        summary[bucket].append(match)
                        seen.add(match)
        return summary

    def _summarize_observation_metrics(self, conversation: Conversation) -> Dict[str, Any]:
        largest_chars = 0
        largest_lines = 0
        truncated_count = 0
        raw_terminal_artifacts = 0
        events = getattr(getattr(conversation, "state", None), "events", []) or []
        for event in events:
            obs = getattr(event, "observation", None)
            if obs is None:
                continue
            payload = self._safe_model_dump(obs)
            if not isinstance(payload, dict):
                continue
            text_parts: List[str] = []
            content = payload.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
            if isinstance(payload.get("text"), str):
                text_parts.append(payload["text"])
            combined = "\n".join(part for part in text_parts if part)
            if combined:
                largest_chars = max(largest_chars, len(combined))
                largest_lines = max(largest_lines, combined.count("\n") + 1)
            if payload.get("truncated") or payload.get("output_truncated"):
                truncated_count += 1
            if payload.get("raw_output_path"):
                raw_terminal_artifacts += 1
        return {
            "largest_tool_observation_chars": largest_chars,
            "largest_file_window_lines": largest_lines,
            "truncated_tool_observation_count": truncated_count,
            "terminal_raw_log_artifacts": raw_terminal_artifacts,
        }

    def get_last_tool_usage_summary(self) -> Dict[str, Any]:
        return dict(self._last_tool_usage_summary)

    def _maybe_condense_conversation(self, conversation: Conversation) -> None:
        if os.environ.get("EXPERIMENT_AGENT_FORCE_CONDENSE_AT_END", "0") != "1":
            return
        try:
            conversation.condense()
        except Exception as exc:
            logger.warning(f"Failed to condense conversation: {exc}")

    def _get_condenser(self):
        """Get or create the context condenser."""
        if not self.enable_condenser:
            return None

        if self._condenser is None:
            # Create a non-streaming LLM for the condenser to avoid callback issues
            condenser_llm = self.llm.model_copy(
                update={"usage_id": f"{self.agent_type}_condenser", "stream": False}
            )
            self._condenser = LLMSummarizingCondenser(
                llm=condenser_llm,
                max_size=self.condenser_max_size,
                keep_first=self.condenser_keep_first,
            )
        return self._condenser

    def _load_conversation_id(self):
        """Load conversation_id from checkpoint for resume."""
        try:
            checkpoint_manager = get_checkpoint_manager(self.workspace_root)
            checkpoint = checkpoint_manager.load_checkpoint(self.agent_type)
            if checkpoint:
                conv_dir = checkpoint.get("conversation_persistence_dir")
                if conv_dir and os.path.exists(conv_dir):
                    # Extract conversation_id from directory name
                    # The persistence directory is typically named as {conversation_id}
                    conv_id = os.path.basename(conv_dir)
                    if conv_id:
                        logger.info(f"Resuming from conversation: {conv_id}")
                        # Return as string - Conversation accepts UUID or string
                        return conv_id
        except Exception as e:
            logger.warning(f"Failed to load conversation_id: {e}")
        return uuid4()

    def _save_checkpoint(self, iteration: int = 0, phase: str = "running"):
        """Save checkpoint for resume."""
        try:
            checkpoint_manager = get_checkpoint_manager(self.workspace_root)
            # Save the conversation persistence directory path
            conv_dir = self.persistence_dir + "/" + str(self.conversation_id) if self.persistence_dir else None
            checkpoint_manager.save_checkpoint(
                agent_type=self.agent_type,
                iteration=iteration,
                phase=phase,
                conversation_persistence_dir=conv_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _build_mcp_config(self) -> Dict[str, Any]:
        config = build_experiment_mcp_config(
            workspace_root=self.workspace_root,
            allowed_servers=[
                "filesystem",
                "fetch",
                "github",
                "playwright",
                "memory",
                "MiniMax",
                "context7",
                "tavily",
            ],
        )
        return config or {"mcpServers": {}}

    def _extract_package_name(self, command: str, args: List[Any]) -> str:
        string_args = [str(token) for token in args if isinstance(token, str)]
        if command == "npx":
            for idx, token in enumerate(string_args):
                if token in {"-p", "--package"} and idx + 1 < len(string_args):
                    next_token = string_args[idx + 1]
                    if next_token and not next_token.startswith("-"):
                        return next_token
            for token in string_args:
                if token and not token.startswith("-"):
                    return token
            return ""
        if command == "uvx":
            for idx, token in enumerate(string_args):
                if token == "--from" and idx + 1 < len(string_args):
                    next_token = string_args[idx + 1]
                    if next_token and not next_token.startswith("-"):
                        return next_token
            for token in string_args:
                if token and not token.startswith("-"):
                    return token
            return ""
        for token in string_args:
            if token and not token.startswith("-"):
                return token
        return ""

    def _prewarm_mcp_servers(self, mcp_config: Dict[str, Any]) -> None:
        if self._mcp_prewarmed:
            return
        if os.environ.get("EXPERIMENT_AGENT_PREWARM_MCP", "1") == "0":
            return
        servers = mcp_config.get("mcpServers") if isinstance(mcp_config, dict) else None
        if not isinstance(servers, dict):
            return
        for name, server in servers.items():
            command = str(server.get("command") or "").strip()
            args = server.get("args") or []
            if command not in NON_LOCAL_MCP_COMMANDS:
                if _is_allowed_local_mcp_command(command):
                    continue
                raise RuntimeError(f"MCP server '{name}' uses non-supported command")
            if not isinstance(args, list):
                args = []
            package_name = self._extract_package_name(command=command, args=args)
            warmup_signature = (
                f"{command}:{package_name}"
                if package_name
                else f"{command}:{' '.join(str(token) for token in args)}"
            )
            if warmup_signature in GLOBAL_MCP_PREWARM_CACHE:
                continue
            merged_env = os.environ.copy()
            server_env = server.get("env") or {}
            if isinstance(server_env, dict):
                merged_env.update({k: str(v) for k, v in server_env.items() if v})
            if command == "npx" and package_name:
                warmup_cmd = [
                    "npx",
                    "-y",
                    "-p",
                    package_name,
                    "node",
                    "-e",
                    "process.exit(0)",
                ]
            elif command == "uvx" and package_name:
                warmup_cmd = [
                    "uvx",
                    "--from",
                    package_name,
                    "python",
                    "-c",
                    "import sys; sys.exit(0)",
                ]
            elif command == "npx":
                warmup_cmd = [command, *args, "--version"]
            else:
                warmup_args = [package_name, "--version"] if package_name else [*args, "--version"]
                warmup_cmd = [command, *warmup_args]
            try:
                result = subprocess.run(
                    warmup_cmd,
                    cwd=self.workspace_root,
                    env=merged_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired as exc:
                logger.warning(
                    f"MCP warmup timed out for '{name}' with command: {' '.join(warmup_cmd)}"
                )
                continue
            if result.returncode != 0:
                logger.warning(
                    f"MCP warmup failed for '{name}' with command: {' '.join(warmup_cmd)}"
                )
                continue
            GLOBAL_MCP_PREWARM_CACHE.add(warmup_signature)
        self._mcp_prewarmed = True

    # Class attribute for template-based system prompt - subclasses should set this
    SYSTEM_PROMPT_TEMPLATE: str = None

    def _create_conversation(
        self, tools: List[Tool], system_prompt: str = None
    ) -> Conversation:
        """Create a conversation with the agent, tools, skills, and optional condenser."""
        instructions = system_prompt or self._build_system_prompt()
        agent_context = self._get_agent_context()
        filter_tools_regex = self._get_filter_tools_regex(tools)
        secrets = self._collect_conversation_secrets()

        mcp_config = self._build_mcp_config()
        self._prewarm_mcp_servers(mcp_config)

        # Build AgentContext with system_message_suffix properly set
        from openhands.sdk.context import AgentContext
        if agent_context is None:
            new_agent_context = AgentContext(
                skills=[],
                system_message_suffix=instructions,
                load_public_skills=False,
            )
        else:
            new_agent_context = AgentContext(
                skills=agent_context.skills,
                system_message_suffix=instructions,
                load_public_skills=False,
            )

        # Check if we should use a template file
        template_path = None
        if self.SYSTEM_PROMPT_TEMPLATE:
            # __file__ is .../src/agents/experiment_agent/agents/base/agent.py
            # 3 dirname calls gives .../src/agents/experiment_agent
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            template_path = os.path.join(base_dir, "prompts", self.SYSTEM_PROMPT_TEMPLATE)

        agent_kwargs = {
            "llm": self.llm,
            "tools": tools,
            "agent_context": new_agent_context,
            "condenser": self._get_condenser(),
            "filter_tools_regex": filter_tools_regex,
            "include_default_tools": default_builtin_tool_names(),
        }
        servers = mcp_config.get("mcpServers") if isinstance(mcp_config, dict) else None
        if isinstance(servers, dict) and servers:
            agent_kwargs["mcp_config"] = mcp_config
        if template_path:
            agent_kwargs["system_prompt_filename"] = template_path

        agent = Agent(**agent_kwargs)

        has_orchestration_tool = any(
            getattr(tool, "name", "") == TaskToolSet.name
            for tool in tools
        )
        conversation_kwargs = {
            "agent": agent,
            "workspace": self.workspace_root,
            "persistence_dir": self.persistence_dir,
            "conversation_id": self.conversation_id,
            "max_iteration_per_run": self.max_turns,
            "token_callbacks": [self.on_token],
        }
        if has_orchestration_tool:
            conversation_kwargs["visualizer"] = DelegationVisualizer(name=self.agent_type)

        conversation = Conversation(**conversation_kwargs)
        if secrets:
            conversation.update_secrets(secrets)

        return conversation

    def _is_retryable_network_error(self, error: Exception) -> bool:
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # Network/connection errors
        retryable_types = [
            "RemoteProtocolError",
            "ConnectionError",
            "TimeoutError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "ProtocolError",
            "IncompleteRead",
            "APIConnectionError",
            "LLMServiceUnavailableError",
        ]
        if any(t.lower() in error_type.lower() for t in retryable_types):
            return True

        retryable_keywords = [
            "peer closed connection",
            "incomplete chunked read",
            "connection reset",
            "broken pipe",
            "timeout",
            "connection refused",
            "network",
            "connection error",
            "service unavailable",
        ]
        if any(kw in error_msg for kw in retryable_keywords):
            return True

        # Check for rate limiting
        if "rate limit" in error_msg or "429" in error_msg:
            return True

        return False

    async def run(
        self,
        user_prompt: str,
        system_prompt: str = None,
        tools: List[Tool] = None,
        **kwargs,
    ) -> Any:
        """Run the agent with the given prompts and tools."""
        max_retries = 5  # Increased from 3
        retry_delays = [5, 15, 30, 60, 120]  # Longer delays

        for attempt in range(max_retries):
            try:
                return await self._run_impl(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    tools=tools,
                    **kwargs,
                )
            except Exception as e:
                is_retryable = self._is_retryable_network_error(e)
                is_last_attempt = attempt == max_retries - 1

                if is_retryable and not is_last_attempt:
                    delay = retry_delays[attempt]
                    self._log_warning(
                        f"Network error: {type(e).__name__}: {str(e)[:200]}"
                    )
                    self._log_info(
                        f"Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    if is_retryable and is_last_attempt:
                        self._log_error(
                            f"Max retries ({max_retries}) reached. Giving up."
                        )
                    raise

    async def _run_impl(
        self,
        user_prompt: str,
        system_prompt: str = None,
        tools: List[Tool] = None,
        **kwargs,
    ) -> Any:
        """Internal implementation of run."""
        if tools is None:
            tools = self._get_tools()

        if system_prompt is None:
            system_prompt = self._build_system_prompt(**kwargs)

        _ = setup_openai_api(model=self.model, verbose=False)

        if self.verbose:
            if self.resume:
                self._log_info(f"Resuming agent from checkpoint...")
            else:
                self._log_info(f"Running agent...")
                self._print_prompt(user_prompt)

        self._persist_runtime_config_snapshot()
        conversation = self._create_conversation(tools, system_prompt)

        # Save checkpoint after creating conversation
        self._save_checkpoint(iteration=kwargs.get("iteration", 0), phase="started")

        try:
            # If resuming, the conversation already has state loaded from persistence
            # Just call run() to continue. Otherwise, send message first.
            if self.resume:
                # Resume from checkpoint - conversation state should be loaded.
                if kwargs.get("append_user_message") and user_prompt:
                    self._log_info("Continuing saved conversation with follow-up feedback...")
                    conversation.send_message(user_prompt)
                else:
                    # OpenHands will continue from where it left off
                    self._log_info("Continuing from saved conversation state...")
            else:
                conversation.send_message(user_prompt)

            conversation.run()
            self._maybe_condense_conversation(conversation)

            # Clear checkpoint after successful completion
            if self.resume:
                checkpoint_manager = get_checkpoint_manager(self.workspace_root)
                checkpoint_manager.clear_checkpoint(self.agent_type)

            # Reset streaming state
            if self._current_stream_state is not None:
                print()  # Newline at end of stream
                self._current_stream_state = None

            # Extract the final assistant text from events.
            # Prefer the last message-like content over tool observations so
            # control JSON such as {"continue_iteration": true} is not
            # overwritten by later non-message events.
            final_output = ""
            for event in reversed(conversation.state.events):
                if hasattr(event, "content") and event.content:
                    if isinstance(event.content, str) and event.content.strip():
                        final_output = event.content
                        break
                    if isinstance(event.content, list):
                        text_parts = []
                        for item in event.content:
                            if hasattr(item, "text") and str(item.text).strip():
                                text_parts.append(str(item.text))
                            elif isinstance(item, dict) and str(item.get("text") or "").strip():
                                text_parts.append(str(item["text"]))
                        if text_parts:
                            final_output = "\n".join(text_parts).strip()
                            break
                if hasattr(event, "text") and isinstance(event.text, str) and event.text.strip():
                    final_output = event.text
                    break

            if not final_output:
                for event in reversed(conversation.state.events):
                    if hasattr(event, "observation") and event.observation:
                        obs = event.observation
                        if isinstance(obs, dict):
                            content = obs.get("content", [])
                            if isinstance(content, list):
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and str(item.get("text") or "").strip():
                                        text_parts.append(str(item["text"]))
                                    elif hasattr(item, "text") and str(item.text).strip():
                                        text_parts.append(str(item.text))
                                if text_parts:
                                    final_output = "\n".join(text_parts).strip()
                                    break
                            if isinstance(obs.get("text"), str) and obs["text"].strip():
                                final_output = obs["text"]
                                break
                    if hasattr(event, "action") and hasattr(event.action, "content"):
                        action_content = event.action.content
                        if isinstance(action_content, str) and action_content.strip():
                            final_output = action_content
                            break

            if self.verbose:
                self._print_output(final_output)

            # Return a result object for compatibility
            class AgentResult:
                def __init__(self, final_output):
                    self.final_output = final_output

            return AgentResult(final_output=final_output)
        finally:
            try:
                self._last_tool_usage_summary = self._summarize_tool_usage(conversation)
                self._persist_conversation_metadata(conversation)
            except Exception as exc:
                logger.warning(f"Failed to persist conversation metadata: {exc}")

    # run_streaming is removed or deprecated since run now supports streaming via callbacks
    # But if we want to keep it for backward compatibility:
    async def run_streaming(
        self,
        user_prompt: str,
        system_prompt: str = None,
        tools: List[Tool] = None,
        **kwargs,
    ) -> Any:
        """Run the agent with streaming output."""
        # Use the standard run method which now handles streaming
        return await self.run(user_prompt, system_prompt, tools, **kwargs)

    def _print_prompt(self, prompt: str) -> None:
        header = f"📤 LLM Request | {self.agent_type}"
        width = 78
        print(
            f"\n{OHColors.BOLD}{OHColors.MAGENTA}{'┌' + '─' * width + '┐'}{OHColors.ENDC}"
        )
        pad = width - len(header) - 2
        print(
            f"{OHColors.BOLD}{OHColors.MAGENTA}│ {header}{' ' * max(0, pad)}│{OHColors.ENDC}"
        )
        print(
            f"{OHColors.BOLD}{OHColors.MAGENTA}{'└' + '─' * width + '┘'}{OHColors.ENDC}"
        )
        print(f"{OHColors.OKCYAN}📥 Input:{OHColors.ENDC}\n")

        lines = prompt.splitlines()[:50]
        for line in lines:
            print(f"{OHColors.OKBLUE}{line}{OHColors.ENDC}")
        if len(prompt.splitlines()) > 50:
            print(f"{OHColors.WARNING}... (truncated){OHColors.ENDC}")
        print("")

    def _print_output(self, output: str) -> None:
        print(f"{OHColors.OKCYAN}📤 Output:{OHColors.ENDC}")

        output_lines = output.splitlines()[:100]
        output_text = "\n".join(output_lines)
        if len(output.splitlines()) > 100:
            output_text += f"\n{OHColors.WARNING}... (truncated){OHColors.ENDC}"

        print(f"{OHColors.OKGREEN}{output_text[:8000]}{OHColors.ENDC}\n")

    def _extract_output(self, result: Any) -> str:
        if isinstance(result, str):
            return result
        elif hasattr(result, "final_output"):
            final_output = getattr(result, "final_output")
            if isinstance(final_output, str) and final_output.strip():
                return final_output
        elif hasattr(result, "output"):
            output = getattr(result, "output")
            if isinstance(output, str) and output.strip():
                return output

        # Handle case where result is an object with events
        if hasattr(result, "events"):
            for event in reversed(result.events):
                # Check for MessageEvent with content
                if hasattr(event, "content") and event.content:
                    if isinstance(event.content, str):
                        return event.content
                    elif isinstance(event.content, list):
                        for item in event.content:
                            if isinstance(item, dict) and "text" in item:
                                return item["text"]
                            elif hasattr(item, "text"):
                                return item.text
                # Check for observation event (OpenHands SDK format)
                if hasattr(event, "observation") and event.observation:
                    obs = event.observation
                    if isinstance(obs, dict):
                        # Try to get content from observation
                        content = obs.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict):
                                    if "text" in item:
                                        return item["text"]
                                elif hasattr(item, "text"):
                                    return item.text
                        # Also check for 'text' field directly in observation
                        if "text" in obs:
                            return obs["text"]
                # Check for ActionEvent with content (alternative format)
                if hasattr(event, "action") and hasattr(event.action, "content"):
                    action_content = event.action.content
                    if isinstance(action_content, str):
                        return action_content

        return str(result) if result else ""

    def _extract_json(self, result: Any) -> Optional[Dict[str, Any]]:
        output = self._extract_output(result)
        return extract_json_from_llm_output(output)

    def _read_text_file(self, path: str) -> str:
        if not path or not os.path.exists(path):
            return ""
        try:
            max_chars = max(256, int(os.environ.get("EXPERIMENT_AGENT_TOOL_MAX_CHARS", "12000")))
            max_lines = max(16, int(os.environ.get("EXPERIMENT_AGENT_FILE_VIEW_MAX_LINES", "160")))
            lines: List[str] = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for _ in range(max_lines):
                    line = f.readline()
                    if line == "":
                        break
                    lines.append(line)
            text = "".join(lines)
            if len(text) > max_chars:
                return text[: max_chars - 16] + "... <truncated>"
            if len(lines) >= max_lines:
                return text.rstrip("\n") + "\n... <truncated>"
            return text
        except Exception:
            return ""

    def _log_info(self, message: str):
        if self.verbose:
            print(f"{self.agent_type}: {message}")
        logger.info(f"{self.agent_type}: {message}")

    def _log_error(self, message: str):
        print(f"{self.agent_type}: ❌ {message}")
        logger.error(f"{self.agent_type}: {message}")

    def _log_success(self, message: str):
        if self.verbose:
            print(f"{self.agent_type}: ✅ {message}")
        logger.info(f"{self.agent_type}: {message}")

    def _log_warning(self, message: str):
        if self.verbose:
            print(f"{self.agent_type}: ⚠️ {message}")
        logger.warning(f"{self.agent_type}: {message}")

    async def _run_agent(
        self,
        user_prompt: str,
        system_prompt: str = None,
        tools: List = None,
        **kwargs,
    ) -> Any:
        """
        Run agent with automatic retry on network errors.
        This is the main entry point used by layer agents.
        """
        return await self.run(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs,
        )

    async def _run_agent_impl(
        self,
        user_prompt: str,
        system_prompt: str = None,
        tools: List = None,
        **kwargs,
    ) -> Any:
        """
        Internal implementation of _run_agent using OpenHands SDK.
        """
        return await self._run_impl(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs,
        )


class PromptBuilder:
    """Helper class for building structured prompts."""

    def __init__(self):
        self.parts: List[str] = []

    def add_header(self, title: str, level: int = 1) -> "PromptBuilder":
        prefix = "#" * level
        self.parts.append(f"{prefix} {title}")
        self.parts.append("")
        return self

    def add_text(self, text: str) -> "PromptBuilder":
        self.parts.append(text)
        self.parts.append("")
        return self

    def add_list(self, items: List[str], ordered: bool = False) -> "PromptBuilder":
        for i, item in enumerate(items, 1):
            prefix = f"{i}." if ordered else "-"
            self.parts.append(f"{prefix} {item}")
        self.parts.append("")
        return self

    def add_code(self, code: str, language: str = "") -> "PromptBuilder":
        self.parts.append(f"```{language}")
        self.parts.append(code)
        self.parts.append("```")
        self.parts.append("")
        return self

    def add_section(self, title: str, content: str) -> "PromptBuilder":
        self.add_header(title, level=2)
        self.add_text(content)
        return self

    def add_key_value(self, key: str, value: str) -> "PromptBuilder":
        self.parts.append(f"**{key}:** {value}")
        return self

    def add_separator(self) -> "PromptBuilder":
        self.parts.append("---")
        self.parts.append("")
        return self

    def build(self) -> str:
        return "\n".join(self.parts)


# Backward compatibility alias
BaseAgent = OpenHandsBaseAgent
