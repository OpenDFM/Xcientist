"""
Logger hooks for Experiment Agent - OpenHands SDK Compatible

This module provides logging hooks compatible with OpenHands SDK.
"""

import json
import os
from typing import Any, Optional


class Colors:
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    MAGENTA = "\033[95m"


class OHColors(Colors):
    """Alias for Colors for OpenHands compatibility."""

    pass


def _truncate_to_tokens(text: str, max_tokens: int = 500) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + "..."


async def process_stream_events(stream, show_tool_args_delta: bool = False) -> dict:
    """Process streaming events from OpenHands conversation."""
    tool_count = 0
    text_chars = 0

    try:
        async for event in stream.stream_events():
            if not hasattr(event, "data"):
                continue

            event_type = type(event.data).__name__
            data = event.data

            if "TextDelta" in event_type:
                if hasattr(data, "delta") and data.delta:
                    text_chars += len(str(data.delta))

            if "ToolStart" in event_type:
                tool_count += 1

            if "ToolEnd" in event_type:
                if hasattr(data, "result"):
                    result = data.result
                    if isinstance(result, dict):
                        if result.get("success") is False:
                            error = result.get("error", "Unknown error")
                            print(f"   {Colors.FAIL}❌ Error: {error}{Colors.ENDC}")

        return {"tool_count": tool_count, "text_chars": text_chars}
    except Exception as e:
        return {"tool_count": tool_count, "text_chars": text_chars, "error": str(e)}


def create_hooks(
    agent_type: str = "Agent",
    show_llm_responses: bool = True,
    show_tools: bool = True,
    show_tool_args: bool = True,
) -> "VerboseRunHooks":
    return VerboseRunHooks(
        show_llm_responses=show_llm_responses,
        show_tools=show_tools,
        show_tool_args=show_tool_args,
        agent_type=agent_type,
    )


class VerboseRunHooks:
    """Verbose logging hooks for OpenHands agents."""

    def __init__(
        self,
        show_llm_responses: bool = True,
        show_tools: bool = True,
        show_tool_args: bool = True,
        agent_type: str = "Agent",
    ):
        self.show_llm_responses = show_llm_responses
        self.show_tools = show_tools
        self.show_tool_args = show_tool_args
        self.agent_type = agent_type
        self.current_agent_name = None
        self.turn_count = 0
        self.current_step = None

    async def on_agent_start(self, *args, **kwargs):
        self.turn_count = 0
        agent_name = kwargs.get("agent_name") or (args[0] if args else self.agent_type)
        self.current_agent_name = agent_name

        color = Colors.OKCYAN
        if "Architect" in self.agent_type:
            color = Colors.MAGENTA
        elif "Manager" in self.agent_type:
            color = Colors.OKBLUE
        elif "Worker" in self.agent_type:
            color = Colors.OKGREEN
        elif "Integrator" in self.agent_type:
            color = Colors.WARNING

        print(f"\n{Colors.BOLD}{color}{'┌' + '─' * 78 + '┐'}{Colors.ENDC}")
        header = f"🏃 {self.agent_type} START: {agent_name}"
        padding = 78 - len(header) - 4
        print(f"{Colors.BOLD}{color}│ {header}{' ' * max(0, padding)} │{Colors.ENDC}")
        print(f"{Colors.BOLD}{color}{'└' + '─' * 78 + '┘'}{Colors.ENDC}")

    async def on_agent_end(self, *args, **kwargs):
        agent_name = self.current_agent_name or self.agent_type

        print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'┌' + '─' * 78 + '┐'}{Colors.ENDC}")
        header = f"✅ {self.agent_type} END: {agent_name}"
        padding = 78 - len(header) - 4
        print(
            f"{Colors.BOLD}{Colors.OKGREEN}│ {header}{' ' * max(0, padding)} │{Colors.ENDC}"
        )
        print(f"{Colors.BOLD}{Colors.OKGREEN}{'└' + '─' * 78 + '┘'}{Colors.ENDC}\n")

    async def on_llm_start(self, *args, **kwargs):
        self.turn_count += 1
        if not self.show_llm_responses:
            return
        print(f"{Colors.WARNING}📤 LLM Request (Turn {self.turn_count}){Colors.ENDC}")

    async def on_llm_end(self, *args, **kwargs):
        if not self.show_llm_responses:
            return
        print(f"{Colors.OKGREEN}✓ LLM Response{Colors.ENDC}")

    async def on_tool_start(self, *args, **kwargs):
        tool_name = kwargs.get("tool_name", "Unknown")
        arguments = kwargs.get("arguments")

        print(f"\n{Colors.OKGREEN}🔧 TOOL: {tool_name}{Colors.ENDC}")

        if self.show_tool_args and arguments and isinstance(arguments, dict):
            for key, value in arguments.items():
                val_str = str(value)
                if len(val_str) > 300:
                    val_str = val_str[:297] + "..."
                print(f"   {Colors.OKCYAN}{key}:{Colors.ENDC} {val_str}")

        if self.show_tools:
            print(f"   {Colors.WARNING}Processing...{Colors.ENDC}", end=" ", flush=True)

    async def on_tool_end(self, *args, **kwargs):
        result = kwargs.get("result")

        if self.show_tools:
            print(f"{Colors.OKGREEN}✓{Colors.ENDC}")
        else:
            print(f"   {Colors.OKGREEN}✓ Done{Colors.ENDC}")

        if result is None:
            print(f"   {Colors.WARNING}📋 Result: (None){Colors.ENDC}")
            return

        try:
            if isinstance(result, dict):
                success = result.get("success")
                if success is False:
                    error_msg = (
                        result.get("error") or result.get("stderr") or "Unknown error"
                    )
                    print(f"   {Colors.FAIL}❌ Error: {error_msg}{Colors.ENDC}")
                elif success is True:
                    print(f"   {Colors.OKGREEN}✓ Success{Colors.ENDC}")
            elif isinstance(result, str):
                if len(result.strip()) > 0:
                    print(
                        f"   {Colors.OKCYAN}📋 Result: {result[:100]}...{Colors.ENDC}"
                    )
        except Exception:
            print(f"   {Colors.FAIL}📋 Result parse error{Colors.ENDC}")

    async def on_handoff(self, *args, **kwargs):
        from_agent = kwargs.get("from_agent", "Unknown")
        to_agent = kwargs.get("to_agent", "Unknown")
        print(f"\n{Colors.WARNING}🔄 Handoff: {from_agent} → {to_agent}{Colors.ENDC}")
