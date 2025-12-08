"""
Custom hooks for OpenAI agents library.

Provides hooks to intercept and display:
- Agent execution flow
- LLM requests and responses
- Tool calls and results
"""

import json
from typing import Any, Dict

from agents import RunHooks


class Colors:
    """ANSI color codes for terminal output."""

    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def _truncate_to_tokens(text: str, max_tokens: int = 500, model: str = "gpt-4") -> str:
    """
    Truncate text to approximately max_tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenization

    Returns:
        Truncated text with token count info
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, ~{len(text)} chars total)"


class VerboseRunHooks(RunHooks):
    """
    Custom hooks to print detailed agent execution information.

    Intercepts:
    - Agent start/end
    - LLM requests/responses
    - Tool calls/results
    """

    def __init__(
        self,
        show_llm_responses: bool = True,
        show_tools: bool = True,
        show_tool_args: bool = True,
    ):
        """
        Initialize verbose hooks.

        Args:
            show_llm_responses: Whether to show LLM response content
            show_tools: Whether to show tool results (detailed output)
            show_tool_args: Whether to show tool call arguments (always recommended)
        """
        super().__init__()
        self.show_llm_responses = show_llm_responses
        self.show_tools = show_tools
        self.show_tool_args = show_tool_args
        self.current_agent_name = None
        self.turn_count = 0
        self.current_step = None  # For displaying step number in agent header

    async def on_agent_start(self, *args, **kwargs):
        """Called when an agent starts running."""
        # Reset turn counter for new agent
        self.turn_count = 0

        # Extract agent name - try multiple approaches
        agent_name = None

        # Try kwargs first
        if "agent" in kwargs:
            agent = kwargs["agent"]
            if hasattr(agent, "name"):
                agent_name = agent.name

        # Try args
        if agent_name is None and args:
            for arg in args:
                if hasattr(arg, "name"):
                    agent_name = arg.name
                    break
                elif isinstance(arg, str) and len(arg) < 100:  # Avoid long strings
                    agent_name = arg
                    break

        # Fallback
        if agent_name is None:
            agent_name = "Agent"

        # Store for later use
        self.current_agent_name = agent_name

        # Print separator and start message with step number if available
        print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'┌' + '─' * 78 + '┐'}{Colors.ENDC}")
        if self.current_step is not None:
            # Include step number in header
            header = f"🏃 AGENT START (Step {self.current_step}): {agent_name}"
            padding = 78 - len(header) - 4  # 4 for "│ " and " │"
            print(
                f"{Colors.BOLD}{Colors.OKCYAN}│ {header}{' ' * padding} │{Colors.ENDC}"
            )
        else:
            print(
                f"{Colors.BOLD}{Colors.OKCYAN}│ 🏃 AGENT START: {agent_name:<62} │{Colors.ENDC}"
            )
        print(f"{Colors.BOLD}{Colors.OKCYAN}{'└' + '─' * 78 + '┘'}{Colors.ENDC}")

    async def on_agent_end(self, *args, **kwargs):
        """Called when an agent finishes running."""
        # Use stored agent name or try to extract
        agent_name = self.current_agent_name

        if agent_name is None:
            agent = kwargs.get("agent_name", args[0] if args else None)
            if hasattr(agent, "name"):
                agent_name = agent.name
            elif isinstance(agent, str):
                agent_name = agent
            else:
                agent_name = "Agent"

        # Print end separator
        print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'┌' + '─' * 78 + '┐'}{Colors.ENDC}")
        print(
            f"{Colors.BOLD}{Colors.OKGREEN}│ ✅ AGENT END: {agent_name:<64} │{Colors.ENDC}"
        )
        print(f"{Colors.BOLD}{Colors.OKGREEN}{'└' + '─' * 78 + '┘'}{Colors.ENDC}\n")

    async def on_llm_start(self, *args, **kwargs):
        """Called when LLM request starts."""
        if not self.show_llm_responses:
            return

        self.turn_count += 1

        print(f"{Colors.WARNING}📤 LLM Request{Colors.ENDC}")

        # Only show input on the first turn
        if self.turn_count > 1:
            return

        # Debug: print what we received (uncomment for debugging)
        # print(f"DEBUG: args count={len(args)}, kwargs keys={list(kwargs.keys())}")
        # for i, arg in enumerate(args):
        #     print(f"DEBUG: args[{i}] type={type(arg).__name__}")

        # Try to extract and display input messages (first 500 tokens)
        try:
            # Try to get messages from various sources
            messages = None

            # Check each arg for messages
            for i, arg in enumerate(args):
                if isinstance(arg, list):
                    messages = arg
                    # print(f"DEBUG: Found list at args[{i}]")
                    break
                elif hasattr(arg, "messages"):
                    messages = arg.messages
                    # print(f"DEBUG: Found messages attribute at args[{i}]")
                    break

            # Also check kwargs
            if messages is None and "messages" in kwargs:
                messages = kwargs["messages"]
                # print(f"DEBUG: Found messages in kwargs")

            # print(f"DEBUG: Final messages type={type(messages)}, is None={messages is None}")
            # if messages and hasattr(messages, "__len__"):
            #     print(f"DEBUG: messages length={len(messages)}")

            if messages:
                # Try to extract text from messages
                input_text = ""

                # Handle different message formats
                if isinstance(messages, list):
                    # Extract content from message list
                    for msg in messages:
                        if isinstance(msg, dict):
                            if "content" in msg:
                                content = msg["content"]
                                if isinstance(content, str):
                                    input_text += content + "\n"
                                elif isinstance(content, list):
                                    # Handle structured content
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            input_text += item["text"] + "\n"
                        elif hasattr(msg, "content"):
                            input_text += str(msg.content) + "\n"

                elif hasattr(messages, "__iter__") and not isinstance(messages, str):
                    # Try to iterate if it's some other iterable (but not string)
                    try:
                        for msg in messages:
                            if hasattr(msg, "content"):
                                content = getattr(msg, "content", None)
                                if content:
                                    input_text += str(content) + "\n"
                            elif isinstance(msg, dict) and "content" in msg:
                                input_text += str(msg["content"]) + "\n"
                    except:
                        pass

                # Display truncated input
                # print(f"DEBUG: input_text length={len(input_text.strip())}")
                if input_text.strip():
                    truncated_input = _truncate_to_tokens(
                        input_text.strip(), max_tokens=500
                    )
                    print(f"{Colors.OKCYAN}📥 Input:{Colors.ENDC}")
                    print(f"{Colors.OKBLUE}{truncated_input}{Colors.ENDC}\n")
                # else:
                #     print(f"  {Colors.WARNING}(Empty input){Colors.ENDC}\n")

        except Exception as e:
            # Show error for debugging
            print(
                f"  {Colors.WARNING}(Failed to extract input: {type(e).__name__}){Colors.ENDC}\n"
            )

    async def on_llm_end(self, *args, **kwargs):
        """Called when LLM response is received."""
        if not self.show_llm_responses:
            return

        response = kwargs.get("response", args[0] if args else None)
        print(f"{Colors.OKGREEN}✓ LLM Response{Colors.ENDC}")

        if response is None:
            return

        try:
            # Try to extract content from response
            content = None
            tool_calls = []

            # Check different response formats
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message"):
                    message = choice.message

                    # Check for text content
                    if hasattr(message, "content") and message.content:
                        content = message.content

                    # Check for tool calls
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = (
                                tool_call.function.name
                                if hasattr(tool_call, "function")
                                else "unknown"
                            )
                            tool_calls.append(tool_name)

            # Display content if available (first 500 tokens)
            if content:
                truncated_output = _truncate_to_tokens(content, max_tokens=500)
                print(f"{Colors.OKCYAN}📤 Output:{Colors.ENDC}")
                print(f"{Colors.OKGREEN}{truncated_output}{Colors.ENDC}\n")

            # Display tool calls if available
            if tool_calls:
                print(
                    f"{Colors.OKCYAN}🔧 Will call tools: {', '.join(tool_calls)}{Colors.ENDC}\n"
                )

        except Exception as e:
            print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}\n")

    async def on_tool_start(self, *args, **kwargs):
        """Called when a tool execution starts."""
        # Extract tool info from ToolContext
        tool_ctx = args[0] if args else None

        # Try to extract tool name
        if hasattr(tool_ctx, "tool_name"):
            tool_name = tool_ctx.tool_name
        else:
            tool_name = kwargs.get("tool_name", "Unknown")

        # Try to extract arguments
        arguments = None
        if hasattr(tool_ctx, "tool_arguments"):
            try:
                arguments = (
                    json.loads(tool_ctx.tool_arguments)
                    if isinstance(tool_ctx.tool_arguments, str)
                    else tool_ctx.tool_arguments
                )
            except:
                arguments = tool_ctx.tool_arguments

        # Always print tool name
        print(f"\n{Colors.OKGREEN}🔧 TOOL: {tool_name}{Colors.ENDC}")

        # Show arguments (controlled by show_tool_args, default True)
        if self.show_tool_args and arguments and isinstance(arguments, dict):
            for key, value in arguments.items():
                # Truncate long values
                val_str = str(value)
                if len(val_str) > 300:
                    val_str = val_str[:297] + "..."
                print(f"   {Colors.OKCYAN}{key}:{Colors.ENDC} {val_str}")

        if self.show_tools:
            print(f"   {Colors.WARNING}Processing...{Colors.ENDC}", end=" ", flush=True)

    async def on_tool_end(self, *args, **kwargs):
        """Called when a tool execution ends."""
        # Extract result - SDK signature: on_tool_end(context, agent, tool, result)
        # args[0]=context, args[1]=agent, args[2]=tool, args[3]=result
        result = args[3] if len(args) > 3 else kwargs.get("result", None)

        # Extract tool name for special handling
        tool_ctx = args[0] if args else None
        tool_name = "Unknown"
        if hasattr(tool_ctx, "tool_name"):
            tool_name = tool_ctx.tool_name
        elif len(args) > 2:
            tool_arg = args[2]
            if hasattr(tool_arg, "name"):
                tool_name = tool_arg.name
            elif isinstance(tool_arg, str):
                tool_name = tool_arg

        # Check if we should show full output (write_file error)
        show_full_output = False
        if tool_name == "write_file":
            if isinstance(result, dict) and result.get("success") is False:
                show_full_output = True
            elif isinstance(result, str) and (
                "An error occurred" in result
                or "Error:" in result
                or "Exception" in result
            ):
                show_full_output = True

        # Always print completion indicator if we printed "Processing..."
        if self.show_tools:
            print(f"{Colors.OKGREEN}✓{Colors.ENDC}")
        else:
            # Print a simple completion message
            print(f"   {Colors.OKGREEN}✓ Done{Colors.ENDC}")

        if result is None:
            print(f"   {Colors.WARNING}📋 Result: (None){Colors.ENDC}")
            return

        # Always display tool output (with truncation)
        try:
            # Convert result to string for display
            result_str = None

            if isinstance(result, dict):
                # Show success status
                success = result.get("success")
                if success is False:
                    # Show error info
                    error_msg = result.get("error", "Unknown error")
                    print(f"   {Colors.FAIL}❌ Error: {error_msg}{Colors.ENDC}")
                    # Still show result content for debugging
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)
                elif success is True:
                    print(f"   {Colors.OKGREEN}✓ Success{Colors.ENDC}", end="")

                    # Try to show key information
                    key_info = []

                    # Common fields to show (with truncation for long values)
                    if "message" in result:
                        msg = str(result["message"])
                        if len(msg) > 60:
                            msg = msg[:57] + "..."
                        key_info.append(f"Message: {msg}")
                    if "total_count" in result:
                        key_info.append(f"Count: {result['total_count']}")
                    if "file_path" in result:
                        path = str(result["file_path"])
                        if len(path) > 50:
                            path = "..." + path[-47:]
                        key_info.append(f"Path: {path}")
                    if "path" in result:
                        path = str(result["path"])
                        if len(path) > 50:
                            path = "..." + path[-47:]
                        key_info.append(f"Path: {path}")
                    if "files" in result and isinstance(result["files"], list):
                        key_info.append(f"Files: {len(result['files'])} items")
                    if "directories" in result and isinstance(
                        result["directories"], list
                    ):
                        key_info.append(f"Dirs: {len(result['directories'])} items")
                    if "imports" in result and isinstance(result["imports"], list):
                        key_info.append(f"Imports: {len(result['imports'])} items")
                    if "classes" in result and isinstance(result["classes"], list):
                        key_info.append(f"Classes: {len(result['classes'])} items")
                    if "functions" in result and isinstance(result["functions"], list):
                        key_info.append(f"Functions: {len(result['functions'])} items")

                    # If we have compact info, show it
                    if key_info:
                        print(f" | {Colors.OKCYAN}{' | '.join(key_info)}{Colors.ENDC}")
                    else:
                        print()  # newline after success

                # Try to extract main content for display
                for content_key in ["content", "output", "result", "data", "text"]:
                    if content_key in result:
                        result_str = str(result[content_key])
                        break

                # If no specific content key, show the whole dict (truncated)
                if result_str is None:
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)

            elif isinstance(result, str):
                result_str = result
            else:
                result_str = str(result)

            # Always display truncated output
            if result_str and len(result_str.strip()) > 0:
                # Avoid displaying Agent objects
                if "Agent(name=" in result_str or "FunctionTool(name=" in result_str:
                    print(
                        f"   {Colors.OKCYAN}📋 Result: (Agent/Tool object){Colors.ENDC}"
                    )
                elif show_full_output:
                    print(f"   {Colors.OKCYAN}📋 Result (Full Error):{Colors.ENDC}")
                    print(f"     {Colors.FAIL}{result_str}{Colors.ENDC}")
                else:
                    # Truncate to 500 tokens (about 2000 chars) for readability
                    truncated_result = _truncate_to_tokens(result_str, max_tokens=5000)
                    print(f"   {Colors.OKCYAN}📋 Result:{Colors.ENDC}")
                    # Show first 15 lines max
                    lines = truncated_result.split("\n")
                    max_display_lines = 15
                    for line in lines[:max_display_lines]:
                        # Truncate very long lines
                        if len(line) > 200:
                            line = line[:117] + "..."
                        if line.strip():
                            print(f"     {Colors.OKBLUE}{line}{Colors.ENDC}")
                    if len(lines) > max_display_lines:
                        print(
                            f"     {Colors.WARNING}... ({len(lines) - max_display_lines} more lines){Colors.ENDC}"
                        )

        except Exception as e:
            # Show error for debugging
            print(
                f"   {Colors.FAIL}📋 Result parse error: {type(e).__name__}: {e}{Colors.ENDC}"
            )

    async def on_handoff(self, *args, **kwargs):
        """Called when control is handed off between agents."""
        from_agent = kwargs.get("from_agent", args[0] if args else "Unknown")
        to_agent = kwargs.get("to_agent", args[1] if len(args) > 1 else "Unknown")

        # Extract agent names
        from_agent_name = (
            from_agent.name if hasattr(from_agent, "name") else str(from_agent)
        )
        to_agent_name = to_agent.name if hasattr(to_agent, "name") else str(to_agent)

        print(
            f"\n{Colors.WARNING}🔄 Handoff: {from_agent_name} → {to_agent_name}{Colors.ENDC}"
        )


def create_verbose_hooks(
    show_llm_responses: bool = True,
    show_tools: bool = True,
    show_tool_args: bool = True,
) -> VerboseRunHooks:
    """
    Create verbose run hooks for agent execution.

    Args:
        show_llm_responses: Whether to show full LLM responses
        show_tools: Whether to show detailed tool results
        show_tool_args: Whether to show tool call arguments (default True)

    Returns:
        VerboseRunHooks instance
    """
    return VerboseRunHooks(
        show_llm_responses=show_llm_responses,
        show_tools=show_tools,
        show_tool_args=show_tool_args,
    )
