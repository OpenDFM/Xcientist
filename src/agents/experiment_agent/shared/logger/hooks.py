import json
import os
from typing import Any, Dict, List, Optional

from agents import RunHooks

from src.agents.experiment_agent.shared.utils.memory_middleware import (
    record_llm_end,
    record_llm_start,
    record_tool_end,
    record_tool_start,
)


class Colors:
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    MAGENTA = "\033[95m"


async def process_stream_events(stream, show_tool_args_delta: bool = False) -> dict:
    tool_count = 0
    text_chars = 0
    announced_tools = set()
    current_tool = None
    
    # Unified Thinking State
    is_printing_block = False

    async for event in stream.stream_events():
        if not hasattr(event, "data"):
            continue

        event_type = type(event.data).__name__
        data = event.data
        
        # Unified Text/Reasoning Handling
        is_text_event = False
        delta_text = ""
        
        # Check for ResponseTextDelta (Legacy/Standard)
        if "ResponseTextDelta" in event_type:
            if hasattr(data, "delta") and data.delta:
                delta_text = str(data.delta)
                is_text_event = True
        
        # Check for Raw Events (Reasoning/Content)
        if hasattr(data, "type"):
            dtype = getattr(data, "type", "")
            if dtype in ["response.reasoning_text.delta", "response.reasoning_summary_text.delta", "response.output_text.delta"]:
                delta = getattr(data, "delta", "")
                if delta:
                    delta_text = str(delta)
                    is_text_event = True
                    
        if is_text_event:
            text_chars += len(delta_text)
            if not is_printing_block:
                print(f"\n{Colors.OKCYAN}🧠 Thinking:{Colors.ENDC}\n{Colors.WARNING}", end="", flush=True)
                is_printing_block = True
            print(delta_text, end="", flush=True)
            continue
            
        # Non-text event: Close block
        if is_printing_block:
            print(f"{Colors.ENDC}\n", flush=True)
            is_printing_block = False

        # Stream function/tool call arguments (often noisy; usually prefer hook-based tool printing)
        if "FunctionCallArgumentsDelta" in event_type:
            if show_tool_args_delta and hasattr(data, "delta") and data.delta:
                print(f"{Colors.OKCYAN}{data.delta}{Colors.ENDC}", end="", flush=True)
            continue

        # Tool call boundaries (optional light tracking)
        if "ResponseOutputItemAdded" in event_type:
            if hasattr(data, "item"):
                item = data.item
                if hasattr(item, "type") and item.type == "function_call":
                    tool_name = getattr(item, "name", None)
                    if tool_name and tool_name != current_tool:
                        current_tool = tool_name
                        tool_count += 1
                        # Announce tool call intent (clean, avoids raw JSON blobs)
                        if tool_name not in announced_tools:
                            announced_tools.add(tool_name)
                        print(
                            f"\n{Colors.BOLD}{Colors.OKCYAN}→ LLM requested TOOL: {tool_name}{Colors.ENDC}"
                        )
                    # Try to show tool arguments snapshot (once), without streaming raw JSON deltas.
                    try:
                        args_str = None
                        if hasattr(item, "arguments") and item.arguments:
                            args_str = str(item.arguments)
                        elif (
                            hasattr(item, "function")
                            and hasattr(item.function, "arguments")
                            and item.function.arguments
                        ):
                            args_str = str(item.function.arguments)
                        if args_str:
                            # Keep it compact; hooks.on_tool_start will print full parsed args later.
                            preview = args_str.strip().replace("\n", " ")
                            if len(preview) > 300:
                                preview = preview[:297] + "..."
                            print(f"{Colors.OKBLUE}  args: {preview}{Colors.ENDC}")
                    except Exception:
                        pass
            continue

        if "ResponseOutputItemDone" in event_type:
            current_tool = None
            continue

    if is_printing_block:
        print(f"{Colors.ENDC}\n", flush=True)

    return {"tool_count": tool_count, "text_chars": text_chars}


def _truncate_to_tokens(text: str, max_tokens: int = 500, model: str = "gpt-4") -> str:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, ~{len(text)} chars total)"


def _as_plain_obj(obj: Any) -> Optional[Any]:
    if obj is None:
        return None
    try:
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            return obj.model_dump()
    except Exception:
        pass
    try:
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            return obj.dict()
    except Exception:
        pass
    try:
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass
    try:
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict) and d:
            return d
    except Exception:
        pass
    return None


def _extract_think_tag(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    start = text.find("<think>")
    end = text.find("</think>")
    if start == -1 or end == -1 or end <= start:
        return None
    inner = text[start + len("<think>") : end].strip()
    return inner if inner else None


def _extract_reasoning_from_modelresponse_output(output_value: Any) -> Optional[str]:
    expanded = _as_plain_obj(output_value)
    if expanded is not None and expanded is not output_value:
        output_value = expanded

    if not isinstance(output_value, list):
        return None

    # Pass 1: explicit reasoning_details / reasoning-like content
    for item in output_value:
        item_dump = _as_plain_obj(item) or item
        if not isinstance(item_dump, dict):
            continue

        # Sometimes reasoning_details is attached at message level
        rd = item_dump.get("reasoning_details", None)
        if isinstance(rd, list) and rd:
            texts = []
            for d in rd:
                if isinstance(d, dict) and isinstance(d.get("text", None), str):
                    texts.append(d["text"])
            joined = "\n".join([t for t in texts if t.strip()])
            if joined.strip():
                return joined

        content = item_dump.get("content", None)
        content_dump = _as_plain_obj(content) or content
        if isinstance(content_dump, list):
            for c in content_dump:
                cd = _as_plain_obj(c) or c
                if not isinstance(cd, dict):
                    continue
                ctype = cd.get("type", None)
                if isinstance(ctype, str) and ctype.lower() in (
                    "reasoning",
                    "thinking",
                ):
                    # try common text fields
                    for k in ("text", "content", "value"):
                        v = cd.get(k, None)
                        if isinstance(v, str) and v.strip():
                            return v
                rd2 = cd.get("reasoning_details", None)
                if isinstance(rd2, list) and rd2:
                    texts = []
                    for d in rd2:
                        if isinstance(d, dict) and isinstance(d.get("text", None), str):
                            texts.append(d["text"])
                    joined = "\n".join([t for t in texts if t.strip()])
                    if joined.strip():
                        return joined

    # Pass 2: <think> tags inside output_text
    for item in output_value:
        item_dump = _as_plain_obj(item) or item
        if not isinstance(item_dump, dict):
            continue
        content = item_dump.get("content", None)
        content_dump = _as_plain_obj(content) or content
        if not isinstance(content_dump, list):
            continue
        for c in content_dump:
            cd = _as_plain_obj(c) or c
            if not isinstance(cd, dict):
                continue
            ctype = cd.get("type", None)
            if isinstance(ctype, str) and ctype == "output_text":
                txt = cd.get("text", None)
                if isinstance(txt, str) and txt:
                    think = _extract_think_tag(txt)
                    if think:
                        return think

    return None


class VerboseRunHooks(RunHooks):
    def __init__(
        self,
        show_llm_responses: bool = True,
        show_tools: bool = True,
        show_tool_args: bool = True,
        agent_type: str = "Agent",
    ):
        super().__init__()
        self.show_llm_responses = show_llm_responses
        self.show_tools = show_tools
        self.show_tool_args = show_tool_args
        self.agent_type = agent_type
        self.current_agent_name = None
        self.turn_count = 0
        self.current_step = None

    async def on_agent_start(self, *args, **kwargs):
        """Called when an agent starts running."""
        self.turn_count = 0

        # Extract agent name
        agent_name = None

        if "agent" in kwargs:
            agent = kwargs["agent"]
            if hasattr(agent, "name"):
                agent_name = agent.name

        if agent_name is None and args:
            for arg in args:
                if hasattr(arg, "name"):
                    agent_name = arg.name
                    break
                elif isinstance(arg, str) and len(arg) < 100:
                    agent_name = arg
                    break

        if agent_name is None:
            agent_name = self.agent_type

        self.current_agent_name = agent_name

        # Color based on agent type
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
        if self.current_step is not None:
            header = (
                f"🏃 {self.agent_type} START (Step {self.current_step}): {agent_name}"
            )
            padding = 78 - len(header) - 4
            print(
                f"{Colors.BOLD}{color}│ {header}{' ' * max(0, padding)} │{Colors.ENDC}"
            )
        else:
            header = f"🏃 {self.agent_type} START: {agent_name}"
            padding = 78 - len(header) - 4
            print(
                f"{Colors.BOLD}{color}│ {header}{' ' * max(0, padding)} │{Colors.ENDC}"
            )
        print(f"{Colors.BOLD}{color}{'└' + '─' * 78 + '┘'}{Colors.ENDC}")

    async def on_agent_end(self, *args, **kwargs):
        """Called when an agent finishes running."""
        agent_name = self.current_agent_name

        if agent_name is None:
            agent = kwargs.get("agent_name", args[0] if args else None)
            if hasattr(agent, "name"):
                agent_name = agent.name
            elif isinstance(agent, str):
                agent_name = agent
            else:
                agent_name = self.agent_type

        print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'┌' + '─' * 78 + '┐'}{Colors.ENDC}")
        header = f"✅ {self.agent_type} END: {agent_name}"
        padding = 78 - len(header) - 4
        print(
            f"{Colors.BOLD}{Colors.OKGREEN}│ {header}{' ' * max(0, padding)} │{Colors.ENDC}"
        )
        print(f"{Colors.BOLD}{Colors.OKGREEN}{'└' + '─' * 78 + '┘'}{Colors.ENDC}\n")

    async def on_llm_start(self, *args, **kwargs):
        """Called when LLM request starts."""
        self.turn_count += 1

        # Record LLM request for memory trace (independent of printing).
        try:
            messages = None
            for arg in args:
                if isinstance(arg, list):
                    messages = arg
                    break
                if hasattr(arg, "messages"):
                    messages = getattr(arg, "messages", None)
                    break
            if messages is None and "messages" in kwargs:
                messages = kwargs["messages"]
            record_llm_start(
                messages=messages, agent_type=self.agent_type, turn=self.turn_count
            )
        except Exception:
            pass

        if not self.show_llm_responses:
            return
        print(f"{Colors.WARNING}📤 LLM Request (Turn {self.turn_count}){Colors.ENDC}")

        # Only show input on the first turn
        if self.turn_count > 1:
            return

        try:
            messages = None
            for i, arg in enumerate(args):
                if isinstance(arg, list):
                    messages = arg
                    break
                elif hasattr(arg, "messages"):
                    messages = arg.messages
                    break

            if messages is None and "messages" in kwargs:
                messages = kwargs["messages"]

            if messages:
                input_text = ""

                if isinstance(messages, list):
                    for msg in messages:
                        if isinstance(msg, dict):
                            if "content" in msg:
                                content = msg["content"]
                                if isinstance(content, str):
                                    input_text += content + "\n"
                                elif isinstance(content, list):
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            input_text += item["text"] + "\n"
                        elif hasattr(msg, "content"):
                            input_text += str(msg.content) + "\n"

                elif hasattr(messages, "__iter__") and not isinstance(messages, str):
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

                if input_text.strip():
                    truncated_input = _truncate_to_tokens(
                        input_text.strip(), max_tokens=500
                    )
                    print(f"{Colors.OKCYAN}📥 Input:{Colors.ENDC}")
                    print(f"{Colors.OKBLUE}{truncated_input}{Colors.ENDC}\n")

        except Exception as e:
            print(
                f"  {Colors.WARNING}(Failed to extract input: {type(e).__name__}){Colors.ENDC}\n"
            )

    async def on_llm_end(self, *args, **kwargs):
        response = kwargs.get("response", args[0] if args else None)
        llm_response = args[-1] if args else response

        try:
            content = None
            reasoning = None
            tool_calls = []

            if hasattr(llm_response, "choices") and llm_response.choices:
                choice = llm_response.choices[0]
                if hasattr(choice, "message"):
                    message = choice.message

                    if hasattr(message, "content") and message.content:
                        content = message.content

                    for attr in (
                        "reasoning",
                        "reasoning_content",
                        "thinking",
                        "thought",
                        "thoughts",
                        "reasoning_details",
                    ):
                        if hasattr(message, attr):
                            val = getattr(message, attr, None)
                            if isinstance(val, str) and val.strip():
                                reasoning = val
                                break
                            if isinstance(val, list) and val:
                                texts = []
                                for item in val:
                                    if isinstance(item, dict) and isinstance(
                                        item.get("text", None), str
                                    ):
                                        texts.append(item["text"])
                                joined = "\n".join([t for t in texts if t.strip()])
                                if joined.strip():
                                    reasoning = joined
                                    break

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = (
                                tool_call.function.name
                                if hasattr(tool_call, "function")
                                else "unknown"
                            )
                            tool_calls.append(tool_name)

            if reasoning is None and hasattr(llm_response, "output"):
                try:
                    reasoning = _extract_reasoning_from_modelresponse_output(
                        getattr(llm_response, "output", None)
                    )
                except Exception:
                    pass

            record_llm_end(
                content=content,
                reasoning=reasoning,
                tool_calls=tool_calls,
                agent_type=self.agent_type,
                turn=self.turn_count,
            )
        except Exception:
            pass

        if not self.show_llm_responses:
            return

        print(f"{Colors.OKGREEN}✓ LLM Response{Colors.ENDC}")

        if llm_response is None:
            return

        try:
            content = None
            reasoning = None
            tool_calls = []

            if hasattr(llm_response, "choices") and llm_response.choices:
                choice = llm_response.choices[0]
                if hasattr(choice, "message"):
                    message = choice.message

                    if hasattr(message, "content") and message.content:
                        content = message.content

                    for attr in (
                        "reasoning",
                        "reasoning_content",
                        "thinking",
                        "thought",
                        "thoughts",
                        "reasoning_details",
                    ):
                        if hasattr(message, attr):
                            val = getattr(message, attr, None)
                            if isinstance(val, str) and val.strip():
                                reasoning = val
                                break
                            if isinstance(val, list) and val:
                                texts = []
                                for item in val:
                                    if isinstance(item, dict) and isinstance(
                                        item.get("text", None), str
                                    ):
                                        texts.append(item["text"])
                                joined = "\n".join([t for t in texts if t.strip()])
                                if joined.strip():
                                    reasoning = joined
                                    break

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_name = (
                                tool_call.function.name
                                if hasattr(tool_call, "function")
                                else "unknown"
                            )
                            tool_calls.append(tool_name)

            # Responses-style fallback: ModelResponse.output
            if reasoning is None and hasattr(llm_response, "output"):
                try:
                    reasoning = _extract_reasoning_from_modelresponse_output(
                        getattr(llm_response, "output", None)
                    )
                except Exception:
                    pass

            show_reasoning_env = (
                os.environ.get("SHOW_LLM_REASONING", "").strip().lower()
            )
            show_reasoning = show_reasoning_env in ("1", "true", "yes", "y", "on")
            if show_reasoning and reasoning:
                truncated_reasoning = _truncate_to_tokens(
                    str(reasoning), max_tokens=300
                )
                print(f"{Colors.OKCYAN}🧠 Thinking:{Colors.ENDC}")
                print(f"{Colors.WARNING}{truncated_reasoning}{Colors.ENDC}\n")

            if content:
                truncated_output = _truncate_to_tokens(content, max_tokens=500)
                print(f"{Colors.OKCYAN}📤 Output:{Colors.ENDC}")
                print(f"{Colors.OKGREEN}{truncated_output}{Colors.ENDC}\n")

            if tool_calls:
                print(
                    f"{Colors.OKCYAN}🔧 Will call tools: {', '.join(tool_calls)}{Colors.ENDC}\n"
                )

        except Exception as e:
            print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}\n")

    async def on_tool_start(self, *args, **kwargs):
        """Called when a tool execution starts."""
        tool_ctx = args[0] if args else None

        if hasattr(tool_ctx, "tool_name"):
            tool_name = tool_ctx.tool_name
        else:
            tool_name = kwargs.get("tool_name", "Unknown")

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

        # Record tool start for memory trace.
        try:
            record_tool_start(
                tool_name=str(tool_name or ""),
                arguments=arguments,
                agent_type=self.agent_type,
            )
        except Exception:
            pass

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
        """Called when a tool execution ends."""
        result = args[3] if len(args) > 3 else kwargs.get("result", None)

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

        # Record tool end for memory trace (raw result).
        try:
            record_tool_end(
                tool_name=str(tool_name or ""),
                result=result,
                agent_type=self.agent_type,
            )
        except Exception:
            pass

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

        if self.show_tools:
            print(f"{Colors.OKGREEN}✓{Colors.ENDC}")
        else:
            print(f"   {Colors.OKGREEN}✓ Done{Colors.ENDC}")

        if result is None:
            print(f"   {Colors.WARNING}📋 Result: (None){Colors.ENDC}")
            return

        try:
            result_str = None

            if isinstance(result, dict):
                success = result.get("success")
                if success is False:
                    error_msg = (
                        result.get("error") or result.get("stderr") or "Unknown error"
                    )
                    print(f"   {Colors.FAIL}❌ Error: {error_msg}{Colors.ENDC}")
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)
                elif success is True:
                    print(f"   {Colors.OKGREEN}✓ Success{Colors.ENDC}", end="")

                    key_info = []
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

                    if key_info:
                        print(f" | {Colors.OKCYAN}{' | '.join(key_info)}{Colors.ENDC}")
                    else:
                        print()

                for content_key in ["content", "output", "result", "data", "text"]:
                    if content_key in result:
                        result_str = str(result[content_key])
                        break

                if result_str is None:
                    result_str = json.dumps(result, indent=2, ensure_ascii=False)

            elif isinstance(result, str):
                result_str = result
            else:
                result_str = str(result)

            if result_str and len(result_str.strip()) > 0:
                if "Agent(name=" in result_str or "FunctionTool(name=" in result_str:
                    print(
                        f"   {Colors.OKCYAN}📋 Result: (Agent/Tool object){Colors.ENDC}"
                    )
                elif show_full_output:
                    print(f"   {Colors.OKCYAN}📋 Result (Full Error):{Colors.ENDC}")
                    print(f"     {Colors.FAIL}{result_str}{Colors.ENDC}")
                else:
                    truncated_result = _truncate_to_tokens(result_str, max_tokens=5000)
                    print(f"   {Colors.OKCYAN}📋 Result:{Colors.ENDC}")
                    lines = truncated_result.split("\n")
                    max_display_lines = 15
                    for line in lines[:max_display_lines]:
                        if len(line) > 200:
                            line = line[:197] + "..."
                        if line.strip():
                            print(f"     {Colors.OKBLUE}{line}{Colors.ENDC}")
                    if len(lines) > max_display_lines:
                        print(
                            f"     {Colors.WARNING}... ({len(lines) - max_display_lines} more lines){Colors.ENDC}"
                        )

        except Exception as e:
            print(
                f"   {Colors.FAIL}📋 Result parse error: {type(e).__name__}: {e}{Colors.ENDC}"
            )

    async def on_handoff(self, *args, **kwargs):
        """Called when control is handed off between agents."""
        from_agent = kwargs.get("from_agent", args[0] if args else "Unknown")
        to_agent = kwargs.get("to_agent", args[1] if len(args) > 1 else "Unknown")

        from_agent_name = (
            from_agent.name if hasattr(from_agent, "name") else str(from_agent)
        )
        to_agent_name = to_agent.name if hasattr(to_agent, "name") else str(to_agent)

        print(
            f"\n{Colors.WARNING}🔄 Handoff: {from_agent_name} → {to_agent_name}{Colors.ENDC}"
        )


def create_hooks(
    agent_type: str = "Agent",
    show_llm_responses: bool = True,
    show_tools: bool = True,
    show_tool_args: bool = True,
) -> VerboseRunHooks:
    return VerboseRunHooks(
        show_llm_responses=show_llm_responses,
        show_tools=show_tools,
        show_tool_args=show_tool_args,
        agent_type=agent_type,
    )
