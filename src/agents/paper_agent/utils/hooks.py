import json
import os
from typing import Any, Optional

from agents import RunHooks


class Colors:
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    MAGENTA = "\033[95m"


def _truncate(text: str, max_chars: int = 8000) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... (truncated, total chars: {len(s)})"


def _try_json_load(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def _truncate_to_tokens(text: str, max_tokens: int = 300) -> str:
    # Keep this lightweight: approximate tokens by words to avoid extra deps.
    s = str(text or "").strip()
    if not s:
        return ""
    parts = s.split()
    if len(parts) <= int(max_tokens):
        return s
    return " ".join(parts[: int(max_tokens)]) + " ... (truncated)"


def _as_plain_obj(obj: Any) -> Any:
    """
    Mirror experiment_agent: normalize pydantic/dataclass/objects to plain dict/list when possible.
    """
    if obj is None:
        return None
    try:
        if isinstance(obj, (dict, list, str, int, float, bool)):
            return obj
    except Exception:
        pass
    try:
        if hasattr(obj, "model_dump"):
            d = obj.model_dump()
            if isinstance(d, (dict, list)) and d:
                return d
    except Exception:
        pass
    try:
        if hasattr(obj, "dict"):
            d = obj.dict()
            if isinstance(d, (dict, list)) and d:
                return d
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
    """
    Align with experiment_agent (exact logic style):
    - Handle Responses-style ModelResponse.output which is often a list of items with nested content.
    - Extract reasoning_details / reasoning content blocks / <think> tags.
    """
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
        self.show_llm_responses = bool(show_llm_responses)
        self.show_tools = bool(show_tools)
        self.show_tool_args = bool(show_tool_args)
        self.agent_type = str(agent_type or "Agent")
        self.current_agent_name: Optional[str] = None
        self.turn_count = 0

    async def on_agent_start(self, *args, **kwargs):
        self.turn_count = 0
        agent_name = None
        if "agent" in kwargs and hasattr(kwargs["agent"], "name"):
            agent_name = kwargs["agent"].name
        if agent_name is None:
            agent_name = self.agent_type
        self.current_agent_name = agent_name

        color = Colors.OKCYAN
        if "Architect" in self.agent_type:
            color = Colors.MAGENTA
        elif "Writer" in self.agent_type:
            color = Colors.OKGREEN

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
        if self.turn_count != 1:
            return
        print(f"{Colors.WARNING}📤 LLM Request (Turn {self.turn_count}){Colors.ENDC}")

    async def on_llm_end(self, *args, **kwargs):
        if not self.show_llm_responses:
            return
        response = kwargs.get("response", args[0] if args else None)
        llm_response = args[-1] if args else response
        print(f"{Colors.OKGREEN}✓ LLM Response{Colors.ENDC}")

        # Align with experiment_agent: optionally display model-provided reasoning (gated by env).
        # NOTE: This prints only when SHOW_LLM_REASONING is enabled.
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
                reasoning = _extract_reasoning_from_modelresponse_output(
                    getattr(llm_response, "output", None)
                )

            show_reasoning_env = (
                os.environ.get("SHOW_LLM_REASONING", "").strip().lower()
            )
            show_reasoning = show_reasoning_env in ("1", "true", "yes", "y", "on")
            if show_reasoning and reasoning:
                truncated_reasoning = _truncate_to_tokens(
                    str(reasoning), max_tokens=300
                )
                if truncated_reasoning:
                    print(f"{Colors.OKCYAN}🧠 Thinking:{Colors.ENDC}")
                    print(f"{Colors.WARNING}{truncated_reasoning}{Colors.ENDC}\n")
            elif show_reasoning and not reasoning:
                print(
                    f"{Colors.OKCYAN}🧠 Thinking:{Colors.ENDC} {Colors.WARNING}(no reasoning provided by model/provider){Colors.ENDC}\n"
                )

            if tool_calls:
                print(
                    f"{Colors.OKCYAN}🔧 Will call tools: {', '.join(tool_calls)}{Colors.ENDC}\n"
                )
        except Exception:
            pass

    async def on_tool_start(self, *args, **kwargs):
        tool_ctx = args[0] if args else None
        tool_name = getattr(tool_ctx, "tool_name", None) or kwargs.get(
            "tool_name", "Unknown"
        )
        print(f"\n{Colors.OKGREEN}🔧 TOOL: {tool_name}{Colors.ENDC}")
        if not self.show_tool_args:
            return
        arguments = None
        if hasattr(tool_ctx, "tool_arguments"):
            arguments = _try_json_load(tool_ctx.tool_arguments)
        if isinstance(arguments, dict):
            for k, v in arguments.items():
                s = _truncate(str(v), max_chars=300).replace("\n", " ")
                print(f"   {Colors.OKCYAN}{k}:{Colors.ENDC} {s}")
        if self.show_tools:
            print(f"   {Colors.WARNING}Processing...{Colors.ENDC}", end=" ", flush=True)

    async def on_tool_end(self, *args, **kwargs):
        result = args[3] if len(args) > 3 else kwargs.get("result", None)
        if self.show_tools:
            print(f"{Colors.OKGREEN}✓{Colors.ENDC}")
        if result is None:
            return
        try:
            if isinstance(result, dict) and result.get("success") is False:
                msg = result.get("error") or result.get("stderr") or "Unknown error"
                print(f"   {Colors.FAIL}❌ Error: {msg}{Colors.ENDC}")
                return
            if isinstance(result, dict) and result.get("success") is True:
                msg = result.get("message") or ""
                if msg:
                    print(
                        f"   {Colors.OKCYAN}{_truncate(str(msg), max_chars=200)}{Colors.ENDC}"
                    )
                return
        except Exception:
            pass


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
