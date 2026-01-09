import os
import sys
import threading
import contextvars
import hashlib
import time
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.agents.experiment_agent.shared.utils.config import (
    MEMORY_ENABLED,
    MEMORY_EMBEDDING_MODEL_PATH,
    MEMORY_LLM_NAME,
    MEMORY_QUERY_METHOD,
    MEMORY_SHARED_DIR,
    MEMORY_MAX_SLOTS_PER_TASK,
    MEMORY_WRITEBACK_ENABLED,
    MEMORY_TOOL_LOGS_ENABLED,
    MEMORY_PROMPT_INJECTION_ENABLED,
    get_openai_config,
)


@dataclass(frozen=True)
class AgentMemoryContext:
    project_root: str
    stage: str
    agent_type: str
    reads_enabled: bool
    target_file_path: str
    target_file_abs: str
    purpose: str
    dependencies: List[str]
    feedback: str


_ctx_var: contextvars.ContextVar[Optional[AgentMemoryContext]] = contextvars.ContextVar(
    "experiment_agent_code_worker_memory_context",
    default=None,
)
_injected_keys_var: contextvars.ContextVar[set] = contextvars.ContextVar(
    "experiment_agent_code_worker_memory_injected_keys",
    default=set(),
)
_event_buffer_var: contextvars.ContextVar[List[Dict[str, Any]]] = (
    contextvars.ContextVar(
        "experiment_agent_code_worker_memory_event_buffer",
        default=[],
    )
)


def record_trace_event(kind: str, payload: Dict[str, Any]) -> None:
    """
    Record a trace event from hooks (LLM/tool lifecycle).
    This is intended to capture a fuller trajectory than tool-return-only recording.
    """
    ctx = _ctx_var.get()
    if ctx is None:
        return
    buf = _event_buffer_var.get()
    if not isinstance(buf, list):
        buf = []
    try:
        buf.append({"kind": str(kind or "event"), "payload": payload})
    finally:
        _event_buffer_var.set(buf)


def record_llm_start(
    messages: Any, agent_type: str = "", turn: Optional[int] = None
) -> None:
    record_trace_event(
        "llm_start",
        {"agent_type": str(agent_type or ""), "turn": turn, "messages": messages},
    )


def record_llm_end(
    content: Optional[str],
    reasoning: Optional[str],
    tool_calls: Optional[List[str]] = None,
    agent_type: str = "",
    turn: Optional[int] = None,
) -> None:
    record_trace_event(
        "llm_end",
        {
            "agent_type": str(agent_type or ""),
            "turn": turn,
            "content": (
                content
                if isinstance(content, str)
                else ("" if content is None else str(content))
            ),
            "reasoning": (
                reasoning
                if isinstance(reasoning, str)
                else ("" if reasoning is None else str(reasoning))
            ),
            "tool_calls": list(tool_calls or []),
        },
    )


def record_tool_start(tool_name: str, arguments: Any, agent_type: str = "") -> None:
    record_trace_event(
        "tool_start",
        {
            "agent_type": str(agent_type or ""),
            "tool_name": str(tool_name or ""),
            "arguments": arguments,
        },
    )


def record_tool_end(tool_name: str, result: Any, agent_type: str = "") -> None:
    def _first_nonempty_line(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        for ln in text.splitlines():
            ln2 = (ln or "").strip()
            if ln2:
                return ln2
        return ""

    def _summarize_result(tool: str, res: Any) -> Dict[str, Any]:
        """
        Summarize tool results for memory extraction.
        We prefer triggers + follow-up-relevant fields over raw logs.
        """
        tool = str(tool or "")
        if not isinstance(res, dict):
            return {
                "type": "raw",
                "text": (
                    "" if res is None else (res if isinstance(res, str) else str(res))
                ),
            }

        success = res.get("success", None)
        summary: Dict[str, Any] = {"success": success}

        if tool == "bash":
            stdout = str(res.get("stdout") or "")
            stderr = str(res.get("stderr") or "")
            rc = res.get("return_code", None)
            summary["return_code"] = rc
            if (
                (success is False)
                or (isinstance(rc, int) and rc != 0)
                or stderr.strip()
            ):
                trigger = _first_nonempty_line(stderr) or _first_nonempty_line(stdout)
                if trigger:
                    summary["trigger"] = trigger
                if stderr.strip():
                    summary["stderr_present"] = True
            else:
                if stdout.strip():
                    summary["stdout_present"] = True
            return summary

        if tool == "file_viewer":
            summary["file_path"] = str(res.get("file_path") or "")
            summary["total_lines"] = res.get("total_lines", None)
            summary["showing"] = str(res.get("showing") or "")
            content = str(res.get("content") or "")
            if content.strip():
                summary["content_present"] = True
            err = res.get("error", None)
            if err:
                summary["trigger"] = str(err)
            return summary

        if tool == "write_file":
            summary["file_path"] = str(res.get("file_path") or "")
            msg = res.get("message", None)
            if msg:
                summary["message"] = str(msg)
            err = res.get("error", None)
            if err:
                summary["trigger"] = str(err)
            return summary

        # Generic tool
        for k in ("error", "stderr", "message"):
            v = res.get(k, None)
            if isinstance(v, str) and v.strip():
                summary["trigger"] = v
                break
        return summary

    record_trace_event(
        "tool_end",
        {
            "agent_type": str(agent_type or ""),
            "tool_name": str(tool_name or ""),
            # Keep RAW tool result for full trajectory fidelity (no truncation/sampling).
            "result": result,
            # Also keep a compact summary to help "what to do next" extraction reliably.
            "result_summary": _summarize_result(str(tool_name or ""), result),
        },
    )


class _ContextSetter:
    def __init__(self, ctx: AgentMemoryContext):
        self._ctx = ctx
        self._token_ctx = None
        self._token_keys = None
        self._token_events = None

    def __enter__(self):
        self._token_ctx = _ctx_var.set(self._ctx)
        self._token_keys = _injected_keys_var.set(set())
        self._token_events = _event_buffer_var.set([])
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._token_events is not None:
                _event_buffer_var.reset(self._token_events)
            if self._token_keys is not None:
                _injected_keys_var.reset(self._token_keys)
        finally:
            if self._token_ctx is not None:
                _ctx_var.reset(self._token_ctx)
        return False


def set_code_worker_memory_context(
    project_root: str,
    target_file_path: str,
    purpose: str,
    dependencies: Optional[List[str]] = None,
    feedback: str = "",
) -> _ContextSetter:
    project_root_abs = os.path.abspath(project_root or "")
    target_file_path_norm = (target_file_path or "").replace("\\", "/")
    target_file_abs = os.path.abspath(
        os.path.join(project_root_abs, target_file_path_norm)
    )
    ctx = AgentMemoryContext(
        project_root=project_root_abs,
        stage="code_worker",
        agent_type="CodeWorker",
        reads_enabled=True,
        target_file_path=target_file_path_norm,
        target_file_abs=target_file_abs,
        purpose=str(purpose or ""),
        dependencies=list(dependencies or []),
        feedback=str(feedback or ""),
    )
    return _ContextSetter(ctx)


def set_agent_memory_context(
    project_root: str = "",
    stage: str = "",
    agent_type: str = "",
    purpose: str = "",
    dependencies: Optional[List[str]] = None,
    feedback: str = "",
) -> _ContextSetter:
    project_root_abs = os.path.abspath(project_root or "")
    ctx = AgentMemoryContext(
        project_root=project_root_abs,
        stage=str(stage or "").strip()
        or str(agent_type or "").strip().lower()
        or "agent",
        agent_type=str(agent_type or ""),
        reads_enabled=True,
        target_file_path="",
        target_file_abs="",
        purpose=str(purpose or ""),
        dependencies=list(dependencies or []),
        feedback=str(feedback or ""),
    )
    return _ContextSetter(ctx)


def get_current_memory_context() -> Optional[AgentMemoryContext]:
    return _ctx_var.get()


def _get_shared_memory_root() -> str:
    return str(MEMORY_SHARED_DIR or "")


def _ensure_memory_import_path() -> None:
    """
    Ensure repo root is on sys.path so `import src.memory` resolves.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class _SharedMemoryStores:
    def __init__(self):
        self._lock = threading.RLock()
        self._initialized = False
        self._disabled_reason: Optional[str] = None
        self.semantic = None
        self.episodic = None
        self.procedural = None
        self.slot_process = None

    def _try_init(self) -> None:
        with self._lock:
            if self._initialized or self._disabled_reason is not None:
                return

            if not bool(MEMORY_ENABLED):
                self._disabled_reason = (
                    "memory disabled by EXPERIMENT_AGENT_MEMORY_ENABLED"
                )
                return

            try:
                _ensure_memory_import_path()
                from src.memory.api.faiss_memory_system_api import FAISSMemorySystem
                from src.memory.api.slot_process_api import SlotProcess
                from src.memory.memory_system.llm import OpenAIClient
            except Exception as e:
                self._disabled_reason = f"memory import failed: {type(e).__name__}: {e}"
                return

            root = _get_shared_memory_root()
            try:
                os.makedirs(root, exist_ok=True)
            except Exception as e:
                self._disabled_reason = (
                    f"shared memory dir create failed: {type(e).__name__}: {e}"
                )
                return

            def _load_or_empty(memory_type: str):
                llm_cfg = {}
                try:
                    llm_cfg = get_openai_config(model=MEMORY_LLM_NAME)
                except Exception:
                    llm_cfg = {}

                store = FAISSMemorySystem(
                    memory_type=memory_type,
                    model_path=str(
                        MEMORY_EMBEDDING_MODEL_PATH or "./.cache/all-MiniLM-L6-v2"
                    ),
                    llm_name=str(MEMORY_LLM_NAME or "gpt-4.1-mini"),
                    openai_api_key=llm_cfg.get("api_key"),
                    openai_base_url=llm_cfg.get("base_url"),
                )
                path = os.path.join(root, memory_type)
                if os.path.exists(os.path.join(path, "faiss.index")):
                    try:
                        store.load(path)
                    except Exception:
                        # If load fails, keep empty store to avoid hard-crash during tool calls.
                        pass
                return store

            try:
                self.semantic = _load_or_empty("semantic")
                self.episodic = _load_or_empty("episodic")
                self.procedural = _load_or_empty("procedural")
                # A shared SlotProcess instance for converting trace slots into different record types.
                llm_cfg = {}
                try:
                    llm_cfg = get_openai_config(model=MEMORY_LLM_NAME)
                except Exception:
                    llm_cfg = {}
                llm_model = OpenAIClient(
                    model=str(MEMORY_LLM_NAME or "gpt-4.1-mini"),
                    api_key=llm_cfg.get("api_key"),
                    base_url=llm_cfg.get("base_url"),
                )
                self.slot_process = SlotProcess(llm_model=llm_model)
                self._initialized = True
            except Exception as e:
                self._disabled_reason = f"memory init failed: {type(e).__name__}: {e}"

    def is_ready(self) -> bool:
        self._try_init()
        return bool(self._initialized) and self._disabled_reason is None

    def disabled_reason(self) -> Optional[str]:
        self._try_init()
        return self._disabled_reason


_STORES = _SharedMemoryStores()


def _stable_key(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def _truncate(text: str, limit: int = 1200) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= limit:
        return t
    return t[: limit - 16] + "... <truncated>"


def _append_writeback_error_log(root: str, message: str) -> None:
    """
    Best-effort logging for writeback failures (never raise).
    """
    try:
        if not root:
            return
        os.makedirs(root, exist_ok=True)
        log_path = os.path.join(root, "writeback_errors.log")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        line = f"{ts}\t{(message or '').strip()}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        return


def _format_record_full(rec: Any) -> str:
    """
    Format a memory record with full useful fields (no truncation, no id).
    """
    if rec is None:
        return ""
    lines: List[str] = []

    # SemanticRecord
    if hasattr(rec, "summary") and hasattr(rec, "detail") and not hasattr(rec, "steps"):
        summary = str(getattr(rec, "summary", "") or "").strip()
        detail = str(getattr(rec, "detail", "") or "").strip()
        tags = getattr(rec, "tags", None)
        if summary:
            lines.append(f"summary: {summary}")
        if detail:
            lines.append("detail:")
            for ln in detail.splitlines():
                lines.append(f"  {ln}")
        if tags:
            lines.append(f"tags: {list(tags)}")
        return "\n".join(lines).strip()

    # EpisodicRecord
    if hasattr(rec, "stage") and hasattr(rec, "summary") and hasattr(rec, "detail"):
        stage = str(getattr(rec, "stage", "") or "").strip()
        summary = str(getattr(rec, "summary", "") or "").strip()
        detail = getattr(rec, "detail", None)
        tags = getattr(rec, "tags", None)
        created_at = str(getattr(rec, "created_at", "") or "").strip()
        if stage:
            lines.append(f"stage: {stage}")
        if summary:
            lines.append(f"summary: {summary}")
        if detail is not None:
            lines.append("detail:")
            if isinstance(detail, dict):
                for k, v in detail.items():
                    lines.append(f"  {k}: {v}")
            else:
                for ln in str(detail).splitlines():
                    lines.append(f"  {ln}")
        if tags:
            lines.append(f"tags: {list(tags)}")
        if created_at:
            lines.append(f"created_at: {created_at}")
        return "\n".join(lines).strip()

    # ProceduralRecord
    if hasattr(rec, "name") and hasattr(rec, "description") and hasattr(rec, "steps"):
        name = str(getattr(rec, "name", "") or "").strip()
        description = str(getattr(rec, "description", "") or "").strip()
        steps = getattr(rec, "steps", None)
        code = getattr(rec, "code", None)
        tags = getattr(rec, "tags", None)
        created_at = str(getattr(rec, "created_at", "") or "").strip()
        updated_at = str(getattr(rec, "updated_at", "") or "").strip()
        if name:
            lines.append(f"name: {name}")
        if description:
            lines.append("description:")
            for ln in description.splitlines():
                lines.append(f"  {ln}")
        if steps:
            lines.append("steps:")
            for s in list(steps):
                lines.append(f"  - {s}")
        if code:
            lines.append("code:")
            for ln in str(code).splitlines():
                lines.append(f"  {ln}")
        if tags:
            lines.append(f"tags: {list(tags)}")
        if created_at:
            lines.append(f"created_at: {created_at}")
        if updated_at:
            lines.append(f"updated_at: {updated_at}")
        return "\n".join(lines).strip()

    # Fallback
    return str(rec)


def _format_hits(title: str, hits: List[Tuple[float, Any]], max_items: int = 3) -> str:
    if not hits:
        return ""
    lines: List[str] = [f"- {title}:"]
    for _score, rec in hits[:max_items]:
        full = _format_record_full(rec)
        if not full:
            continue
        lines.append("  -")
        for ln in full.splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines).strip()


def _build_query_for_target_file(ctx: AgentMemoryContext) -> str:
    deps = ", ".join([d.replace("\\", "/") for d in (ctx.dependencies or []) if d])
    parts = [
        f"target_file={ctx.target_file_path}",
        f"purpose={ctx.purpose}",
    ]
    if deps:
        parts.append(f"dependencies={deps}")
    if ctx.feedback:
        parts.append(f"feedback={_truncate(ctx.feedback, limit=500)}")
    return "\n".join(parts)


def _build_query_for_bash_failure(
    ctx: AgentMemoryContext, command: str, stderr: str
) -> str:
    parts = [
        f"target_file={ctx.target_file_path}",
        f"purpose={ctx.purpose}",
        f"command={_truncate(command, limit=400)}",
        f"stderr={_truncate(stderr, limit=1200)}",
    ]
    return "\n".join(parts)


def maybe_augment_tool_result(
    tool_name: str,
    tool_args: Dict[str, Any],
    result: Any,
) -> Any:
    ctx = _ctx_var.get()
    if ctx is None:
        return result

    injected = _injected_keys_var.get()
    if not isinstance(injected, set):
        injected = set()

    if not _STORES.is_ready():
        # Do not spam the model with this on every call.
        key = "memory:disabled"
        if key in injected:
            return result
        injected.add(key)
        _injected_keys_var.set(injected)
        reason = _STORES.disabled_reason() or "unknown"
        hint = f"[MEMORY] disabled: {reason}"
        return _attach_hint(result, hint)

    hint_text = ""
    log_text = ""

    _record_tool_event(tool_name=tool_name, tool_args=tool_args, result=result)

    # Only CodeWorker is allowed to read/inject memory during tool calls.
    if not bool(getattr(ctx, "reads_enabled", False)):
        return result

    if tool_name == "file_viewer":
        file_path = str(tool_args.get("file_path") or "")
        full_path = file_path
        if not os.path.isabs(full_path):
            full_path = os.path.abspath(os.path.join(ctx.project_root, file_path))
        full_path = os.path.abspath(full_path)

        if full_path == ctx.target_file_abs:
            key = "memory:target_file_first_read"
            if key not in injected:
                injected.add(key)
                _injected_keys_var.set(injected)
                q = _build_query_for_target_file(ctx)
                hint_text, log_text = _retrieve_memory_hints_with_log(
                    q, trigger="file_viewer:first_target_file_read"
                )

    elif tool_name == "bash":
        if isinstance(result, dict):
            ok = bool(result.get("success"))
            stderr = str(result.get("stderr") or "")
            if (not ok) and stderr.strip():
                cmd = str(tool_args.get("command") or "")
                sig = _stable_key(cmd + "\n" + stderr[:200])
                key = f"memory:bash_fail:{sig}"
                if key not in injected:
                    injected.add(key)
                    _injected_keys_var.set(injected)
                    q = _build_query_for_bash_failure(ctx, cmd, stderr)
                    hint_text, log_text = _retrieve_memory_hints_with_log(
                        q, trigger="bash:failure"
                    )

    if (not hint_text) and (not (MEMORY_TOOL_LOGS_ENABLED and log_text)):
        return result

    out = result
    if hint_text:
        out = _attach_hint(out, hint_text)
    if MEMORY_TOOL_LOGS_ENABLED and log_text:
        out = _attach_log(out, log_text)
    return out


def _attach_hint(result: Any, hint_text: str) -> Any:
    hint_text = (hint_text or "").strip()
    if not hint_text:
        return result
    if isinstance(result, dict):
        out = dict(result)
        existing = str(out.get("memory_hints") or "").strip()
        out["memory_hints"] = (
            hint_text if not existing else (existing + "\n\n" + hint_text)
        )
        return out
    return {"success": True, "result": result, "memory_hints": hint_text}


def _attach_log(result: Any, log_text: str) -> Any:
    log_text = (log_text or "").strip()
    if not log_text:
        return result
    if isinstance(result, dict):
        out = dict(result)
        existing = str(out.get("memory_log") or "").strip()
        out["memory_log"] = log_text if not existing else (existing + "\n" + log_text)
        return out
    return {"success": True, "result": result, "memory_log": log_text}


def _record_tool_event(tool_name: str, tool_args: Dict[str, Any], result: Any) -> None:
    """
    Record a small set of high-signal events.

    Full tool traces (args/result) are already captured via hooks: tool_start/tool_end.
    Keeping this minimal avoids duplicating the same trajectory twice in the buffer.
    """
    ctx = _ctx_var.get()
    if ctx is None:
        return

    buf = _event_buffer_var.get()
    if not isinstance(buf, list):
        buf = []

    try:
        # High-signal markers (useful for quick scans / compact procedural extraction).
        if tool_name == "file_viewer":
            file_path = str(tool_args.get("file_path") or "")
            full_path = file_path
            if not os.path.isabs(full_path):
                full_path = os.path.abspath(os.path.join(ctx.project_root, file_path))
            full_path = os.path.abspath(full_path)
            if full_path == ctx.target_file_abs:
                buf.append(
                    {"kind": "target_file_viewed", "file_path": ctx.target_file_path}
                )

        if tool_name == "bash" and isinstance(result, dict):
            cmd = str(tool_args.get("command") or "")
            ok = bool(result.get("success"))
            stderr = str(result.get("stderr") or "")
            stdout = str(result.get("stdout") or "")
            if ok:
                buf.append(
                    {
                        "kind": "bash_ok",
                        "command": _truncate(cmd, 400),
                        "stdout": _truncate(stdout, 600),
                    }
                )
            else:
                if stderr.strip():
                    buf.append(
                        {
                            "kind": "bash_fail",
                            "command": _truncate(cmd, 400),
                            "stderr": _truncate(stderr, 1200),
                        }
                    )
    finally:
        _event_buffer_var.set(buf)


async def writeback_current_task_async(
    success: bool, error: str = "", final_output: str = ""
) -> None:
    ctx = _ctx_var.get()
    if ctx is None:
        return
    if not _STORES.is_ready():
        return
    if not bool(MEMORY_WRITEBACK_ENABLED):
        return

    buf = _event_buffer_var.get()
    if not isinstance(buf, list):
        buf = []

    agent_id = "experiment_agent"
    root = _get_shared_memory_root()

    tags = [
        "source:experiment_agent",
        f"agent:{(ctx.agent_type or ctx.stage or 'agent').strip()}",
    ]
    if ctx.target_file_path:
        tags.append(f"file:{ctx.target_file_path}")
    tags.append("result:success" if success else "result:failed")

    # Build a generic trace slot and let src/memory extract semantic/episodic/procedural memories.
    trace: Dict[str, Any] = {
        "target_file": ctx.target_file_path,
        "purpose": ctx.purpose,
        "dependencies": ctx.dependencies,
        "success": bool(success),
        "error": _truncate(error, 1200) if error else "",
        "final_output": _truncate(final_output, 4000) if final_output else "",
        "tool_trace": buf,
    }

    def _build_tool_signals(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build compact (trigger -> next step friendly) signals by pairing tool_start/tool_end.
        We keep only follow-up-relevant fields (command/path, success, return_code, trigger).
        """
        pending: List[Dict[str, Any]] = []
        signals: List[Dict[str, Any]] = []

        for ev in events:
            if not isinstance(ev, dict):
                continue
            kind = str(ev.get("kind") or "")
            payload = ev.get("payload", None)
            if not isinstance(payload, dict):
                continue
            if kind == "tool_start":
                pending.append(payload)
                continue
            if kind != "tool_end":
                continue

            tool_name = str(payload.get("tool_name") or "")
            # Prefer compact summary if present; fall back to raw result.
            result = payload.get("result_summary", payload.get("result", None))
            args = None
            # Pair with the most recent unmatched start of the same tool.
            for i in range(len(pending) - 1, -1, -1):
                if str(pending[i].get("tool_name") or "") == tool_name:
                    args = pending.pop(i).get("arguments", None)
                    break

            sig: Dict[str, Any] = {"tool_name": tool_name}
            if isinstance(result, dict):
                if "success" in result:
                    sig["success"] = result.get("success")
                if "return_code" in result:
                    sig["return_code"] = result.get("return_code")
                if "trigger" in result:
                    sig["trigger"] = result.get("trigger")
            if isinstance(args, dict):
                if tool_name == "bash":
                    sig["command"] = str(args.get("command") or "")
                    wd = str(args.get("working_dir") or "")
                    if wd:
                        sig["working_dir"] = wd
                elif tool_name in ("file_viewer", "write_file", "edit_file"):
                    fp = args.get("file_path", None)
                    if fp is not None:
                        sig["file_path"] = str(fp)

            # Keep only non-empty signals.
            if any(
                k in sig for k in ("trigger", "return_code", "command", "file_path")
            ):
                signals.append(sig)

        return signals[:60]

    # Add compact signals to help the memory extractor focus on "what to do next".
    try:
        trace["signals"] = _build_tool_signals(buf)
    except Exception:
        trace["signals"] = []

    try:
        from src.memory.memory_system import WorkingSlot
    except Exception:
        return

    if _STORES.slot_process is None:
        return

    sp = _STORES.slot_process

    max_slots = int(MEMORY_MAX_SLOTS_PER_TASK or 50)

    # Feed the full trajectory to the memory LLM and extract multiple concrete slots.
    trajectory_payload: Dict[str, Any] = {
        "context": {
            "stage": str(ctx.stage or ""),
            "agent_type": str(ctx.agent_type or ""),
            "target_file": str(ctx.target_file_path or ""),
            "purpose": str(ctx.purpose or ""),
            "dependencies": list(ctx.dependencies or []),
        },
        "trace": trace,
    }

    extracted_slots: List[WorkingSlot] = []
    try:
        extracted_slots = await sp.transfer_trajectory_to_working_slots(
            trajectory=trajectory_payload, max_slots=max_slots
        )
    except Exception:
        extracted_slots = []

    # Fallback: keep prior single-slot behavior if extraction fails or yields nothing.
    if not extracted_slots:
        extracted_slots = [
            WorkingSlot(
                stage=str(ctx.stage or "agent"),
                topic=ctx.target_file_path or str(ctx.stage or "agent_task"),
                summary=_truncate(
                    (
                        f"Task({ctx.stage}): {ctx.target_file_path or ctx.purpose or 'run'}. "
                        f"Result: {'success' if success else 'failed'}."
                    ),
                    400,
                ),
                attachments={"trace": trace},
                tags=list(tags),
            )
        ]

    # Enrich slots with base tags (do NOT inject full trace again to keep memories compact).
    normalized_slots: List[WorkingSlot] = []
    for s in extracted_slots:
        try:
            merged_tags = list(
                dict.fromkeys(list(tags) + list(getattr(s, "tags", []) or []))
            )
            normalized_slots.append(
                WorkingSlot(
                    stage=str(getattr(s, "stage", "") or "writeback"),
                    topic=str(
                        getattr(s, "topic", "") or ctx.target_file_path or "task"
                    ),
                    summary=str(getattr(s, "summary", "") or "").strip(),
                    attachments=getattr(s, "attachments", {}) or {},
                    tags=merged_tags,
                )
            )
        except Exception:
            continue

    # Route to memory systems and persist (batched).
    # IMPORTANT: SlotProcess may depend on external LLM config; failures here used to silently
    # stop writeback entirely (caller swallows exceptions). We fallback to a minimal episodic record.
    inputs = []
    try:
        sp.clear_container()
        sp.filtered_slot_container = []
        sp.routed_slot_container = []
        for s in normalized_slots:
            sp.add_slot(s)
        routed_slot_container = await sp.filter_and_route_slots()
        inputs = await sp.generate_long_term_memory(routed_slot_container)
    except Exception as e:
        msg = (
            "writeback slot_process failed"
            f" stage={str(ctx.stage or '')}"
            f" agent_type={str(ctx.agent_type or '')}"
            f" target_file={str(ctx.target_file_path or '')}"
            f" err={type(e).__name__}: {str(e)}"
        )
        _append_writeback_error_log(root=root, message=msg)
        inputs = [
            {
                "memory_type": "episodic",
                "input": {
                    "stage": str(ctx.stage or "writeback"),
                    "summary": _truncate(
                        (
                            f"Task({ctx.stage}): {ctx.target_file_path or ctx.purpose or 'run'}. "
                            f"Result: {'success' if success else 'failed'}."
                        ),
                        120,
                    ),
                    "detail": {
                        "situation": _truncate(str(ctx.purpose or "") or "", 400),
                        "actions": [],
                        "results": [
                            "success" if bool(success) else "failed",
                            _truncate(str(error or "") or "", 200),
                        ],
                        "metrics": {},
                        "artifacts": [],
                    },
                    "tags": list(tags),
                },
            }
        ]

    sem_recs = []
    epi_recs = []
    proc_recs = []
    for item in inputs:
        memory_type = item.get("memory_type")
        payload = item.get("input")
        if not isinstance(payload, dict):
            continue
        if memory_type == "semantic" and _STORES.semantic is not None:
            try:
                sem_recs.append(_STORES.semantic.instantiate_sem_record(**payload))
            except Exception:
                continue
        elif memory_type == "episodic" and _STORES.episodic is not None:
            try:
                epi_recs.append(_STORES.episodic.instantiate_epi_record(**payload))
            except Exception:
                continue
        elif memory_type == "procedural" and _STORES.procedural is not None:
            try:
                proc_recs.append(_STORES.procedural.instantiate_proc_record(**payload))
            except Exception:
                continue

    # Persist with one save per store.
    with _STORES._lock:
        if _STORES.semantic is not None and sem_recs:
            _STORES.semantic.add(sem_recs, agent_id=agent_id)
            _STORES.semantic.save(os.path.join(root, "semantic"))
        if _STORES.episodic is not None and epi_recs:
            _STORES.episodic.add(epi_recs, agent_id=agent_id)
            _STORES.episodic.save(os.path.join(root, "episodic"))
        if _STORES.procedural is not None and proc_recs:
            _STORES.procedural.add(proc_recs, agent_id=agent_id)
            _STORES.procedural.save(os.path.join(root, "procedural"))

    # Cleanup in-memory containers.
    sp.clear_container()
    sp.filtered_slot_container = []
    sp.routed_slot_container = []


def writeback_current_task(
    success: bool, error: str = "", final_output: str = ""
) -> None:
    # Backward compatible sync wrapper (best-effort).
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    try:
        if loop and loop.is_running():
            loop.create_task(
                writeback_current_task_async(
                    success=success, error=error, final_output=final_output
                )
            )
        else:
            asyncio.run(
                writeback_current_task_async(
                    success=success, error=error, final_output=final_output
                )
            )
    except Exception:
        pass


def _retrieve_memory_hints_with_log(
    query_text: str, trigger: str = ""
) -> Tuple[str, str]:
    q = (query_text or "").strip()
    if not q:
        return "", ""

    agent_id = "experiment_agent"

    preferred_method = (
        str(MEMORY_QUERY_METHOD or "").strip().lower()
        if str(MEMORY_QUERY_METHOD or "").strip().lower()
        in ("embedding", "bm25", "overlapping")
        else "embedding"
    )

    proc_hits: List[Tuple[float, Any]] = []
    sem_hits: List[Tuple[float, Any]] = []
    epi_hits: List[Tuple[float, Any]] = []
    proc_method = preferred_method
    sem_method = preferred_method
    epi_method = preferred_method
    t0 = time.perf_counter()

    try:
        if _STORES.procedural:
            try:
                proc_hits = _STORES.procedural.query(
                    q, method=preferred_method, limit=3, agent_id=agent_id
                )
            except Exception:
                proc_hits = _STORES.procedural.query(
                    q, method="bm25", limit=3, agent_id=agent_id
                )
                proc_method = "bm25"
        else:
            proc_hits = []
    except Exception:
        proc_hits = []
        proc_method = "error"
    try:
        if _STORES.semantic:
            try:
                sem_hits = _STORES.semantic.query(
                    q, method=preferred_method, limit=2, agent_id=agent_id
                )
            except Exception:
                sem_hits = _STORES.semantic.query(
                    q, method="bm25", limit=2, agent_id=agent_id
                )
                sem_method = "bm25"
        else:
            sem_hits = []
    except Exception:
        sem_hits = []
        sem_method = "error"
    try:
        if _STORES.episodic:
            try:
                epi_hits = _STORES.episodic.query(
                    q, method=preferred_method, limit=1, agent_id=agent_id
                )
            except Exception:
                epi_hits = _STORES.episodic.query(
                    q, method="bm25", limit=1, agent_id=agent_id
                )
                epi_method = "bm25"
        else:
            epi_hits = []
    except Exception:
        epi_hits = []
        epi_method = "error"

    def _top_ids(hits: List[Tuple[float, Any]], k: int) -> List[str]:
        ids: List[str] = []
        for _score, rec in (hits or [])[:k]:
            rid = getattr(rec, "id", None)
            if isinstance(rid, str) and rid:
                ids.append(rid)
        return ids

    query_preview = _truncate(q.replace("\n", " "), limit=220)
    proc_top_ids = _top_ids(proc_hits, k=3)
    sem_top_ids = _top_ids(sem_hits, k=2)
    epi_top_ids = _top_ids(epi_hits, k=1)

    blocks = []
    blocks.append(
        "[MEMORY_HINTS] Relevant cross-experiment memories (procedural > semantic > episodic)."
    )
    b1 = _format_hits("Procedural", proc_hits, max_items=3)
    b2 = _format_hits("Semantic", sem_hits, max_items=2)
    b3 = _format_hits("Episodic (recent)", epi_hits, max_items=1)
    for b in (b1, b2, b3):
        if b:
            blocks.append(b)

    hint = ""
    if len(blocks) > 1:
        # Do NOT truncate memory hints; caller explicitly wants full details.
        hint = "\n".join(blocks)

    dt_ms = int((time.perf_counter() - t0) * 1000)
    log = (
        "[MEMORY_LOG] "
        + (f"trigger={trigger}; " if trigger else "")
        + f"preferred={preferred_method}; used(proc/sem/epi)={proc_method}/{sem_method}/{epi_method}; "
        + f"hits(proc/sem/epi)={len(proc_hits)}/{len(sem_hits)}/{len(epi_hits)}; "
        + f"dt_ms={dt_ms}; "
        + f'query="{query_preview}"; '
        + f"top_ids(proc/sem/epi)={proc_top_ids}/{sem_top_ids}/{epi_top_ids}"
    )
    return hint, log


def _retrieve_memory_hints(query_text: str) -> str:
    hint, _log = _retrieve_memory_hints_with_log(query_text, trigger="")
    return hint


def retrieve_memory_for_worker_prompt(
    target_file_path: str,
    purpose: str,
    dependencies: Optional[List[str]] = None,
    feedback: str = "",
) -> str:
    """
    Retrieve memory hints for injecting into the CodeWorker prompt (pre-tool-call).

    Returns:
        A compact memory hint block, or "" if disabled/unavailable.
    """
    if not bool(MEMORY_ENABLED) or not bool(MEMORY_PROMPT_INJECTION_ENABLED):
        return ""
    if not _STORES.is_ready():
        return ""

    deps = ", ".join([d.replace("\\", "/") for d in (dependencies or []) if d])
    norm_target = str(target_file_path or "").replace("\\", "/")
    parts = [
        f"target_file={norm_target}",
        f"purpose={str(purpose or '')}",
    ]
    if deps:
        parts.append(f"dependencies={deps}")
    if feedback:
        parts.append(f"feedback={_truncate(str(feedback or ''), limit=500)}")
    q = "\n".join(parts).strip()
    return _retrieve_memory_hints(q)


def retrieve_memory_for_agent_prompt(
    agent_type: str,
    stage: str,
    purpose: str = "",
    user_prompt: str = "",
    feedback: str = "",
) -> str:
    """
    Generic memory retrieval for prompt injection for any agent.
    """
    if not bool(MEMORY_ENABLED) or not bool(MEMORY_PROMPT_INJECTION_ENABLED):
        return ""
    if not _STORES.is_ready():
        return ""

    parts = [
        f"agent_type={str(agent_type or '')}",
        f"stage={str(stage or '')}",
    ]
    if purpose:
        parts.append(f"purpose={_truncate(str(purpose or ''), limit=400)}")
    if feedback:
        parts.append(f"feedback={_truncate(str(feedback or ''), limit=400)}")
    if user_prompt:
        parts.append(f"user_prompt={_truncate(str(user_prompt or ''), limit=800)}")
    q = "\n".join(parts).strip()
    return _retrieve_memory_hints(q)
