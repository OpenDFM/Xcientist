from time import perf_counter
from typing import Any, Optional


class LigRuntime:
    """Thin wrapper around LigAgent chat/tool calls with op-level tracing."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def llm_text(
        self,
        *,
        session: Optional[Any],
        stage: str,
        op_name: str,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        started_at = perf_counter()
        resolved_model = model or getattr(self.agent, "model", "gpt-4.1")
        try:
            result = self.agent.chat(prompt, model=resolved_model, **kwargs)
            self._record(
                session,
                "llm_call",
                stage=stage,
                op_name=op_name,
                model=resolved_model,
                status="success",
                latency_ms=round((perf_counter() - started_at) * 1000.0, 2),
            )
            return result
        except Exception as exc:
            self._record(
                session,
                "llm_call",
                stage=stage,
                op_name=op_name,
                model=resolved_model,
                status="error",
                error=str(exc),
                latency_ms=round((perf_counter() - started_at) * 1000.0, 2),
            )
            raise

    def llm_json(
        self,
        *,
        session: Optional[Any],
        stage: str,
        op_name: str,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        raw = self.llm_text(
            session=session,
            stage=stage,
            op_name=op_name,
            prompt=prompt,
            model=model,
            **kwargs,
        )
        return self.agent._parse_json_response(raw)

    def tool_call(
        self,
        *,
        session: Optional[Any],
        stage: str,
        op_name: str,
        tool_name: str,
        **kwargs: Any,
    ) -> Any:
        started_at = perf_counter()
        try:
            result = self.agent.run_tool(name=tool_name, **kwargs)
            self._record(
                session,
                "tool_call",
                stage=stage,
                op_name=op_name,
                tool_name=tool_name,
                status="success",
                latency_ms=round((perf_counter() - started_at) * 1000.0, 2),
            )
            return result
        except Exception as exc:
            self._record(
                session,
                "tool_call",
                stage=stage,
                op_name=op_name,
                tool_name=tool_name,
                status="error",
                error=str(exc),
                latency_ms=round((perf_counter() - started_at) * 1000.0, 2),
            )
            raise

    def _record(self, session: Optional[Any], event_type: str, **payload: Any) -> None:
        if session is not None:
            session.record_event(event_type, **payload)
