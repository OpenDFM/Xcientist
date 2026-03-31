"""Shared LigAgent utilities for reference context assembly and JSON response parsing."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional

from src.agents.idea_agent.utils.core.json_utils import pretty_json
from src.agents.idea_agent.utils.core.response_parsing import parse_json_response
from src.agents.idea_agent.agent.artifacts import (
    ensure_artifact_structure,
)


def collect_paper_context_entries(
    artifact: Dict[str, Any],
    reference_batches: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    del artifact
    entries: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for batch in reference_batches or []:
        for reference in batch or []:
            if not isinstance(reference, dict):
                continue
            node_id = str(
                reference.get("node_id")
                or reference.get("paper_id")
                or reference.get("title")
                or ""
            ).strip()
            if not node_id or node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            summary = str(reference.get("summary") or "").strip()
            insight = str(reference.get("insight") or "").strip()
            if insight and insight not in summary:
                summary = f"{summary} Insight: {insight}".strip()
            if not summary:
                summary = "No summary available."
            entries.append(
                {
                    "paper_id": node_id,
                    "title": reference.get("title") or reference.get("paper_title") or node_id,
                    "summary": summary,
                    "source": reference.get("source") or "survey_keynote",
                    "authors": reference.get("authors") or [],
                }
            )
    return entries


def generate_idea_introduction(
    chat_fn,
    prompt_template: str,
    model: str,
    topic: str,
    best_entry: Dict[str, Any],
    paper_entries: List[Dict[str, Any]],
    mature_idea: str,
    logger,
) -> str:
    entries = paper_entries or []
    if not entries:
        return fallback_introduction_text(best_entry, entries, mature_idea)
    prompt = prompt_template.format(
        topic=topic,
        mature_idea=mature_idea or "",
        idea=pretty_json(best_entry),
        papers=pretty_json(entries),
    )
    try:
        response = chat_fn(prompt, temperature=0.3, max_output_tokens=65536, model=model)
        payload = parse_json_response(response)
        intro = payload.get("introduction") or payload.get("intro")
        if intro:
            return intro.strip()
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Introduction generation failed: %s", exc)
    return fallback_introduction_text(best_entry, entries, mature_idea)


def fallback_introduction_text(
    best_entry: Dict[str, Any], paper_entries: List[Dict[str, Any]], mature_idea: str
) -> str:
    title = best_entry.get("title", "This work")
    abstract = best_entry.get("abstract") or ""
    mature_anchor = str(mature_idea or "").strip()
    if mature_anchor:
        intro_lines = [
            f"{title} refines the mature idea by repairing a concrete limitation in the current design. {abstract}".strip()
        ]
    else:
        intro_lines = [
            f"{title} builds on recent literature to tackle the current topic. {abstract}".strip()
        ]
    if paper_entries:
        cite_lines = []
        for entry in paper_entries:
            cite_lines.append(
                f"- {entry.get('title') or entry.get('paper_id')}: {entry.get('summary', 'No summary available.')}"
            )
        intro_lines.append("Key references informing this idea:\n" + "\n".join(cite_lines))
    return "\n\n".join(intro_lines)


def align_public_idea_entry(
    chat_fn,
    prompt_template: str,
    model: str,
    topic: str,
    best_entry: Dict[str, Any],
    mature_idea: str,
    refinement_scope: str,
    paper_entries: List[Dict[str, Any]],
    logger,
) -> Dict[str, Any]:
    prompt = prompt_template.format(
        topic=topic,
        mature_idea=mature_idea or "",
        refinement_scope=refinement_scope or "",
        idea=pretty_json(best_entry),
        papers=pretty_json(paper_entries or []),
    )
    try:
        response = chat_fn(prompt, temperature=0.2, max_output_tokens=8192, model=model)
        payload = parse_json_response(response)
        if not isinstance(payload, dict):
            return dict(best_entry)
        aligned = dict(best_entry)
        for key in ("title", "abstract", "core_contribution", "method", "risks"):
            value = str(payload.get(key) or "").strip()
            if value:
                aligned[key] = value
        return aligned
    except Exception as exc:  # pragma: no cover - network
        logger.warning("⚠️ Public idea alignment failed: %s", exc)
        return dict(best_entry)
class LigRuntime:
    """Thin wrapper around LigAgent chat/tool calls with op-level tracing."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent

    def llm_text(
        self,
        *,
        session: Optional[Any],
        stage: str,
        workflow_name: Optional[str] = None,
        op_name: str,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        started_at = perf_counter()
        if hasattr(self.agent, "resolve_stage_model"):
            resolved_model = self.agent.resolve_stage_model(
                stage=stage,
                workflow_name=workflow_name,
                requested_model=model,
            )
        else:
            resolved_model = model or getattr(self.agent, "model", "gpt-5-mini")
        try:
            result = self.agent.chat(prompt, model=resolved_model, stage=stage, **kwargs)
            self._record(
                session,
                "llm_call",
                stage=stage,
                workflow_name=workflow_name,
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
                workflow_name=workflow_name,
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
        workflow_name: Optional[str] = None,
        op_name: str,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        raw = self.llm_text(
            session=session,
            stage=stage,
            workflow_name=workflow_name,
            op_name=op_name,
            prompt=prompt,
            model=model,
            **kwargs,
        )
        return parse_json_response(raw)

    def _record(self, session: Optional[Any], event_type: str, **payload: Any) -> None:
        if session is not None:
            session.record_event(event_type, **payload)


class LigSession:
    """Lightweight per-run state wrapper for LigAgent."""

    def __init__(self, artifact: Dict[str, Any]) -> None:
        self.artifact = ensure_artifact_structure(artifact)
        self._pending_slots: Dict[str, Any] = {}
        self._pending_events: List[Dict[str, Any]] = []

    def set_slot(self, name: str, value: Any) -> None:
        self._pending_slots[name] = value

    def record_event(self, event_type: str, **payload: Any) -> None:
        event = {"event": event_type}
        event.update(payload)
        self._pending_events.append(event)

    def drain_patch(self) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        slots = self._pending_slots
        events = self._pending_events
        self._pending_slots = {}
        self._pending_events = []
        return slots, events
