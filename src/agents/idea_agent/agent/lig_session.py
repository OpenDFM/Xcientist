from typing import Any, Dict


class LigSession:
    """Lightweight per-run state wrapper for LigAgent."""

    def __init__(self, artifact: Dict[str, Any]) -> None:
        self.artifact = artifact
        self.artifact.setdefault("context_slots", {})
        self.artifact.setdefault("operation_trace", [])

    def set_slot(self, name: str, value: Any) -> None:
        self.artifact["context_slots"][name] = value

    def get_slot(self, name: str, default: Any = None) -> Any:
        return self.artifact["context_slots"].get(name, default)

    def record_event(self, event_type: str, **payload: Any) -> None:
        event = {"event": event_type}
        event.update(payload)
        self.artifact["operation_trace"].append(event)
