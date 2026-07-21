"""Helpers for optional ablation-results ingestion into LigAgent runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional


def _resolve_ablation_results_file(raw_path: Any) -> Optional[Path]:
    path_text = str(raw_path).strip() if raw_path is not None else ""
    if not path_text:
        return None

    path = Path(path_text).expanduser()
    if not path.exists():
        return None

    if path.is_file():
        return path if path.suffix.lower() == ".json" else None

    json_files = sorted(
        candidate for candidate in path.iterdir() if candidate.is_file() and candidate.suffix.lower() == ".json"
    )
    if not json_files:
        return None
    if len(json_files) > 1:
        raise ValueError(
            f"Expected exactly one JSON file under ablation_results_path={path}, found {len(json_files)}."
        )
    return json_files[0]


def load_ablation_results_payload(run_inputs: Mapping[str, Any]) -> Any:
    """Resolve ablation results from an optional JSON directory or JSON file path."""
    resolved_file = _resolve_ablation_results_file(run_inputs.get("ablation_results_path"))
    if resolved_file is None:
        return None

    with resolved_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if payload else None


def ingest_ablation_results_if_available(
    agent: Any,
    run_inputs: Mapping[str, Any],
    logger: Optional[Any] = None,
) -> bool:
    """Inject ablation results when a non-empty payload is available."""
    payload = load_ablation_results_payload(run_inputs)
    if not payload:
        if logger is not None:
            logger.info("[LigAgent] No ablation results provided; skipping ingestion.")
        return False

    agent.ingest_ablation_results(payload)
    if logger is not None:
        logger.info("[LigAgent] Injected ablation results into artifact state.")
    return True
