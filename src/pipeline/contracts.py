"""Pipeline artifact contracts and prefinish review gates.

The pipeline shell treats these contract checks as the hard boundary between phases.
An agent may finish its internal loop, but the phase is not complete until the
registered artifact contract passes here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from src.agents.experiment_agent.runtime.ablation_results import (
    REQUIRED_COMPONENT_FIELDS,
    REQUIRED_SUMMARY_FIELDS,
    validate_ablation_results_payload,
)
from src.agents.experiment_agent.runtime.idea_components import canonical_component_names
from src.agents.idea_agent.utils.workflow.idea_contract import normalize_idea_contract


SURVEY = "survey"
IDEA = "idea"
EXPERIMENT = "experiment"

REPORT_FILENAME = "contract_report.json"
EXPERIMENT_ABLATION_RESULTS_REL = Path("agent_reports") / "ablation" / "final" / "ablation_results.json"
EXPERIMENT_SYMBOLIC_MEMORY_RECEIPT_REL = (
    Path("agent_reports") / "ablation" / "final" / "symbolic_memory_receipt.json"
)


@dataclass(frozen=True)
class ContractIssue:
    """A single contract failure that can be handed back to the phase agent."""

    code: str
    message: str
    path: str = ""
    field: str = ""

    def to_dict(self) -> Dict[str, str]:
        payload = {
            "code": self.code,
            "message": self.message,
        }
        if self.path:
            payload["path"] = self.path
        if self.field:
            payload["field"] = self.field
        return payload


@dataclass(frozen=True)
class ContractReport:
    """Serializable phase contract report."""

    phase: str
    valid: bool
    workspace: str
    artifacts: Dict[str, str]
    issues: List[ContractIssue] = field(default_factory=list)
    normalized: Dict[str, Any] = field(default_factory=dict)
    checked_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z"))

    @property
    def repair_payload(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "status": "PASS" if self.valid else "FAIL",
            "issues": [issue.to_dict() for issue in self.issues],
            "artifacts": self.artifacts,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "status": "PASS" if self.valid else "FAIL",
            "valid": self.valid,
            "checked_at": self.checked_at,
            "workspace": self.workspace,
            "artifacts": self.artifacts,
            "issues": [issue.to_dict() for issue in self.issues],
            "normalized": self.normalized,
            "repair_payload": self.repair_payload,
        }


ContractChecker = Callable[[Path, Mapping[str, Any]], ContractReport]


def _read_json(path: Path) -> Tuple[Optional[Any], Optional[ContractIssue]]:
    if not path.exists():
        return None, ContractIssue(
            code="missing_file",
            message=f"Required file is missing: {path.name}",
            path=str(path),
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as exc:
        return None, ContractIssue(
            code="invalid_json",
            message=f"Invalid JSON: {exc}",
            path=str(path),
        )


def _missing_file_issues(paths: Iterable[Path]) -> List[ContractIssue]:
    issues: List[ContractIssue] = []
    for path in paths:
        if not path.exists():
            issues.append(
                ContractIssue(
                    code="missing_file",
                    message=f"Required file is missing: {path.name}",
                    path=str(path),
                )
            )
    return issues


def _same_resolved_path(left: Any, right: Path) -> bool:
    left_text = str(left or "").strip()
    if not left_text:
        return False
    return os.path.abspath(os.path.expanduser(left_text)) == os.path.abspath(str(right))


def _write_report(workspace: Path, report: ContractReport) -> ContractReport:
    workspace.mkdir(parents=True, exist_ok=True)
    report_path = workspace / REPORT_FILENAME
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    return report


def contract_report_path(workspace: str | Path) -> Path:
    return Path(workspace) / REPORT_FILENAME


def load_contract_report(workspace: str | Path) -> Optional[Dict[str, Any]]:
    path = contract_report_path(workspace)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def is_valid_contract_report(workspace: str | Path, phase: str) -> bool:
    payload = load_contract_report(workspace)
    if not payload:
        return False
    return (
        payload.get("phase") == phase
        and bool(payload.get("valid"))
        and payload.get("status") == "PASS"
    )


def validate_survey_contract(workspace: Path, context: Mapping[str, Any]) -> ContractReport:
    _ = context
    survey_md = workspace / "survey.md"
    survey_json = workspace / "survey.json"
    evaluation = workspace / "evaluation.txt"
    artifacts = {
        "survey_markdown": str(survey_md),
        "survey_json": str(survey_json),
        "evaluation": str(evaluation),
    }
    issues = _missing_file_issues((survey_md, survey_json, evaluation))

    payload, read_issue = _read_json(survey_json)
    if read_issue:
        if read_issue.code != "missing_file" or read_issue.path not in [issue.path for issue in issues]:
            issues.append(read_issue)
    elif not isinstance(payload, dict):
        issues.append(
            ContractIssue(
                code="invalid_schema",
                message="survey.json must be a JSON object.",
                path=str(survey_json),
            )
        )
    else:
        paper = payload.get("paper")
        references = payload.get("references")
        if not isinstance(paper, str) or not paper.strip():
            issues.append(
                ContractIssue(
                    code="invalid_field",
                    message="survey.json.paper must be a non-empty string.",
                    path=str(survey_json),
                    field="paper",
                )
            )
        if not isinstance(references, list):
            issues.append(
                ContractIssue(
                    code="invalid_field",
                    message="survey.json.references must be a list.",
                    path=str(survey_json),
                    field="references",
                )
            )

    normalized = {}
    if isinstance(payload, dict):
        normalized = {
            "paper_chars": len(str(payload.get("paper") or "")),
            "reference_count": len(payload.get("references") or [])
            if isinstance(payload.get("references"), list)
            else 0,
        }

    return _write_report(
        workspace,
        ContractReport(
            phase=SURVEY,
            valid=not issues,
            workspace=str(workspace),
            artifacts=artifacts,
            issues=issues,
            normalized=normalized,
        ),
    )


def validate_idea_contract(workspace: Path, context: Mapping[str, Any]) -> ContractReport:
    idea_filename = str(context.get("idea_result_filename") or "idea_result.json")
    idea_path = workspace / idea_filename
    artifacts = {"idea_result": str(idea_path)}
    candidate_path = workspace / "idea_candidate.json"
    replanned_path = workspace / "replanned_idea_result.json"
    if candidate_path.exists():
        artifacts["idea_candidate"] = str(candidate_path)
    if replanned_path.exists():
        artifacts["replanned_idea_result"] = str(replanned_path)
    issues: List[ContractIssue] = []

    payload, read_issue = _read_json(idea_path)
    if read_issue:
        issues.append(read_issue)
        normalized: Dict[str, Any] = {}
    else:
        try:
            normalized_idea = normalize_idea_contract(
                payload,
                allow_legacy=False,
                keep_extra=True,
            )
            normalized = {
                "title": normalized_idea.get("title", ""),
                "component_count": len(normalized_idea.get("components") or []),
                "tags": normalized_idea.get("tags") or [],
            }
        except (TypeError, ValueError) as exc:
            issues.append(
                ContractIssue(
                    code="invalid_schema",
                    message=str(exc),
                    path=str(idea_path),
                )
            )
            normalized = {}

    return _write_report(
        workspace,
        ContractReport(
            phase=IDEA,
            valid=not issues,
            workspace=str(workspace),
            artifacts=artifacts,
            issues=issues,
            normalized=normalized,
        ),
    )


def validate_experiment_contract(workspace: Path, context: Mapping[str, Any]) -> ContractReport:
    _ = context
    ablation_path = workspace / EXPERIMENT_ABLATION_RESULTS_REL
    receipt_path = workspace / EXPERIMENT_SYMBOLIC_MEMORY_RECEIPT_REL
    idea_path = workspace / "idea.json"
    artifacts = {
        "idea": str(idea_path),
        "ablation_results": str(ablation_path),
        "symbolic_memory_receipt": str(receipt_path),
    }
    issues = _missing_file_issues((idea_path, ablation_path, receipt_path))
    normalized: Dict[str, Any] = {}
    component_names: List[str] = []

    payload, read_issue = _read_json(ablation_path)
    if read_issue:
        if read_issue.path not in [issue.path for issue in issues]:
            issues.append(read_issue)
    elif not isinstance(payload, dict):
        issues.append(
            ContractIssue(
                code="invalid_schema",
                message=f"{EXPERIMENT_ABLATION_RESULTS_REL.as_posix()} must be a JSON object.",
                path=str(ablation_path),
            )
        )
    else:
        try:
            component_names = canonical_component_names(str(workspace), idea_json_path=str(idea_path))
        except Exception as exc:
            component_names = []
            issues.append(
                ContractIssue(
                    code="invalid_schema",
                    message=f"Failed to load canonical idea components: {exc}",
                    path=str(idea_path),
                )
            )
        if component_names:
            valid, error = validate_ablation_results_payload(
                payload,
                canonical_component_names=component_names,
            )
            if not valid:
                issues.append(
                    ContractIssue(
                        code="invalid_schema",
                        message=error or f"Invalid {EXPERIMENT_ABLATION_RESULTS_REL.as_posix()} payload.",
                        path=str(ablation_path),
                    )
                )
            normalized = {
                "canonical_components": component_names,
                "required_component_fields": list(REQUIRED_COMPONENT_FIELDS),
                "required_summary_fields": list(REQUIRED_SUMMARY_FIELDS),
            }

    receipt_payload, receipt_read_issue = _read_json(receipt_path)
    if receipt_read_issue:
        if receipt_read_issue.path not in [issue.path for issue in issues]:
            issues.append(receipt_read_issue)
    elif not isinstance(receipt_payload, dict):
        issues.append(
            ContractIssue(
                code="invalid_schema",
                message=f"{EXPERIMENT_SYMBOLIC_MEMORY_RECEIPT_REL.as_posix()} must be a JSON object.",
                path=str(receipt_path),
            )
        )
    else:
        receipt_status = str(receipt_payload.get("status") or "").strip().upper()
        if receipt_status != "PASS":
            blocker = str(receipt_payload.get("blocker") or "").strip()
            issues.append(
                ContractIssue(
                    code="symbolic_memory_writeback_failed",
                    message=(
                        "Experiment finalization must write ablation results back to symbolic memory "
                        f"before the phase can complete. Receipt status is `{receipt_status or 'UNKNOWN'}`"
                        + (f": {blocker}" if blocker else ".")
                    ),
                    path=str(receipt_path),
                    field="status",
                )
            )
        if str(receipt_payload.get("hook") or "").strip() != "final_science_prefinish":
            issues.append(
                ContractIssue(
                    code="invalid_field",
                    message="symbolic memory receipt must come from `final_science_prefinish`.",
                    path=str(receipt_path),
                    field="hook",
                )
            )
        if not _same_resolved_path(receipt_payload.get("ablation_results_path"), ablation_path):
            issues.append(
                ContractIssue(
                    code="path_mismatch",
                    message=(
                        "symbolic memory receipt `ablation_results_path` must match "
                        f"{EXPERIMENT_ABLATION_RESULTS_REL.as_posix()}."
                    ),
                    path=str(receipt_path),
                    field="ablation_results_path",
                )
            )
        symbolic_memory_path = str(receipt_payload.get("symbolic_memory_path") or "").strip()
        symbolic_memory_file = str(receipt_payload.get("symbolic_memory_file_path") or "").strip()
        if not symbolic_memory_path:
            issues.append(
                ContractIssue(
                    code="missing_field",
                    message="symbolic memory receipt must include non-empty `symbolic_memory_path`.",
                    path=str(receipt_path),
                    field="symbolic_memory_path",
                )
            )
        expected_memory_file = (
            os.path.join(symbolic_memory_path, "symbolic_memory.json")
            if symbolic_memory_path
            else ""
        )
        if not symbolic_memory_file:
            symbolic_memory_file = expected_memory_file
        if not symbolic_memory_file or not os.path.exists(symbolic_memory_file):
            issues.append(
                ContractIssue(
                    code="missing_file",
                    message=f"symbolic memory file is missing: {symbolic_memory_file or expected_memory_file}",
                    path=symbolic_memory_file or expected_memory_file,
                    field="symbolic_memory_file_path",
                )
            )
        records_created = receipt_payload.get("records_created")
        try:
            records_created_int = int(records_created)
        except Exception:
            records_created_int = -1
        if component_names and records_created_int < len(component_names):
            issues.append(
                ContractIssue(
                    code="invalid_field",
                    message=(
                        "`records_created` must cover every canonical component "
                        f"({records_created_int} < {len(component_names)})."
                    ),
                    path=str(receipt_path),
                    field="records_created",
                )
            )
        record_ids = receipt_payload.get("record_ids")
        if not isinstance(record_ids, list) or len(record_ids) != max(records_created_int, 0):
            issues.append(
                ContractIssue(
                    code="invalid_field",
                    message="symbolic memory receipt `record_ids` must have one entry per created record.",
                    path=str(receipt_path),
                    field="record_ids",
                )
            )
        normalized.update(
            {
                "symbolic_memory_path": symbolic_memory_path,
                "symbolic_memory_file_path": symbolic_memory_file,
                "records_created": records_created_int,
                "receipt_status": receipt_status,
            }
        )

    return _write_report(
        workspace,
        ContractReport(
            phase=EXPERIMENT,
            valid=not issues,
            workspace=str(workspace),
            artifacts=artifacts,
            issues=issues,
            normalized=normalized,
        ),
    )


CONTRACT_CHECKERS: Dict[str, ContractChecker] = {
    SURVEY: validate_survey_contract,
    IDEA: validate_idea_contract,
    EXPERIMENT: validate_experiment_contract,
}


def validate_phase_contract(
    phase: str,
    workspace: str | Path,
    context: Optional[Mapping[str, Any]] = None,
) -> ContractReport:
    if phase not in CONTRACT_CHECKERS:
        raise ValueError(f"Unknown phase contract: {phase}")
    return CONTRACT_CHECKERS[phase](Path(workspace), context or {})


def run_prefinish_hook(
    phase: str,
    workspace: str | Path,
    context: Optional[Mapping[str, Any]] = None,
) -> ContractReport:
    """Run the hard prefinish gate for a pipeline phase."""

    report = validate_phase_contract(phase, workspace, context)
    if not report.valid:
        raise ContractValidationError(report)
    return report


class ContractValidationError(RuntimeError):
    """Raised when a phase cannot finish because its contract failed."""

    def __init__(self, report: ContractReport):
        super().__init__(self._format_message(report))
        self.report = report
        self.repair_payload = report.repair_payload

    @staticmethod
    def _format_message(report: ContractReport) -> str:
        issues = "; ".join(issue.message for issue in report.issues)
        return f"{report.phase} prefinish contract failed: {issues}"


__all__ = [
    "EXPERIMENT",
    "IDEA",
    "REPORT_FILENAME",
    "SURVEY",
    "ContractIssue",
    "ContractReport",
    "ContractValidationError",
    "contract_report_path",
    "is_valid_contract_report",
    "load_contract_report",
    "run_prefinish_hook",
    "validate_phase_contract",
    "validate_experiment_contract",
    "validate_idea_contract",
    "validate_survey_contract",
]
