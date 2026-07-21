"""OpenHarness finalization worker with deterministic prefinish hook."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.agents.experiment_agent.runtime.openharness_vendor import ensure_vendored_openharness_path


ensure_vendored_openharness_path()

from openharness.hooks.types import HookResult

from src.agents.experiment_agent.config import get_agent_model
from src.agents.experiment_agent.runtime.artifacts import (
    ArtifactRegistry,
    ArtifactSpec,
    artifact_prompt_context,
    write_artifact_registry_snapshot,
)
from src.agents.experiment_agent.runtime.finalization_hooks import (
    run_final_science_prefinish_hooks,
)
from src.agents.experiment_agent.runtime.manifests import artifact_paths, load_json_file
from src.agents.experiment_agent.runtime.openharness_runner import (
    OpenHarnessAgentRunner,
    extract_json_object,
    validate_json_schema_fragment,
)
from src.agents.experiment_agent.runtime.report_layout import phase_rel
from src.agents.experiment_agent.telemetry import print_kv_table, print_phase


FINALIZATION_WORKER = "finalization_worker"

FINALIZATION_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["READY_FOR_FINALIZATION"]},
        "checked_inputs": {"type": "array", "items": {"type": "string"}},
        "repair_summary": {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["status", "checked_inputs", "repair_summary", "notes"],
}


def build_finalization_artifact_registry(
    *,
    workspace_root: str,
    project_root: str,
) -> ArtifactRegistry:
    """Build a runtime-only registry for finalization outputs."""
    _ = project_root
    registry = ArtifactRegistry(workspace_root=os.path.realpath(workspace_root))
    runtime_specs = [
        ArtifactSpec(
            artifact_id="final.ablation_results",
            stage="finalization",
            path=phase_rel("ablation", "final", "ablation_results.json"),
            kind="json",
            required=True,
            schema_name="ablation_results",
            writer="runtime",
            description="Runtime-owned final ablation results.",
        ),
        ArtifactSpec(
            artifact_id="runtime.symbolic_memory_receipt",
            stage="finalization",
            path=phase_rel("ablation", "final", "symbolic_memory_receipt.json"),
            kind="json",
            required=True,
            writer="runtime",
            description="Runtime-owned symbolic-memory writeback receipt.",
        ),
        ArtifactSpec(
            artifact_id="runtime.ablation_materialization_report",
            stage="finalization",
            path=phase_rel("ablation", "final", "materialization_report.json"),
            kind="json",
            required=True,
            writer="runtime",
            description="Runtime-owned final materialization report.",
        ),
        ArtifactSpec(
            artifact_id="runtime.ablation_results_manifest",
            stage="finalization",
            path=phase_rel("ablation", "final", "ablation_results_manifest.json"),
            kind="json",
            required=True,
            writer="runtime",
            description="Runtime-owned ablation-results contract manifest.",
        ),
    ]
    for spec in runtime_specs:
        registry.add(spec)
    return registry


def _schema_text() -> str:
    return json.dumps(FINALIZATION_RESPONSE_SCHEMA, ensure_ascii=False, indent=2)


def _response_template() -> str:
    return json.dumps(
        {
            "status": "READY_FOR_FINALIZATION",
            "checked_inputs": [
                "agent_reports/science/phase.json",
                "idea.json",
                "agent_reports/ablation/final/symbolic_memory_receipt.json if repairing a prior failure",
            ],
            "repair_summary": "what was repaired after prior hook feedback, or 'none'",
            "notes": "brief finalization readiness note",
        },
        ensure_ascii=False,
        indent=2,
    )


def _finalization_feedback(receipt: Dict[str, Any]) -> str:
    receipt_path = str(receipt.get("receipt_path") or receipt.get("symbolic_memory_receipt_path") or "")
    blocker = str(receipt.get("blocker") or "finalization failed").strip()
    repair = str(receipt.get("repair_instructions") or "").strip()
    materialization = receipt.get("materialization")
    lines = [
        "Xcientist finalization prefinish hook blocked completion.",
        "",
        "The finalization worker must repair this in the same OpenHarness session, then finish again.",
        "Do not directly write runtime-owned final artifacts under `agent_reports/ablation/final/`; the hook owns them.",
        "",
        f"Receipt: `{receipt_path}`" if receipt_path else "Receipt: `(not available)`",
        f"Blocker: {blocker}",
    ]
    if repair:
        lines.extend(["", "Repair instructions:", repair])
    if isinstance(materialization, dict) and materialization.get("blocker"):
        lines.extend(["", "Materialization blocker:", str(materialization.get("blocker"))])
    lines.extend(
        [
            "",
            "When ready, stop with this exact JSON shape:",
            "```json",
            _response_template(),
            "```",
        ]
    )
    return "\n".join(lines)


def _build_finalization_prefinish_gate(
    *,
    workspace_root: str,
    project_root: str,
    experiment_id: str,
    config: Optional[Any],
) -> Any:
    paths = artifact_paths(workspace_root, project_root)

    async def _gate(stop_payload: Dict[str, Any]) -> HookResult:
        text = str(stop_payload.get("assistant_text") or "").strip()
        try:
            payload = extract_json_object(text)
        except Exception as exc:
            return HookResult(
                hook_type="xcientist_finalization_prefinish_gate",
                success=False,
                blocked=True,
                reason=(
                    "Xcientist finalization prefinish hook blocked completion because the final "
                    "response is not exactly one JSON object.\n\n"
                    f"Error: {exc}\n\n"
                    "Return JSON shaped like this:\n"
                    "```json\n"
                    f"{_response_template()}\n"
                    "```\n\nExpected schema:\n"
                    "```json\n"
                    f"{_schema_text()}\n"
                    "```"
                ),
            )
        issues = validate_json_schema_fragment(payload, FINALIZATION_RESPONSE_SCHEMA)
        if issues:
            return HookResult(
                hook_type="xcientist_finalization_prefinish_gate",
                success=False,
                blocked=True,
                reason=(
                    "Xcientist finalization prefinish hook blocked completion because the final "
                    "JSON does not satisfy the required schema.\n\n"
                    "Schema issues:\n"
                    + "\n".join(f"- {issue}" for issue in issues)
                    + "\n\nReturn JSON shaped like this:\n"
                    "```json\n"
                    f"{_response_template()}\n"
                    "```\n\nExpected schema:\n"
                    "```json\n"
                    f"{_schema_text()}\n"
                    "```"
                ),
                metadata={"schema_issues": issues},
            )

        receipt = run_final_science_prefinish_hooks(
            workspace_root=workspace_root,
            project_root=project_root,
            experiment_id=experiment_id,
            config=config,
        )
        receipt.setdefault("receipt_path", paths["symbolic_memory_receipt"])
        if receipt.get("status") == "PASS":
            return HookResult(
                hook_type="xcientist_finalization_prefinish_gate",
                success=True,
                blocked=False,
                metadata={"receipt": receipt},
            )
        return HookResult(
            hook_type="xcientist_finalization_prefinish_gate",
            success=False,
            blocked=True,
            reason=_finalization_feedback(receipt),
            metadata={"receipt": receipt},
        )

    return _gate


def _finalization_prompt(
    *,
    workspace_root: str,
    project_root: str,
    experiment_id: str,
    registry: ArtifactRegistry,
) -> str:
    paths = artifact_paths(workspace_root, project_root)
    return f"""# Finalization Worker

Experiment: `{experiment_id}`
Workspace: `{workspace_root}`
Project: `{project_root}`

You are the final experiment materialization worker. Your completion is gated by
the finalization prefinish hook. The hook is the only component allowed to write
`agent_reports/ablation/final/ablation_results.json` and the symbolic-memory
receipt.

Read only the inputs needed to decide whether finalization is ready:
- `idea.json`
- `agent_reports/science/phase.json`
- prior finalization receipt if present: `{paths["symbolic_memory_receipt"]}`
- relevant source/config files only if the hook previously reported a repair.

If the hook returns feedback, repair the cited configuration, converter, or
runtime evidence contract in this same session. Do not fabricate missing science
results. Missing component evidence means upstream science is incomplete; expose
that blocker instead of editing final ablation outputs by hand.

{artifact_prompt_context(registry)}

When ready, stop using tools and return exactly one JSON object:
```json
{_response_template()}
```
"""


async def run_finalization_agent(
    *,
    experiment_id: str,
    workspace_root: str,
    project_root: str,
    config: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run finalization as an OpenHarness STOP/prefinish hook."""
    workspace_root = os.path.realpath(workspace_root)
    project_root = os.path.realpath(project_root)
    paths = artifact_paths(workspace_root, project_root)
    registry = build_finalization_artifact_registry(
        workspace_root=workspace_root,
        project_root=project_root,
    )
    write_artifact_registry_snapshot(registry)
    artifact_context = registry.to_context(
        stage="finalization",
        step_id="final_science_prefinish",
        attempt=1,
    )
    gate = _build_finalization_prefinish_gate(
        workspace_root=workspace_root,
        project_root=project_root,
        experiment_id=experiment_id,
        config=config,
    )
    runner = OpenHarnessAgentRunner(
        model=get_agent_model(FINALIZATION_WORKER, "finalization"),
        workspace_root=workspace_root,
        verbose=verbose,
        artifact_context=artifact_context,
        extra_tool_metadata={"xcientist_prefinish_gate": gate},
        enable_mcp=False,
    )
    print_phase("FINALIZATION", "OpenHarness prefinish hook", width=76)
    await runner.run_text(
        system_prompt=(
            "You are the Xcientist finalization worker. The final artifacts are "
            "runtime-owned and may only be materialized by the STOP/prefinish hook."
        ),
        user_prompt=_finalization_prompt(
            workspace_root=workspace_root,
            project_root=project_root,
            experiment_id=experiment_id,
            registry=registry,
        ),
        agent_name=FINALIZATION_WORKER,
        cwd=workspace_root,
    )
    receipt = load_json_file(paths["symbolic_memory_receipt"])
    if not isinstance(receipt, dict) or receipt.get("status") != "PASS":
        receipt = receipt if isinstance(receipt, dict) else {}
        receipt.setdefault("status", "FAIL")
        receipt.setdefault("hook", "final_science_prefinish")
        receipt.setdefault("ablation_results_path", "")
        receipt.setdefault("symbolic_memory_path", "")
        receipt.setdefault("blocker", "finalization worker completed without a PASS symbolic-memory receipt")
        receipt.setdefault("receipt_path", paths["symbolic_memory_receipt"])
    else:
        receipt.setdefault("receipt_path", paths["symbolic_memory_receipt"])
    print_kv_table(
        "Finalization Receipt",
        {
            "status": receipt.get("status"),
            "ablation_results": receipt.get("ablation_results_path", ""),
            "symbolic_memory_receipt": paths["symbolic_memory_receipt"],
        },
        width=88,
        mask_sensitive=False,
    )
    return receipt


__all__ = [
    "FINALIZATION_WORKER",
    "FINALIZATION_RESPONSE_SCHEMA",
    "build_finalization_artifact_registry",
    "run_finalization_agent",
]
