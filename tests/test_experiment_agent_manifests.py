import asyncio
import hashlib
import json
import os
import shlex
import sys

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
harness_src_dir = os.path.join(src_dir, "harness", "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, harness_src_dir)

from src.agents.experiment_agent.agents.code import (
    CODE_REVIEWER,
    CODE_REVIEWER_IDS,
    CODE_WORKER,
    EXPERIMENT_CODE_PLANNER,
    CodeAgent,
    run_code_agent,
)
from src.agents.experiment_agent.agents.prepare import PrepareAgent, run_prepare
from src.agents.experiment_agent.agents.science import (
    EXPERIMENT_SCIENCE_PLANNER,
    SCIENCE_REVIEWER,
    SCIENCE_REVIEWER_IDS,
    SCIENCE_WORKER,
    ScienceAgent,
    run_science_agent,
)
from src.agents.experiment_agent.runtime.ablation_results import (
    write_ablation_results_artifacts,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    ensure_canonical_workspace_artifacts,
    extract_plan_steps,
    load_json_file,
    load_workspace_state,
    resolve_prepare_idea_path,
)
from src.agents.experiment_agent.runtime.artifacts import (
    ArtifactLedger,
    ArtifactRegistry,
    ArtifactSpec,
    XcientistHookExecutor,
    artifact_schema_repair_guide,
    artifact_tools,
    build_step_artifact_registry,
    validate_artifact_contract,
)
from src.agents.experiment_agent.runtime.code_review_context import (
    audit_code_scientific_invariants,
    build_code_review_context,
)
from src.agents.experiment_agent.runtime.contracts import (
    validate_repo_contract_fields,
    validate_science_condition_plan,
)
from src.agents.experiment_agent.runtime.phase_runner import (
    execute_step_with_prefinish_review,
    materialize_executable_plan,
    planner_artifact_prefinish_gate,
    planner_output_schema,
)
from src.agents.experiment_agent.runtime.openharness_runner import validate_json_schema_fragment
from src.agents.experiment_agent.runtime.project_integrity import audit_code_project_integrity
from openharness.hooks.events import HookEvent
from openharness.tools.base import ToolExecutionContext


def _write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_json_fence(text: str):
    start = text.index("```json") + len("```json")
    end = text.index("```", start)
    return json.loads(text[start:end].strip())


def _pass_review(scope: str, **extra):
    payload = {
        "status": "PASS",
        "checked_artifacts": [extra.pop("checked_artifact", "agent_reports/evidence.json")],
        "findings": [],
        "evidence_summary": "ok",
        "phase_completion_status": "complete",
        "ready_for_next_phase": True,
        "blocking_issues": [],
        "required_followup": [],
        "artifact_role": "phase_result",
        "run_level": "full",
        "self_contained_project": True,
        "self_contained_violations": [],
        "artifact_ledger_present": True,
        "artifact_ledger_path": "agent_reports/_runtime/artifact_ledger.jsonl",
        "scope": scope,
        "terminal_blocker": False,
        "next_worker_input": "",
        "review_scope": ["contract", "code", "experiments", "logs", "artifacts", "evidence", "safety"],
    }
    payload.update(extra)
    return payload


def _worker_report(*artifact_ids: str):
    return {
        "summary": "done",
        "artifact_ids_touched": list(artifact_ids),
        "remaining_blockers": [],
    }


def _matrix_review(reviewer_id: str, *, checked_artifact: str = "agent_reports/_runtime/artifact_ledger.jsonl", **structured):
    return {
        "reviewer_id": reviewer_id,
        "reviewer_kind": "agent",
        "status": "PASS",
        "blocking": True,
        "summary": "ok",
        "checked_artifacts": [checked_artifact],
        "issues": [],
        "structured_findings": structured,
    }


def _code_handoff_payload(project_files=None, *, log_path="results/smoke/log.txt", metrics_path="results/smoke/metrics.json"):
    return {
        "project_files": project_files or ["project/runner.py"],
        "verification": "bounded real-data verification passed",
        "verify_command": "pytest -q",
        "returncode": 0,
        "logs": [log_path],
        "metrics_files": [metrics_path],
    }


def _code_smoke_payload(*, log_path="results/smoke/log.txt", metrics_path="results/smoke/metrics.json"):
    return {
        "command": "pytest -q",
        "returncode": 0,
        "raw_outputs": [metrics_path],
        "logs": [log_path],
        "metrics_files": [metrics_path],
        "dataset_bindings": {"train": "dataset_candidate/train.json", "evaluation": ["dataset_candidate/eval.json"]},
        "component_toggles": [
            {"condition_id": "full_component_a", "enabled_components": ["component_a"], "disabled_components": []},
            {"condition_id": "without_component_a", "enabled_components": [], "disabled_components": ["component_a"], "flag": "--disable-component-a"},
        ],
        "bounded_runtime": {"max_batches": 1, "max_epochs": 1},
    }


def _science_evidence_payload(step):
    condition_id = step["condition_id"]
    metrics_path = f"results/science/{condition_id}/metrics.json"
    log_path = f"results/science/{condition_id}/run.log"
    return {
        "condition_id": condition_id,
        "enabled_components": step.get("enabled_components") or [],
        "disabled_components": step.get("disabled_components") or [],
        "reference_condition_id": step.get("reference_condition_id"),
        "run_level": "full",
        "command": step["command"],
        "returncode": 0,
        "output_dir": step["output_dir"],
        "raw_outputs": [metrics_path],
        "logs": [log_path],
        "metrics_files": [metrics_path],
        "dataset_bindings": {
            "train": step.get("train_dataset_binding"),
            "evaluation": step.get("evaluation_dataset_bindings") or [],
        },
        "model_bindings": {"backend": "none_required_for_test"},
        "duration_sec": 1.0,
    }


def _science_evidence_command(step):
    condition_id = step["condition_id"]
    payload = shlex.quote(json.dumps(_science_evidence_payload(step)))
    return (
        f"mkdir -p results/science/{condition_id} && "
        f"printf '{{\"acc\": 0.9}}' > results/science/{condition_id}/metrics.json && "
        f"printf 'formal run log\\n' > results/science/{condition_id}/run.log && "
        f"printf %s {payload} > \"$XCIENTIST_ARTIFACT_PATH\""
    )


async def _run_artifact_command(artifact_context, artifact_id: str, command: str):
    context = ToolExecutionContext(
        cwd=artifact_context["workspace_root"],
        metadata={"xcientist_artifact_context": artifact_context},
    )
    tool = {item.name: item for item in artifact_tools()}["run_artifact_command"]
    result = await tool.execute(
        tool.input_model(artifact_id=artifact_id, command=command),
        context,
    )
    assert result.is_error is False, result.output


async def _write_artifact_json(artifact_context, artifact_id: str, payload):
    context = ToolExecutionContext(
        cwd=artifact_context["workspace_root"],
        metadata={"xcientist_artifact_context": artifact_context},
    )
    tool = {item.name: item for item in artifact_tools()}["write_artifact"]
    result = await tool.execute(
        tool.input_model(artifact_id=artifact_id, json_content=payload),
        context,
    )
    assert result.is_error is False, result.output


async def _record_sources(artifact_context, artifact_id: str, sources, reason: str):
    context = ToolExecutionContext(
        cwd=artifact_context["workspace_root"],
        metadata={"xcientist_artifact_context": artifact_context},
    )
    tool = {item.name: item for item in artifact_tools()}["record_sources"]
    result = await tool.execute(
        tool.input_model(artifact_id=artifact_id, sources=list(sources), reason=reason),
        context,
    )
    assert result.is_error is False, result.output


def _idea_payload():
    return {
        "title": "demo",
        "components": [{"component": "component_a", "explanation": "demo explanation"}],
    }


def _write_idea_json(path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_idea_payload(), f, ensure_ascii=False, indent=2)


def _science_protocol(reference_condition_id=""):
    return {
        "run_level": "full",
        "setup_rationale": "Use local runner and prepared dataset evidence to choose a full bounded test setup.",
        "source_basis": [{"path": "idea.json", "basis": "local idea and prepared workspace inspection"}],
        "runtime_probe_summary": "Probe confirmed command syntax and output locations; formal run uses full protocol.",
        "training_protocol": {
            "epochs": 3,
            "max_batches": 8,
            "batch_size": 4,
            "device": "cpu",
            "seed": 7,
            "expected_runtime_sec": 60,
            "full_setup_basis": "small but formal local regression setup",
        },
        "evaluation_protocol": {
            "horizons": [1],
            "mask_rates": [0.1],
            "mask_patterns": ["random"],
            "metrics": ["acc"],
            "reference_condition_id": reference_condition_id or "",
            "perturbation_boundary": (
                "Perturbations are applied at the raw input boundary before shared preprocessing."
            ),
            "preprocessing_boundary": (
                "All conditions use the same preprocessing path before inference and metric computation."
            ),
            "ablation_isolation_assumptions": [
                "Only the declared component toggle differs from the reference condition.",
                "Dataset, seed, preprocessing, horizon, mask rate, and metric settings stay fixed.",
            ],
        },
    }


def _code_step(tmp_path, project_root, paths, *, rounds=1):
    return {
        "step_id": "final_integration_smoke",
        "goal": "smoke",
        "component_scope": ["component_a"],
        "code_artifacts": [
            {
                "path": "project/runner.py",
                "artifact_type": "entrypoint",
                "symbols": ["run"],
                "responsibility": "run experiment",
                "dependencies": [],
                "config_keys": [],
                "entrypoint_role": "all_components_runner",
            }
        ],
        "interface_contract": {"entrypoint": "python runner.py"},
        "implementation_requirements": {
            "data": "dataset_candidate",
            "evidence": "agent_reports/code/artifacts/final_integration_smoke.json",
            "metrics": "project/metrics.json",
            "no_mocks": True,
        },
        "component_disable_hooks": [{"component": "component_a", "flag": "--disable-component-a"}],
        "experiment_bindings": {
            "dataset": "dataset_candidate",
            "metrics_json": "project/metrics.json",
            "evidence_artifact": "agent_reports/code/artifacts/final_integration_smoke.json",
        },
        "repo_source_paths": [],
        "repo_copy_intent": "none",
        "project_target_paths": ["project/runner.py"],
        "input_paths": {"data_dir": str(tmp_path / "dataset_candidate")},
        "repos_policy": "reference_or_copy",
        "project_must_be_self_contained": True,
        "write_scope": "project",
        "verify_command": "pytest -q",
        "done_condition": (
            "Smoke passes using real dataset_candidate files and writes metrics/evaluation evidence "
            "to agent_reports/code/artifacts/final_integration_smoke.json. sys.path injection, editable installs "
            "of repos/, and imports reaching outside project/ are forbidden. No mocks, synthetic "
            "data, dry-run-only, or imports-only evidence."
        ),
        "artifact_ids": [
            "code.final_integration_smoke.handoff",
            "code.final_integration_smoke.evidence",
        ],
    }


def _reference_step(tmp_path, project_root, paths):
    return {
        "condition_id": "full_component_a",
        "goal": "run all-components reference",
        "enabled_components": ["component_a"],
        "disabled_components": [],
        "reference_condition_id": None,
        "train_dataset_binding": {"path": str(tmp_path / "dataset_candidate")},
        "evaluation_dataset_bindings": [{"path": str(tmp_path / "dataset_candidate")}],
        "metric_bindings": [{"name": "acc"}],
        "component_set_description": "all components enabled",
        "result_interpretation_rule": "reference for component-disabled conditions",
        "repo_source_paths": [],
        "repo_copy_intent": "none",
        "project_target_paths": [],
        "input_paths": {"data_dir": str(tmp_path / "dataset_candidate")},
        "repos_policy": "reference_or_copy",
        "project_must_be_self_contained": True,
        "command": "python project/run.py --save results/science/full_component_a --metrics-json results/science/full_component_a/metrics.json",
        "output_dir": "results/science/full_component_a",
        "raw_evidence": ["results/science/full_component_a/metrics.json"],
        "pass_condition": "metric captured",
        "artifact_ids": ["science.full_component_a.evidence"],
        **_science_protocol(""),
    }


def _component_disabled_step(tmp_path, project_root, paths):
    return {
        "condition_id": "without_component_a",
        "goal": "run with component_a disabled",
        "enabled_components": [],
        "disabled_components": ["component_a"],
        "reference_condition_id": "full_component_a",
        "train_dataset_binding": {"path": str(tmp_path / "dataset_candidate")},
        "evaluation_dataset_bindings": [{"path": str(tmp_path / "dataset_candidate")}],
        "metric_bindings": [{"name": "acc"}],
        "component_set_description": "component_a disabled",
        "result_interpretation_rule": "positive if removal worsens accuracy",
        "repo_source_paths": [],
        "repo_copy_intent": "none",
        "project_target_paths": [],
        "input_paths": {"data_dir": str(tmp_path / "dataset_candidate")},
        "repos_policy": "reference_or_copy",
        "project_must_be_self_contained": True,
        "command": "python project/run.py --disable-component-a --save results/science/without_component_a --metrics-json results/science/without_component_a/metrics.json",
        "output_dir": "results/science/without_component_a",
        "raw_evidence": ["results/science/without_component_a/metrics.json"],
        "pass_condition": "metric captured",
        "artifact_ids": ["science.without_component_a.evidence"],
        **_science_protocol("full_component_a"),
    }


def test_runtime_artifact_paths_use_reviewer_and_materialization_names(tmp_path):
    paths = artifact_paths(str(tmp_path))
    assert paths["prepare_reviewer"].endswith(os.path.join("agent_reports", "prepare", "phase.json"))
    assert paths["code_reviewer"].endswith(os.path.join("agent_reports", "code", "phase.json"))
    assert paths["science_reviewer"].endswith(os.path.join("agent_reports", "science", "phase.json"))
    assert paths["ablation_materialization_report"].endswith(os.path.join("agent_reports", "ablation", "final", "materialization_report.json"))
    assert paths["artifact_registry"].endswith(os.path.join("agent_reports", "_runtime", "artifact_registry.json"))
    assert paths["artifact_ledger"].endswith(os.path.join("agent_reports", "_runtime", "artifact_ledger.jsonl"))


def test_extract_plan_steps_accepts_list_and_canonical_shapes():
    assert extract_plan_steps([{"step_id": "a"}]) == [{"step_id": "a"}]
    assert extract_plan_steps({"stages": [{"step_id": "a"}]}) == [{"step_id": "a"}]
    assert extract_plan_steps({"steps": [{"step_id": "a"}]}) == [{"step_id": "a"}]


def test_prepare_idea_resolver_uses_canonical_report_location(tmp_path):
    paths = artifact_paths(str(tmp_path))
    canonical_prepare = tmp_path / "agent_reports" / "prepare" / "artifacts" / "idea.md"
    canonical_prepare.parent.mkdir(parents=True, exist_ok=True)
    canonical_prepare.write_text("# prepare", encoding="utf-8")

    assert resolve_prepare_idea_path(str(tmp_path)) == str(canonical_prepare)

    updates = ensure_canonical_workspace_artifacts(str(tmp_path))
    assert updates == {}
    assert paths["artifact_ledger"].endswith(os.path.join("agent_reports", "_runtime", "artifact_ledger.jsonl"))
    for rel in (
        ("agent_reports", "_runtime"),
        ("agent_reports", "prepare", "plan"),
        ("agent_reports", "prepare", "artifacts"),
        ("agent_reports", "code", "plan"),
        ("agent_reports", "code", "artifacts"),
        ("agent_reports", "science", "plan"),
        ("agent_reports", "science", "evidence"),
        ("agent_reports", "ablation", "final"),
    ):
        assert (tmp_path.joinpath(*rel)).is_dir()


def test_agent_entrypoints_and_role_constants_are_reviewer_based():
    assert callable(run_prepare)
    assert callable(run_code_agent)
    assert callable(run_science_agent)
    assert ScienceAgent.__name__ == "ScienceAgent"
    assert CODE_REVIEWER == "code_reviewer"
    assert SCIENCE_REVIEWER == "science_reviewer"
    assert SCIENCE_WORKER == "science_worker"
    assert EXPERIMENT_CODE_PLANNER == "experiment_code_planner"
    assert EXPERIMENT_SCIENCE_PLANNER == "experiment_science_planner"


def test_prepare_reviewer_prompt_uses_unified_review_schema():
    from src.agents.experiment_agent.agents.prepare.reviewer import prepare_reviewer_prompt

    prompt = prepare_reviewer_prompt("prepare_resource_relevance")

    assert "Return exactly this unified review report shape" in prompt
    assert '"reviewer_id": "prepare_resource_relevance"' in prompt
    assert '"issues": [' in prompt
    assert '"structured_findings"' in prompt
    assert "Output fields:" not in prompt
    assert "`evidence_summary`" not in prompt


def test_code_agent_uses_worker_internal_prefinish_review_and_contract_hook(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_dir))
    _write_json(paths["prepare_reviewer"], _pass_review("prepare"))
    _write_idea_json(tmp_path / "idea.json")

    evidence_path = tmp_path / "agent_reports" / "code" / "artifacts" / "final_integration_smoke.json"
    plan_payload = {
        "stages": [_code_step(tmp_path, project_dir, paths)],
        "summary": "code planned",
        "usage_notes": "run smoke",
    }
    responses = iter(
        [
            {"output": plan_payload},
            {"output": _worker_report("code.final_integration_smoke.handoff", "code.final_integration_smoke.evidence")},
            *[
                {"output": _matrix_review(reviewer_id, checked_artifact=str(evidence_path))}
                for reviewer_id in CODE_REVIEWER_IDS
            ],
        ]
    )
    calls = []

    async def fake_run(self, *args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("agent_name") == EXPERIMENT_CODE_PLANNER:
            planner_response = next(responses)
            await _write_artifact_json(self.artifact_context, "code.plan", planner_response["output"])
            gate = (kwargs.get("extra_tool_metadata") or {}).get("xcientist_prefinish_gate")
            assert gate is not None
            gate_result = await gate({"assistant_text": json.dumps(planner_response["output"])})
            assert gate_result.blocked is False
            return planner_response
        if kwargs.get("agent_name") == CODE_WORKER:
            handoff_payload = shlex.quote(json.dumps(_code_handoff_payload()))
            smoke_payload = shlex.quote(json.dumps(_code_smoke_payload()))
            await _run_artifact_command(
                self.artifact_context,
                "code.final_integration_smoke.handoff",
                (
                    "mkdir -p project && "
                    "mkdir -p results/smoke && "
                    "printf 'def run():\\n    return True\\n' > project/runner.py && "
                    "printf '{\"accuracy\": 0.9}' > results/smoke/metrics.json && "
                    "printf 'smoke log\\n' > results/smoke/log.txt && "
                    f"printf %s {handoff_payload} "
                    "> \"$XCIENTIST_ARTIFACT_PATH\""
                ),
            )
            await _run_artifact_command(
                self.artifact_context,
                "code.final_integration_smoke.evidence",
                (
                    "mkdir -p results/smoke && "
                    "printf '{\"accuracy\": 0.9}' > results/smoke/metrics.json && "
                    "printf 'smoke log\\n' > results/smoke/log.txt && "
                    f"printf %s {smoke_payload} "
                    "> \"$XCIENTIST_ARTIFACT_PATH\""
                ),
            )
            worker_response = next(responses)
            gate = (kwargs.get("extra_tool_metadata") or {}).get("xcientist_prefinish_gate")
            assert gate is not None
            gate_result = await gate({"assistant_text": json.dumps(worker_response["output"])})
            assert gate_result.blocked is False
            return worker_response
        return next(responses)

    monkeypatch.setattr(CodeAgent, "run", fake_run)
    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.code.planner.scan_project_self_contained",
        lambda *_: {"self_contained_project": True, "self_contained_violations": []},
    )

    result = asyncio.run(
        run_code_agent(
            experiment_id="demo",
            idea_path=str(tmp_path / "idea.md"),
            project_root=str(project_dir),
            workspace_root=str(tmp_path),
            plan="run code",
            verbose=False,
        )
    )

    assert result["status"] == "completed"
    state = load_workspace_state(str(tmp_path))
    assert state["code_reviewer"]["status"] == "PASS"
    step_review = load_json_file(str(tmp_path / "agent_reports" / "code" / "review" / "final_integration_smoke" / "latest.json"))
    assert step_review["prefinish_contract"]["status"] == "PASS"
    assert [call.get("agent_name") for call in calls] == [
        EXPERIMENT_CODE_PLANNER,
        CODE_WORKER,
        *list(CODE_REVIEWER_IDS),
    ]
    assert calls[-1]["purpose"] == "prefinish_review"


def test_prefinish_contract_failure_is_repaired_inside_worker_loop(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_dir))
    _write_json(paths["prepare_reviewer"], _pass_review("prepare"))
    _write_idea_json(tmp_path / "idea.json")

    missing_evidence = tmp_path / "agent_reports" / "code" / "artifacts" / "missing.json"
    repaired_evidence = tmp_path / "agent_reports" / "code" / "artifacts" / "final_integration_smoke.json"
    plan_payload = {
        "stages": [_code_step(tmp_path, project_dir, paths, rounds=2)],
        "summary": "code planned",
        "usage_notes": "run smoke",
    }
    responses = iter(
        [
            {"output": plan_payload},
            {"output": _worker_report()},
            {"output": _worker_report("code.final_integration_smoke.handoff", "code.final_integration_smoke.evidence")},
            *[
                {"output": _matrix_review(reviewer_id, checked_artifact=str(repaired_evidence))}
                for reviewer_id in CODE_REVIEWER_IDS
            ],
        ]
    )
    calls = []
    blocked_reasons = []

    async def fake_run(self, *args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("agent_name") == EXPERIMENT_CODE_PLANNER:
            planner_response = next(responses)
            await _write_artifact_json(self.artifact_context, "code.plan", planner_response["output"])
            gate = (kwargs.get("extra_tool_metadata") or {}).get("xcientist_prefinish_gate")
            assert gate is not None
            gate_result = await gate({"assistant_text": json.dumps(planner_response["output"])})
            assert gate_result.blocked is False
            return planner_response
        if kwargs.get("agent_name") == CODE_WORKER:
            gate = (kwargs.get("extra_tool_metadata") or {}).get("xcientist_prefinish_gate")
            assert gate is not None
            first_worker_response = next(responses)
            first_gate_result = await gate({"assistant_text": json.dumps(first_worker_response["output"])})
            assert first_gate_result.blocked is True
            blocked_reasons.append(first_gate_result.reason)
            handoff_payload = shlex.quote(json.dumps(_code_handoff_payload()))
            smoke_payload = shlex.quote(json.dumps(_code_smoke_payload()))
            await _run_artifact_command(
                self.artifact_context,
                "code.final_integration_smoke.handoff",
                (
                    "mkdir -p project && "
                    "mkdir -p results/smoke && "
                    "printf 'def run():\\n    return True\\n' > project/runner.py && "
                    "printf '{\"accuracy\": 0.9}' > results/smoke/metrics.json && "
                    "printf 'smoke log\\n' > results/smoke/log.txt && "
                    f"printf %s {handoff_payload} "
                    "> \"$XCIENTIST_ARTIFACT_PATH\""
                ),
            )
            await _run_artifact_command(
                self.artifact_context,
                "code.final_integration_smoke.evidence",
                (
                    "mkdir -p results/smoke && "
                    "printf '{\"accuracy\": 0.9}' > results/smoke/metrics.json && "
                    "printf 'smoke log\\n' > results/smoke/log.txt && "
                    f"printf %s {smoke_payload} "
                    "> \"$XCIENTIST_ARTIFACT_PATH\""
                ),
            )
            repaired_worker_response = next(responses)
            repaired_gate_result = await gate({"assistant_text": json.dumps(repaired_worker_response["output"])})
            assert repaired_gate_result.blocked is False
            return repaired_worker_response
        return next(responses)

    monkeypatch.setattr(CodeAgent, "run", fake_run)
    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.code.planner.scan_project_self_contained",
        lambda *_: {"self_contained_project": True, "self_contained_violations": []},
    )

    result = asyncio.run(
        run_code_agent(
            experiment_id="demo",
            idea_path=str(tmp_path / "idea.md"),
            project_root=str(project_dir),
            workspace_root=str(tmp_path),
            plan="run code",
            verbose=False,
        )
    )

    assert result["status"] == "completed"
    assert [call.get("agent_name") for call in calls] == [
        EXPERIMENT_CODE_PLANNER,
        CODE_WORKER,
        *list(CODE_REVIEWER_IDS),
    ]
    assert blocked_reasons
    assert "Xcientist prefinish hook blocked worker completion" in blocked_reasons[0]
    assert "Allowed ids for this step" in blocked_reasons[0]
    assert "code.final_integration_smoke.handoff" in blocked_reasons[0]
    assert "code.final_integration_smoke.evidence" in blocked_reasons[0]
    step_review = load_json_file(str(tmp_path / "agent_reports" / "code" / "review" / "final_integration_smoke" / "latest.json"))
    first_attempt = load_json_file(str(tmp_path / "agent_reports" / "code" / "review" / "final_integration_smoke" / "attempts" / "001.json"))
    second_attempt = load_json_file(str(tmp_path / "agent_reports" / "code" / "review" / "final_integration_smoke" / "attempts" / "002.json"))
    assert step_review["prefinish_contract"]["status"] == "PASS"
    assert first_attempt["prefinish_contract"]["status"] == "FAIL"
    assert second_attempt["prefinish_contract"]["status"] == "PASS"


def test_prefinish_hook_rejects_worker_ambiguous_json_and_returns_schema(tmp_path):
    step = {
        "step_id": "strict_worker_json",
        "goal": "return strict worker json",
    }
    blocked_reasons = []

    async def call_worker(_step, _previous_review, _artifact_context, prefinish_gate=None):
        assert prefinish_gate is not None
        result = await prefinish_gate(
            {
                "assistant_text": (
                    '{"summary":"done","artifact_ids_touched":[],"remaining_blockers":[]}\n'
                    '{"summary":"duplicate","artifact_ids_touched":[],"remaining_blockers":[]}'
                )
            }
        )
        blocked_reasons.append(result.reason)
        assert result.blocked is True
        return _worker_report()

    async def call_reviewer(_step, _worker_payload, _artifact_context):
        raise AssertionError("reviewer must not run when worker final JSON schema is invalid")

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=[step],
            scope="code",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "FAIL"
    assert blocked_reasons
    assert "could not parse the worker final response as JSON" in blocked_reasons[0]
    assert "exactly one unambiguous JSON object" in blocked_reasons[0]
    assert '"artifact_ids_touched"' in blocked_reasons[0]
    assert '"remaining_blockers"' in blocked_reasons[0]


def test_execute_step_resumes_previously_passed_step_without_worker(tmp_path):
    step = {
        "condition_id": "full_all_components",
        "goal": "already completed",
    }
    review_path = tmp_path / "agent_reports" / "science" / "review" / "full_all_components" / "latest.json"
    hook_path = tmp_path / "agent_reports" / "science" / "hook" / "full_all_components" / "latest.json"
    _write_json(
        str(review_path),
        {
            "status": "PASS",
            "scope": "science",
            "step_id": "full_all_components",
            "evidence_summary": "previously reviewed",
            "checked_artifacts": [],
            "blocking_issues": [],
            "review_matrix": {
                "reviewer_id": "aggregate",
                "status": "PASS",
                "structured_findings": {"step_id": "full_all_components"},
                "reports": [],
            },
        },
    )
    _write_json(
        str(hook_path),
        {
            "scope": "science",
            "step_id": "full_all_components",
            "status": "PASS",
            "review_status": "PASS",
            "returned_to_worker": False,
            "review_report_path": str(review_path),
            "prefinish_contract": {
                "status": "PASS",
                "hook": "step_prefinish_contract",
                "issues": [],
            },
        },
    )

    async def call_worker(*_args, **_kwargs):
        raise AssertionError("worker should be skipped for previously passed step")

    async def call_reviewer(*_args, **_kwargs):
        raise AssertionError("reviewer should be skipped for previously passed step")

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=[step],
            scope="science",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "PASS"
    assert len(result["step_reports"]) == 1
    assert result["step_reports"][0]["step_id"] == "full_all_components"
    assert result["step_reports"][0]["prefinish_contract"]["status"] == "PASS"


def test_prefinish_hook_rejects_non_unified_reviewer_payload_with_schema(tmp_path):
    step = {
        "stage_id": "env",
        "goal": "prepare env",
    }
    blocked_reasons = []

    async def call_worker(_step, _previous_review, artifact_context, prefinish_gate=None):
        await _run_artifact_command(
            artifact_context,
            "prepare.env",
                (
                    "mkdir -p project/env project/.venv/bin && "
                    "touch project/.venv/bin/python && "
                    "printf 'numpy\\n' > project/env/requirements.txt && "
                    "printf '{\"status\":\"READY\",\"selection_rationale\":\"local venv verified\","
                "\"venv_path\":\"project/.venv\",\"python_path\":\"project/.venv/bin/python\","
                "\"install_commands\":[\"pip install numpy\"],\"import_smoke\":\"ok\","
                "\"smoke_logs\":[\"project/env/requirements.txt\"]}' "
                "> \"$XCIENTIST_ARTIFACT_PATH\""
            ),
        )
        worker_payload = {
            "summary": "done",
            "outcome": "READY",
            "artifact_ids_touched": ["prepare.env"],
            "remaining_blockers": [],
        }
        assert prefinish_gate is not None
        gate_result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        blocked_reasons.append(gate_result.reason)
        assert gate_result.blocked is True
        return worker_payload

    async def call_reviewer(_step, _worker_payload, _artifact_context):
        return {"status": "PASS"}

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=[step],
            scope="prepare",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "FAIL"
    assert blocked_reasons
    assert "Reviewer output does not satisfy the unified schema" in blocked_reasons[0]
    assert "Return exactly the unified prefinish review JSON schema" in blocked_reasons[0]
    assert '"reviewer_id"' in blocked_reasons[0]
    assert '"issues"' in blocked_reasons[0]


def test_prefinish_hook_rejects_nonblocking_agent_reviewer_failure(tmp_path):
    step = {
        "stage_id": "env",
        "goal": "prepare env",
    }
    blocked_reasons = []

    async def call_worker(_step, _previous_review, artifact_context, prefinish_gate=None):
        await _run_artifact_command(
            artifact_context,
            "prepare.env",
            (
                "mkdir -p project/env project/.venv/bin && "
                "touch project/.venv/bin/python && "
                "printf 'numpy\\n' > project/env/requirements.txt && "
                "printf '{\"status\":\"READY\",\"selection_rationale\":\"local venv verified\","
                "\"venv_path\":\"project/.venv\",\"python_path\":\"project/.venv/bin/python\","
                "\"install_commands\":[\"pip install numpy\"],\"import_smoke\":\"ok\","
                "\"smoke_logs\":[\"project/env/requirements.txt\"]}' "
                "> \"$XCIENTIST_ARTIFACT_PATH\""
            ),
        )
        worker_payload = {
            "summary": "done",
            "outcome": "READY",
            "artifact_ids_touched": ["prepare.env"],
            "remaining_blockers": [],
        }
        assert prefinish_gate is not None
        gate_result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        blocked_reasons.append(gate_result.reason)
        assert gate_result.blocked is True
        return worker_payload

    async def call_reviewer(_step, _worker_payload, _artifact_context):
        return {
            "reviewer_id": "prepare_handoff_completeness",
            "reviewer_kind": "agent",
            "status": "FAIL",
            "blocking": False,
            "summary": "missing evidence",
            "checked_artifacts": ["agent_reports/prepare/artifacts/env.json"],
            "issues": [
                {
                    "code": "missing_evidence",
                    "message": "Env report does not cite the actual smoke log.",
                    "required_fix": "Repair the prepare.env artifact and finish again.",
                    "evidence": ["agent_reports/prepare/artifacts/env.json"],
                }
            ],
            "structured_findings": {},
        }

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=[step],
            scope="prepare",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "FAIL"
    assert blocked_reasons
    assert "`blocking` must be true" in blocked_reasons[0]
    assert "Return exactly the unified prefinish review JSON schema" in blocked_reasons[0]


def test_prefinish_contract_requires_artifact_tool_ledger_and_skips_runtime_reports(tmp_path):
    evidence = tmp_path / "project" / "env" / "requirements.txt"
    reports_dir = tmp_path / "agent_reports"
    reports_dir.mkdir()
    step = {
        "stage_id": "env",
        "goal": "prepare env",
    }

    async def call_worker(_step, _previous_review, artifact_context, prefinish_gate=None):
        await _run_artifact_command(
            artifact_context,
            "prepare.env",
            (
                "mkdir -p project/env project/.venv/bin && "
                "touch project/.venv/bin/python && "
                "printf 'numpy\\n' > project/env/requirements.txt && "
                "printf '{\"status\":\"READY\",\"selection_rationale\":\"local venv verified\","
                "\"venv_path\":\"project/.venv\",\"python_path\":\"project/.venv/bin/python\","
                "\"install_commands\":[\"pip install numpy\"],\"import_smoke\":\"ok\","
                "\"smoke_logs\":[\"project/env/requirements.txt\"]}' "
                "> \"$XCIENTIST_ARTIFACT_PATH\""
            ),
        )
        worker_payload = {
            "summary": "done",
            "outcome": "READY",
            "artifact_ids_touched": ["prepare.env"],
            "remaining_blockers": [],
        }
        assert prefinish_gate is not None
        gate_result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        assert gate_result.blocked is False
        return worker_payload

    async def call_reviewer(_step, _worker_payload, _artifact_context):
        return _matrix_review("prepare_handoff_completeness", checked_artifact="project/env/requirements.txt")

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=[step],
            scope="prepare",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "PASS"
    worker_report = load_json_file(str(reports_dir / "prepare" / "worker" / "env" / "latest.json"))
    assert worker_report["artifact_ids_touched"] == ["prepare.env"]
    review_report = load_json_file(str(reports_dir / "prepare" / "review" / "env" / "latest.json"))
    assert review_report["prefinish_contract"]["status"] == "PASS"
    assert review_report["prefinish_contract"]["contract_path"] == ""
    ledger = (reports_dir / "_runtime" / "artifact_ledger.jsonl").read_text(encoding="utf-8")
    assert '"event": "prefinish_contract"' in ledger
    assert evidence.read_text(encoding="utf-8") == "numpy\n"


def test_artifact_contract_rejects_stale_ledger_hash(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="science.demo.evidence",
            stage="science",
            path="agent_reports/science/evidence/demo.json",
            kind="json",
            required=True,
            schema_name="",
        )
    )
    evidence_path = tmp_path / "agent_reports" / "science" / "evidence" / "demo.json"
    _write_json(str(evidence_path), {"status": "first"})
    ArtifactLedger(str(tmp_path)).append(
        {
            "event": "write_artifact",
            "artifact_id": "science.demo.evidence",
            "path": str(evidence_path),
            "kind": "json",
            "sha256": _sha256_file(str(evidence_path)),
            "schema_name": "",
            "schema_issues": [],
            "stage": "science",
            "step_id": "demo",
        }
    )
    _write_json(str(evidence_path), {"status": "tampered_after_ledger"})

    contract = validate_artifact_contract(registry=registry, review_status="PASS")

    assert contract["status"] == "FAIL"
    assert any("Ledger hash mismatch" in issue for issue in contract["issues"])


def test_prepare_prefinish_accepts_credible_blocked_stage_and_stops(tmp_path):
    reports_dir = tmp_path / "agent_reports"
    reports_dir.mkdir()
    step = {
        "stage_id": "env",
        "goal": "prepare env",
    }

    blocker_payload = {
        "status": "BLOCKED",
        "blocker": {
            "reason": "No compatible Python dependency set could be resolved for the selected resources.",
            "attempted_queries": ["local requirements inspection", "pip dry-run"],
            "rejected_candidates": ["requirements set A: version conflict"],
            "missing_requirements": ["compatible dependency lockfile"],
            "user_action_required": "Provide a compatible dependency lockfile or approve changing the selected repo.",
            "evidence_paths": ["project/env/blocked_notes.txt"],
        },
    }

    async def call_worker(_step, _previous_review, artifact_context, prefinish_gate=None):
        await _run_artifact_command(
            artifact_context,
            "prepare.env",
            (
                "mkdir -p project/env && "
                "printf 'pip resolver conflict\\n' > project/env/blocked_notes.txt && "
                f"printf %s {shlex.quote(json.dumps(blocker_payload))} > \"$XCIENTIST_ARTIFACT_PATH\""
            ),
        )
        worker_payload = {
            "summary": "env blocked by dependency conflict",
            "outcome": "BLOCKED",
            "artifact_ids_touched": ["prepare.env"],
            "remaining_blockers": ["compatible dependency lockfile is missing"],
        }
        assert prefinish_gate is not None
        gate_result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        assert gate_result.blocked is False
        return worker_payload

    async def call_reviewer(_step, _worker_payload, _artifact_context):
        return _matrix_review("prepare_handoff_completeness", checked_artifact="agent_reports/prepare/artifacts/env.json")

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=[step],
            scope="prepare",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "BLOCKED"
    assert result["blocked_step"]["stage_id"] == "env"
    review_report = load_json_file(str(reports_dir / "prepare" / "review" / "env" / "latest.json"))
    assert review_report["prefinish_contract"]["status"] == "PASS"
    assert review_report["prepare_stage_contract"]["stage_status"] == "BLOCKED"


def test_code_prefinish_hook_blocks_mask_created_but_not_consumed(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "util.py").write_text(
        "\n".join(
            [
                "class MaskDataLoader:",
                "    def __init__(self, x, y, batch_size):",
                "        self.x = x",
                "",
                "def apply_point_mask(x, rate):",
                "    return x, x == x",
                "",
                "def apply_block_mask(x, rate):",
                "    return x, x == x",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (project_dir / "evaluate.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "import util",
                "",
                "def evaluate_masked(x_test, y_test):",
                "    x_test_masked, test_mask = util.apply_point_mask(x_test, 0.1)",
                "    x_test_filled = np.nan_to_num(x_test_masked, nan=0.0)",
                "    loader = util.MaskDataLoader(x_test_filled, y_test, 32)",
                "    return loader",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    step = {
        "step_id": "masked_eval",
        "goal": "implement masked evaluation",
        "artifact_ids": ["code.masked_eval.handoff"],
        "project_target_paths": ["project/evaluate.py", "project/util.py"],
        "code_artifacts": [
            {
                "path": "project/evaluate.py",
                "artifact_type": "python_module",
                "symbols": ["evaluate_masked"],
                "responsibility": "masked missingness evaluation",
                "dependencies": ["project/util.py"],
                "config_keys": ["mask_rate"],
                "entrypoint_role": "evaluation",
            }
        ],
        "verify_command": "python -m py_compile project/evaluate.py project/util.py",
    }
    reviewer_called = False

    async def call_worker(_step, _previous_review, artifact_context, prefinish_gate=None):
        handoff_payload = shlex.quote(
            json.dumps(
                _code_handoff_payload(
                    ["project/evaluate.py", "project/util.py"],
                    log_path="results/masked_eval/log.txt",
                    metrics_path="results/masked_eval/metrics.json",
                )
            )
        )
        await _run_artifact_command(
            artifact_context,
            "code.masked_eval.handoff",
            (
                "mkdir -p results/masked_eval && "
                "printf '{\"acc\": 0.9}' > results/masked_eval/metrics.json && "
                "printf 'compile log\\n' > results/masked_eval/log.txt && "
                f"printf %s {handoff_payload} > \"$XCIENTIST_ARTIFACT_PATH\""
            ),
        )
        worker_payload = _worker_report("code.masked_eval.handoff")
        gate_result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        assert gate_result.blocked is True
        assert "masked-evaluation mask" in gate_result.reason
        return worker_payload

    async def call_reviewer(_step, _worker_payload, _artifact_context):
        nonlocal reviewer_called
        reviewer_called = True
        return _matrix_review("implementation_correctness")

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=[step],
            scope="code",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "FAIL"
    assert reviewer_called is False
    invariant_report = load_json_file(
        str(tmp_path / "agent_reports" / "code" / "review" / "masked_eval" / "code_scientific_invariants" / "latest.json")
    )
    assert invariant_report["status"] == "FAIL"
    assert invariant_report["issues"][0]["code"] == "masked_evaluation_mask_not_propagated"


def test_code_invariant_blocks_stress_masks_that_bypass_missingness_preprocessing(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "gap_fill.py").write_text(
        "\n".join(
            [
                "class GapAdaptiveCachedPriorFill:",
                "    def transform(self, x):",
                "        return x, x[..., 0] != 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (project_dir / "stress_test.py").write_text(
        "\n".join(
            [
                "import util",
                "def point_mask(x, mask_rate):",
                "    return x, x[..., 0] != 0",
                "def run_stress(test_loader):",
                "    masked_xs = []",
                "    ys = []",
                "    for x, y, _m in test_loader.get_iterator():",
                "        x_masked, stress_m = point_mask(x, 0.4)",
                "        masked_xs.append(x_masked)",
                "        ys.append(y)",
                "    masked_loader = util.DataLoader(masked_xs, ys, 32)",
                "    return masked_loader",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = audit_code_scientific_invariants(
        workspace_root=str(tmp_path),
        review_context={"surface_tags": ["masking", "data", "metrics"]},
    )

    assert payload["status"] == "FAIL"
    assert payload["issues"][0]["rule"] == "stress_missingness_bypasses_preprocessing"
    assert "raw/pre-fill input boundary" in payload["issues"][0]["fix"]


def test_code_prefinish_requires_record_sources_for_copy_and_modify(tmp_path):
    project_dir = tmp_path / "project"
    repo_source = tmp_path / "repos" / "demo" / "model.py"
    project_dir.mkdir()
    repo_source.parent.mkdir(parents=True)
    repo_source.write_text("class RepoModel: pass\n", encoding="utf-8")
    (project_dir / "model.py").write_text("class RepoModel: pass\n", encoding="utf-8")
    step = {
        "step_id": "copy_repo_model",
        "goal": "copy the selected repository model into the self-contained project surface",
        "component_scope": ["component_a"],
        "code_artifacts": [
            {
                "path": "project/model.py",
                "artifact_type": "python_module",
                "symbols": ["RepoModel"],
                "responsibility": "canonical copied model implementation",
                "dependencies": [],
                "config_keys": [],
                "entrypoint_role": "model_component",
            }
        ],
        "interface_contract": {"compile": "python -m py_compile project/model.py"},
        "implementation_requirements": {"source": "repos/demo/model.py"},
        "component_disable_hooks": [{"component": "component_a", "flag": "--disable-component-a"}],
        "experiment_bindings": {"model_file": "project/model.py"},
        "repo_source_paths": ["repos/demo/model.py"],
        "repo_copy_intent": "copy_and_modify",
        "project_target_paths": ["project/model.py"],
        "input_paths": {"source": "repos/demo/model.py"},
        "repos_policy": "reference_or_copy",
        "project_must_be_self_contained": True,
        "write_scope": "project",
        "verify_command": "python -m py_compile project/model.py",
        "done_condition": (
            "Compile passes; record_sources must list repos/demo/model.py in the artifact ledger. "
            "sys.path injection, editable installs of repos/, and imports reaching outside project/ are forbidden."
        ),
        "artifact_ids": ["code.copy_repo_model.handoff"],
    }
    reviewer_called = False

    async def call_worker(_step, _previous_review, artifact_context, prefinish_gate=None):
        handoff_payload = shlex.quote(
            json.dumps(
                _code_handoff_payload(
                    ["project/model.py"],
                    log_path="results/copy_repo_model/log.txt",
                    metrics_path="results/copy_repo_model/metrics.json",
                )
            )
        )
        await _run_artifact_command(
            artifact_context,
            "code.copy_repo_model.handoff",
            (
                "mkdir -p results/copy_repo_model && "
                "printf '{\"acc\": 0.9}' > results/copy_repo_model/metrics.json && "
                "printf 'compile log\\n' > results/copy_repo_model/log.txt && "
                f"printf %s {handoff_payload} > \"$XCIENTIST_ARTIFACT_PATH\""
            ),
        )
        worker_payload = _worker_report("code.copy_repo_model.handoff")
        first_result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        assert first_result.blocked is True
        assert "record_sources" in first_result.reason
        assert reviewer_called is False
        await _record_sources(
            artifact_context,
            "code.copy_repo_model.handoff",
            ["repos/demo/model.py"],
            "copied the selected repository model into project/model.py for a self-contained implementation",
        )
        second_result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        assert second_result.blocked is False
        return worker_payload

    async def call_reviewer(_step, _worker_payload, _artifact_context):
        nonlocal reviewer_called
        reviewer_called = True
        return _matrix_review("implementation_correctness", checked_artifact="project/model.py")

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=[step],
            scope="code",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "PASS"
    assert reviewer_called is True
    provenance_report = load_json_file(
        str(tmp_path / "agent_reports" / "code" / "review" / "copy_repo_model" / "code_source_provenance" / "latest.json")
    )
    assert provenance_report["status"] == "PASS"


def test_artifact_tools_write_and_record_ledger(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="demo.json",
            stage="code",
            path="project/demo.json",
            kind="json",
        )
    )
    context_payload = registry.to_context(stage="code", step_id="demo", attempt=1)
    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={"xcientist_artifact_context": context_payload},
    )
    tools = {tool.name: tool for tool in artifact_tools()}

    result = asyncio.run(
        tools["write_artifact"].execute(
            tools["write_artifact"].input_model(
                artifact_id="demo.json",
                json_content={"ok": True},
            ),
            context,
        )
    )

    assert result.is_error is False
    payload = load_json_file(str(tmp_path / "project" / "demo.json"))
    assert payload == {"ok": True}
    ledger = (tmp_path / "agent_reports" / "_runtime" / "artifact_ledger.jsonl").read_text(encoding="utf-8")
    assert '"artifact_id": "demo.json"' in ledger
    assert '"event": "write_artifact"' in ledger


def test_planner_prefinish_gate_requires_managed_plan_artifact(tmp_path):
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_root))
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="code.plan",
            stage="code_planner",
            path="agent_reports/code/plan/latest.json",
            kind="json",
            schema_name="code_plan",
        )
    )
    payload = {
        "stages": [_code_step(tmp_path, project_root, paths)],
        "summary": "code planned",
        "usage_notes": "run from workspace root",
    }
    gate = planner_artifact_prefinish_gate(
        output_schema=planner_output_schema(step_schema={"type": "object"}),
        registry=registry,
        plan_artifact_id="code.plan",
    )

    result = asyncio.run(gate({"assistant_text": json.dumps(payload)}))

    assert result.blocked is True
    assert "managed planner artifact contract is not satisfied" in result.reason
    assert "Required worker-owned artifact has no artifact-tool ledger write: code.plan" in result.reason
    assert "Schema: `code_plan`" in result.reason


def test_planner_prefinish_gate_rejects_plan_artifact_response_drift(tmp_path):
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_root))
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="code.plan",
            stage="code_planner",
            path="agent_reports/code/plan/latest.json",
            kind="json",
            schema_name="code_plan",
        )
    )
    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={"xcientist_artifact_context": registry.to_context(stage="code_planner", step_id="plan", attempt=1)},
    )
    tool = {item.name: item for item in artifact_tools()}["write_artifact"]
    payload = {
        "stages": [_code_step(tmp_path, project_root, paths)],
        "summary": "code planned",
        "usage_notes": "run from workspace root",
    }
    write_result = asyncio.run(
        tool.execute(
            tool.input_model(
                artifact_id="code.plan",
                json_content=payload,
            ),
            context,
        )
    )
    assert write_result.is_error is False, write_result.output
    gate = planner_artifact_prefinish_gate(
        output_schema=planner_output_schema(step_schema={"type": "object"}),
        registry=registry,
        plan_artifact_id="code.plan",
    )

    pass_result = asyncio.run(gate({"assistant_text": json.dumps(payload)}))
    drifted = {
        **payload,
        "stages": [{**payload["stages"][0], "goal": "silently changed after artifact write"}],
    }
    drift_result = asyncio.run(gate({"assistant_text": json.dumps(drifted)}))
    summary_drift_result = asyncio.run(
        gate({"assistant_text": json.dumps({**payload, "summary": "silently changed summary"})})
    )

    assert pass_result.blocked is False
    assert drift_result.blocked is True
    assert "`stages` must exactly match the managed plan artifact" in drift_result.reason
    assert summary_drift_result.blocked is True
    assert "`summary` must exactly match the managed plan artifact" in summary_drift_result.reason


def test_run_artifact_command_rejects_cross_workspace_command(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="demo.json",
            stage="code",
            path="project/demo.json",
            kind="json",
        )
    )
    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={"xcientist_artifact_context": registry.to_context(stage="code", step_id="demo", attempt=1)},
    )
    tool = {item.name: item for item in artifact_tools()}["run_artifact_command"]

    result = asyncio.run(
        tool.execute(
            tool.input_model(
                artifact_id="demo.json",
                command=(
                    "find /aistor/hpc_stor03/sjtu_home/hanqi.li/agent_workspace "
                    "-path '*/agent_reports/science/phase.json' > \"$XCIENTIST_ARTIFACT_PATH\""
                ),
            ),
            context,
        )
    )

    assert result.is_error is True
    assert "workspace-boundary hook blocked" in result.output
    assert "historical Xcientist workspaces" in result.output


def test_run_artifact_command_rejects_relative_parent_workspace_search(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="demo.json",
            stage="code",
            path="project/demo.json",
            kind="json",
        )
    )
    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={"xcientist_artifact_context": registry.to_context(stage="code", step_id="demo", attempt=1)},
    )
    tool = {item.name: item for item in artifact_tools()}["run_artifact_command"]

    result = asyncio.run(
        tool.execute(
            tool.input_model(
                artifact_id="demo.json",
                command="find .. -path '*/agent_reports/science/phase.json' > \"$XCIENTIST_ARTIFACT_PATH\"",
            ),
            context,
        )
    )

    assert result.is_error is True
    assert "workspace-boundary hook blocked" in result.output
    assert "historical Xcientist workspaces" in result.output


def test_record_sources_rejects_cross_workspace_sources(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="demo.json",
            stage="code",
            path="project/demo.json",
            kind="json",
        )
    )
    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={"xcientist_artifact_context": registry.to_context(stage="code", step_id="demo", attempt=1)},
    )
    tool = {item.name: item for item in artifact_tools()}["record_sources"]

    result = asyncio.run(
        tool.execute(
            tool.input_model(
                artifact_id="demo.json",
                sources=[
                    "/aistor/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/Xcientist/workspace/old/project/model.py",
                ],
                reason="copied legacy implementation",
            ),
            context,
        )
    )

    assert result.is_error is True
    assert "source-provenance hook blocked" in result.output
    assert "current experiment workspace" in result.output


def test_repo_copy_intent_requires_exact_enum(tmp_path):
    project_dir = tmp_path / "project"
    source_path = tmp_path / "repos" / "demo" / "model.py"
    project_dir.mkdir()
    source_path.parent.mkdir(parents=True)
    source_path.write_text("class Demo: pass\n", encoding="utf-8")

    errors = validate_repo_contract_fields(
        {
            "repo_source_paths": ["repos/demo/model.py"],
            "repo_copy_intent": "reference_only because no code is copied",
            "project_target_paths": [],
        },
        project_dir=str(project_dir),
        workspace_root=str(tmp_path),
    )

    assert "`repo_copy_intent` must be exactly" in "\n".join(errors)


def test_artifact_tool_rejects_invalid_code_plan_without_overwriting_existing(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    plan_path = tmp_path / "agent_reports" / "code" / "plan" / "latest.json"
    _write_json(
        str(plan_path),
        {
            "stages": [
                {
                    "step_id": "final_integration_smoke",
                    "goal": "run smoke",
                    "input_paths": [],
                    "write_scope": ["project"],
                    "project_target_paths": ["project/smoke.py"],
                    "repo_source_paths": [],
                    "repo_copy_intent": "reference_only",
                    "component_scope": ["all"],
                    "code_artifacts": {"entrypoint_role": "smoke", "paths": ["project/smoke.py"]},
                    "smoke_tests": ["python project/smoke.py"],
                    "repos_policy": "reference_or_copy",
                    "project_must_be_self_contained": True,
                    "done_condition": "final smoke passes",
                    "artifact_ids": [],
                }
            ]
        },
    )
    before = plan_path.read_text(encoding="utf-8")
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="code.plan",
            stage="code_planner",
            path="agent_reports/code/plan/latest.json",
            kind="json",
            schema_name="code_plan",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="code_planner", step_id="plan", attempt=1)}
    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata=metadata,
    )
    tools = {tool.name: tool for tool in artifact_tools()}

    write_result = asyncio.run(
        tools["write_artifact"].execute(
            tools["write_artifact"].input_model(
                artifact_id="code.plan",
                json_content={"stages": [{"step_id": "bad"}]},
            ),
            context,
        )
    )
    assert write_result.is_error is True
    assert "code.plan" in write_result.output
    assert "component_scope" in write_result.output
    assert "final_integration_smoke" in write_result.output
    assert plan_path.read_text(encoding="utf-8") == before


def test_artifact_tool_rejects_placeholder_code_plan_semantics(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    reports_dir = tmp_path / "agent_reports"
    reports_dir.mkdir()
    repos_dir = tmp_path / "repos" / "Graph-WaveNet"
    repos_dir.mkdir(parents=True)
    (repos_dir / "model.py").write_text("class gwnet: pass\n", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    plan_path = reports_dir / "code" / "plan" / "latest.json"
    _write_json(str(plan_path), {"stages": []})

    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="code.plan",
            stage="code_planner",
            path="agent_reports/code/plan/latest.json",
            kind="json",
            schema_name="code_plan",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="code_planner", step_id="plan", attempt=1)}
    context = ToolExecutionContext(cwd=tmp_path, metadata=metadata)
    tools = {tool.name: tool for tool in artifact_tools()}

    placeholder_plan = {
        "stages": [
            {
                "step_id": "test_step",
                "goal": "Test goal",
                "component_scope": ["test_component"],
                "code_artifacts": [
                    {
                        "path": "project/test.py",
                        "artifact_type": "test",
                        "symbols": [],
                        "responsibility": "test",
                        "dependencies": [],
                        "config_keys": [],
                        "entrypoint_role": "test",
                    }
                ],
                "interface_contract": {"test": "test"},
                "implementation_requirements": {"test": "test"},
                "component_disable_hooks": [{"component": "test_component", "flag": "--disable-test"}],
                "experiment_bindings": {"test": "test"},
                "repo_source_paths": ["repos/Graph-WaveNet/model.py"],
                "repo_copy_intent": "reference_only",
                "project_target_paths": ["project/test.py"],
                "input_paths": {},
                "repos_policy": "reference_or_copy",
                "project_must_be_self_contained": True,
                "write_scope": "project/",
                "verify_command": "echo test",
                "done_condition": "test condition",
                "artifact_ids": ["test_artifact"],
            },
            {
                "step_id": "final_integration_smoke",
                "goal": "Final integration smoke test",
                "component_scope": ["component_a"],
                "code_artifacts": [
                    {
                        "path": "project/final.txt",
                        "artifact_type": "test",
                        "symbols": [],
                        "responsibility": "test",
                        "dependencies": [],
                        "config_keys": [],
                        "entrypoint_role": "test",
                    }
                ],
                "interface_contract": {"test": "test"},
                "implementation_requirements": {"test": "test"},
                "component_disable_hooks": [{"component": "component_a", "flag": "--disable-component-a"}],
                "experiment_bindings": {"test": "test"},
                "repo_source_paths": ["repos/Graph-WaveNet/model.py"],
                "repo_copy_intent": "reference_only",
                "project_target_paths": ["project/final.txt"],
                "input_paths": {},
                "repos_policy": "reference_or_copy",
                "project_must_be_self_contained": True,
                "write_scope": "project/",
                "verify_command": "echo test",
                "done_condition": "test condition",
                "artifact_ids": ["code.final_integration_smoke.handoff"],
            },
        ],
        "summary": "Test summary",
        "usage_notes": "Test notes",
    }

    write_result = asyncio.run(
        tools["write_artifact"].execute(
            tools["write_artifact"].input_model(
                artifact_id="code.plan",
                json_content=placeholder_plan,
            ),
            context,
        )
    )

    assert write_result.is_error is True
    assert "Placeholder test/demo values are rejected" in write_result.output
    assert "component_scope" in write_result.output
    assert "verify_command" in write_result.output
    assert load_json_file(str(plan_path)) == {"stages": []}


def test_artifact_post_tool_hook_feedbacks_repeated_successful_writes(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="demo.handoff",
            stage="code",
            path="agent_reports/code/artifacts/demo.handoff.json",
            kind="json",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="code", step_id="demo", attempt=1)}
    context = ToolExecutionContext(cwd=tmp_path, metadata=metadata)
    tools = {tool.name: tool for tool in artifact_tools()}
    hook = XcientistHookExecutor(metadata)

    results = []
    for _ in range(3):
        write_result = asyncio.run(
            tools["write_artifact"].execute(
                tools["write_artifact"].input_model(
                    artifact_id="demo.handoff",
                    json_content={"status": "ok"},
                ),
                context,
            )
        )
        assert write_result.is_error is False
        results.append(
            asyncio.run(
                hook.execute(
                    HookEvent.POST_TOOL_USE,
                    {
                        "tool_name": "write_artifact",
                        "tool_input": {"artifact_id": "demo.handoff"},
                        "tool_output": write_result.output,
                        "tool_is_error": False,
                        "event": HookEvent.POST_TOOL_USE.value,
                    },
                )
            )
        )

    assert results[0].blocked is False
    assert results[1].blocked is False
    assert results[2].blocked is True
    assert "Return the required final JSON response directly" in results[2].reason


def test_step_artifact_registry_keeps_controlled_outputs_in_agent_reports(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_dir))
    cases = [
        ("prepare", {"stage_id": "env"}),
        ("code", _code_step(tmp_path, project_dir, paths)),
        ("science", _component_disabled_step(tmp_path, project_dir, paths)),
    ]

    for scope, step in cases:
        registry = build_step_artifact_registry(
            workspace_root=str(tmp_path),
            scope=scope,
            step=step,
        )
        for spec in registry.worker_required_specs():
            rel = os.path.relpath(spec.resolved_path(str(tmp_path)), str(tmp_path))
            assert rel.startswith("agent_reports" + os.sep), (scope, spec.artifact_id, rel)


def test_code_review_context_always_selects_full_reviewer_matrix(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "runner.py").write_text("def run():\n    return True\n", encoding="utf-8")
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="code.tiny.handoff",
            stage="code",
            path="agent_reports/code/artifacts/tiny.handoff.json",
            kind="json",
        )
    )

    context = build_code_review_context(
        workspace_root=str(tmp_path),
        step={"step_id": "tiny", "project_target_paths": ["project/runner.py"]},
        worker_payload={"artifact_ids_touched": ["code.tiny.handoff"]},
        registry=registry,
        plan_steps=[],
    )

    assert context["run_full_matrix"] is True
    assert context["selected_code_reviewer_ids"] == list(CODE_REVIEWER_IDS)


def test_prepare_artifact_tool_rejects_ready_manifest_with_missing_declared_path(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="prepare.dataset",
            stage="prepare",
            path="agent_reports/prepare/artifacts/dataset.json",
            kind="json",
            schema_name="prepare_dataset",
        )
    )
    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={"xcientist_artifact_context": registry.to_context(stage="prepare", step_id="dataset", attempt=1)},
    )
    tool = {item.name: item for item in artifact_tools()}["write_artifact"]

    result = asyncio.run(
        tool.execute(
            tool.input_model(
                artifact_id="prepare.dataset",
                json_content={
                    "status": "READY",
                    "selection_rationale": "selected real dataset candidate",
                    "selected_datasets": [{"name": "demo"}],
                    "expected_files": ["dataset_candidate/missing.csv"],
                    "schema_probe": {"summary": "schema inspected"},
                    "checksums": {"missing.csv": "abc"},
                },
            ),
            context,
        )
    )

    assert result.is_error is True
    assert "declared path does not exist" in result.output
    assert "Expected schema / repair template" in result.output
    assert "Schema: `prepare_dataset`" in result.output
    assert "READY template" in result.output
    assert "BLOCKED template" in result.output
    assert '"status": "READY"' in result.output


def test_science_artifact_tool_rejects_incomplete_evidence_manifest(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="science.demo.evidence",
            stage="science",
            path="agent_reports/science/evidence/demo.json",
            kind="json",
            schema_name="science_evidence",
        )
    )
    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata={"xcientist_artifact_context": registry.to_context(stage="science", step_id="demo", attempt=1)},
    )
    tool = {item.name: item for item in artifact_tools()}["write_artifact"]

    result = asyncio.run(
        tool.execute(
            tool.input_model(
                artifact_id="science.demo.evidence",
                json_content={"raw_outputs": ["results/science/demo/metrics.json"]},
            ),
            context,
        )
    )

    assert result.is_error is True
    assert "science_evidence missing required `command`" in result.output
    assert "science_evidence missing required `metrics_files`" in result.output
    assert "Expected schema / repair template" in result.output
    assert "Schema: `science_evidence`" in result.output
    assert '"returncode": 0' in result.output
    assert '"metrics_files"' in result.output


def test_artifact_prefinish_hook_feedback_includes_schema_repair_template(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="science.demo.evidence",
            stage="science",
            path="agent_reports/science/evidence/demo.json",
            kind="json",
            schema_name="science_evidence",
        )
    )
    hook = XcientistHookExecutor(
        {"xcientist_artifact_context": registry.to_context(stage="science", step_id="demo", attempt=1)}
    )

    result = asyncio.run(
        hook.execute(
            HookEvent.STOP,
            {
                "event": HookEvent.STOP.value,
                "assistant_text": json.dumps(_worker_report("science.demo.evidence")),
            },
        )
    )

    assert result.blocked is True
    assert "Expected schema / repair templates" in result.reason
    assert "Schema: `science_evidence`" in result.reason
    assert '"returncode": 0' in result.reason
    assert '"output_dir": "results/science/<condition_id>"' in result.reason


def test_stop_hook_runs_artifact_contract_after_successful_prefinish_gate(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="science.demo.evidence",
            stage="science",
            path="agent_reports/science/evidence/demo.json",
            kind="json",
            schema_name="science_evidence",
        )
    )
    hook = XcientistHookExecutor(
        {
            "xcientist_artifact_context": registry.to_context(stage="science", step_id="demo", attempt=1),
            "xcientist_prefinish_gate": lambda _payload: {"status": "PASS"},
        }
    )

    result = asyncio.run(
        hook.execute(
            HookEvent.STOP,
            {
                "event": HookEvent.STOP.value,
                "assistant_text": json.dumps(_worker_report("science.demo.evidence")),
            },
        )
    )

    assert result.blocked is True
    assert any(item.hook_type == "xcientist_prefinish_gate" and item.success for item in result.results)
    assert any(item.hook_type == "xcientist_artifact_prefinish" and item.blocked for item in result.results)
    assert "Required worker-owned artifact has no artifact-tool ledger write: science.demo.evidence" in result.reason
    assert "Expected schema / repair templates" in result.reason


def test_xcientist_hook_blocks_direct_write_to_managed_artifact(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="demo.file",
            stage="code",
            path="project/demo.py",
            kind="file",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="code", step_id="demo", attempt=1)}
    hook = XcientistHookExecutor(metadata)

    result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "write_file",
                "tool_input": {"path": "project/demo.py", "content": "x = 1\n"},
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )

    assert result.blocked is True
    assert "write_artifact" in result.reason


def test_xcientist_hook_blocks_direct_write_to_runtime_report(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="runtime.code.demo.worker_report",
            stage="code",
            path="agent_reports/code/worker/demo/latest.json",
            kind="json",
            required=False,
            writer="runtime",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="code", step_id="demo", attempt=1)}
    hook = XcientistHookExecutor(metadata)

    result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "write_file",
                "tool_input": {
                    "path": "agent_reports/code/worker/demo/latest.json",
                    "content": "{}",
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )

    assert result.blocked is True
    assert "Direct write to runtime-owned report path is blocked" in result.reason


def test_xcientist_hook_blocks_implicit_planner_report_writes(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="code.plan",
            stage="code_planner",
            path="agent_reports/code/plan/latest.json",
            kind="json",
            schema_name="code_plan",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="code_planner", step_id="plan", attempt=1)}
    hook = XcientistHookExecutor(metadata)

    write_result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "write_file",
                "tool_input": {
                    "path": "agent_reports/code/plan/planner_report.json",
                    "content": "{}",
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )
    bash_result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "bash",
                "tool_input": {
                    "command": "printf '{}' > agent_reports/code/plan/planner_report.json",
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )

    assert write_result.blocked is True
    assert bash_result.blocked is True
    assert "Return the required structured JSON response directly" in write_result.reason
    assert "runtime-owned report" in bash_result.reason


def test_xcientist_hook_blocks_direct_write_to_ablation_final_output(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="science.demo.evidence",
            stage="science",
            path="agent_reports/science/evidence/demo.json",
            kind="json",
            schema_name="science_evidence",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="science", step_id="demo", attempt=1)}
    hook = XcientistHookExecutor(metadata)

    result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "write_file",
                "tool_input": {
                    "path": "agent_reports/ablation/final/ablation_results.json",
                    "content": "{}",
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )

    assert result.blocked is True
    assert "agent_reports/ablation/final/ablation_results.json" in result.reason
    assert "runtime will persist reports" in result.reason


def test_run_artifact_command_blocks_side_effect_write_to_ablation_final_output(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="science.demo.evidence",
            stage="science",
            path="agent_reports/science/evidence/demo.json",
            kind="json",
            schema_name="science_evidence",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="science", step_id="demo", attempt=1)}
    hook = XcientistHookExecutor(metadata)

    result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "run_artifact_command",
                "tool_input": {
                    "artifact_id": "science.demo.evidence",
                    "command": "printf '{}' > agent_reports/ablation/final/ablation_results.json",
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )

    assert result.blocked is True
    assert "run_artifact_command" in result.reason
    assert "agent_reports/ablation/final/ablation_results.json" in result.reason


def test_workspace_hygiene_hook_blocks_unmanaged_top_level_outputs(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="science.demo.evidence",
            stage="science",
            path="agent_reports/science/evidence/demo.json",
            kind="json",
        )
    )
    unmanaged_output_dir = tmp_path / "unmanaged_outputs" / "metr"
    unmanaged_output_dir.mkdir(parents=True)
    (unmanaged_output_dir / "checkpoint.pt").write_text("bad", encoding="utf-8")
    metadata = {"xcientist_artifact_context": registry.to_context(stage="science", step_id="demo", attempt=1)}
    hook = XcientistHookExecutor(metadata)

    result = asyncio.run(
        hook.execute(
            HookEvent.STOP,
            {
                "event": HookEvent.STOP.value,
                "assistant_text": json.dumps(_worker_report("science.demo.evidence")),
            },
        )
    )

    assert result.blocked is True
    assert "outside declared output roots" in result.reason
    assert "checkpoint.pt" in result.reason
    assert (tmp_path / "agent_reports" / "_runtime" / "stray_outputs.json").exists()


def test_xcientist_hook_blocks_cross_workspace_reads_and_searches(tmp_path):
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="prepare.plan",
            stage="prepare_planner",
            path="agent_reports/prepare/plan/latest.json",
            kind="json",
            schema_name="prepare_plan",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="prepare_planner", step_id="plan", attempt=1)}
    hook = XcientistHookExecutor(metadata)

    read_result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "read_file",
                "tool_input": {
                    "path": "/aistor/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/Xcientist/workspace/old/agent_reports/prepare/plan/latest.json",
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )
    grep_result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "grep",
                "tool_input": {
                    "pattern": "prepare",
                    "root": "/aistor/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/Xcientist/workspace",
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )
    bash_result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "bash",
                "tool_input": {
                    "command": "find /aistor/hpc_stor03/sjtu_home/hanqi.li/agent_workspace -path '*/agent_reports/prepare/plan/latest.json'",
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )
    repo_workspace_result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "read_file",
                "tool_input": {
                    "path": os.path.join(
                        project_root,
                        "workspace",
                        "old_run",
                        "agent_reports",
                        "science",
                        "phase.json",
                    ),
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )
    relative_bash_result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "bash",
                "tool_input": {
                    "command": "find .. -path '*/agent_reports/prepare/plan/latest.json'",
                },
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )
    workspace_result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "read_file",
                "tool_input": {"path": str(tmp_path / "idea.json")},
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )
    source_result = asyncio.run(
        hook.execute(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": "read_file",
                "tool_input": {"path": os.path.join(project_root, "README.md")},
                "event": HookEvent.PRE_TOOL_USE.value,
            },
        )
    )

    assert read_result.blocked is True
    assert grep_result.blocked is True
    assert bash_result.blocked is True
    assert repo_workspace_result.blocked is True
    assert relative_bash_result.blocked is True
    assert "workspace-boundary hook blocked" in read_result.reason
    assert "historical Xcientist workspaces" in bash_result.reason
    assert "repository-level workspace history" in repo_workspace_result.reason
    assert "historical Xcientist workspaces" in relative_bash_result.reason
    assert workspace_result.blocked is False
    assert source_result.blocked is False


def test_science_agent_aggregates_reviewer_component_results(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_dir))
    _write_idea_json(tmp_path / "idea.json")
    evidence_path = tmp_path / "results" / "science" / "without_component_a" / "metrics.json"

    plan_payload = {
        "stages": [_reference_step(tmp_path, project_dir, paths), _component_disabled_step(tmp_path, project_dir, paths)],
        "summary": "science planned",
        "usage_notes": "run science",
    }
    def _science_reviews(condition_id, enabled, disabled, reference, checked_artifact, *, result, value, analysis):
        reviews = []
        for reviewer_id in (
            "protocol_compliance",
            "protocol_semantics",
            "condition_toggle",
            "evidence_plausibility",
            "statistical_interpretation",
            "idea_alignment",
        ):
            structured = {}
            if reviewer_id == "statistical_interpretation":
                structured = {
                    "condition_id": condition_id,
                    "enabled_components": enabled,
                    "disabled_components": disabled,
                    "reference_condition_id": reference,
                    "component_result": {
                        "result": result,
                        "metric": "acc",
                        "value": value,
                        "confidence": 0.9,
                        "analysis": analysis,
                        "method_context": "component_a disabled" if disabled else "all components",
                        "follow_up_required": False,
                    },
                }
            reviews.append(
                {"output": _matrix_review(reviewer_id, checked_artifact=str(checked_artifact), **structured)}
            )
        return reviews

    responses = iter(
        [
            {"output": plan_payload},
            {"output": _worker_report("science.full_component_a.evidence")},
            *_science_reviews(
                "full_component_a",
                ["component_a"],
                [],
                None,
                tmp_path / "results" / "science" / "full_component_a" / "metrics.json",
                result="positive",
                value="1.0",
                analysis="reference ran",
            ),
            {"output": _worker_report("science.without_component_a.evidence")},
            *_science_reviews(
                "without_component_a",
                [],
                ["component_a"],
                "full_component_a",
                evidence_path,
                result="positive",
                value="+0.1",
                analysis="component matters",
            ),
        ]
    )
    calls = []

    async def fake_run(self, *args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("agent_name") == EXPERIMENT_SCIENCE_PLANNER:
            planner_response = next(responses)
            await _write_artifact_json(self.artifact_context, "science.plan", planner_response["output"])
            gate = (kwargs.get("extra_tool_metadata") or {}).get("xcientist_prefinish_gate")
            assert gate is not None
            gate_result = await gate({"assistant_text": json.dumps(planner_response["output"])})
            assert gate_result.blocked is False
            return planner_response
        if kwargs.get("agent_name") == SCIENCE_WORKER:
            artifact_id = next(
                spec["artifact_id"]
                for spec in self.artifact_context["artifact_specs"]
                if spec["artifact_id"].startswith("science.") and spec["artifact_id"].endswith(".evidence")
            )
            condition_id = artifact_id.split(".")[1]
            await _run_artifact_command(
                self.artifact_context,
                artifact_id,
                _science_evidence_command(next(step for step in plan_payload["stages"] if step["condition_id"] == condition_id)),
            )
            worker_response = next(responses)
            gate = (kwargs.get("extra_tool_metadata") or {}).get("xcientist_prefinish_gate")
            assert gate is not None
            gate_result = await gate({"assistant_text": json.dumps(worker_response["output"])})
            assert gate_result.blocked is False
            return worker_response
        return next(responses)

    monkeypatch.setattr(ScienceAgent, "run", fake_run)
    result = asyncio.run(
        run_science_agent(
            experiment_id="demo",
            idea_path=str(tmp_path / "idea.md"),
            project_root=str(project_dir),
            workspace_root=str(tmp_path),
            plan="run science",
            verbose=False,
        )
    )

    assert result["status"] == "completed"
    phase_payload = load_workspace_state(str(tmp_path))["science_reviewer"]
    assert phase_payload["science_component_results"]["component_a"]["result"] == "positive"
    assert phase_payload["summary"]["feasible"] is True
    assert [call.get("agent_name") for call in calls] == [
        EXPERIMENT_SCIENCE_PLANNER,
        SCIENCE_WORKER,
        *list(SCIENCE_REVIEWER_IDS),
        SCIENCE_WORKER,
        *list(SCIENCE_REVIEWER_IDS),
    ]
    assert calls[0]["enable_mcp"] is True
    assert calls[-1]["purpose"] == "prefinish_review"


def test_science_phase_summary_blocks_missing_component_result(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_json(
        str(tmp_path / "idea.json"),
        {
            "title": "demo",
            "components": [
                {"component": "component_a", "explanation": "A"},
                {"component": "component_b", "explanation": "B"},
            ],
        },
    )
    agent = ScienceAgent(
        experiment_id="demo",
        idea_path=str(tmp_path / "idea.md"),
        project_root=str(project_dir),
        workspace_root=str(tmp_path),
        plan="run science",
        verbose=False,
    )
    plan_payload = {
        "stages": [
            {
                "condition_id": "full",
                "enabled_components": ["component_a", "component_b"],
                "disabled_components": [],
            },
            {
                "condition_id": "without_component_a",
                "enabled_components": ["component_b"],
                "disabled_components": ["component_a"],
                "reference_condition_id": "full",
            },
        ],
        "summary": "science planned",
        "usage_notes": "run science",
    }
    step_result = {
        "status": "PASS",
        "step_reports": [
            {"status": "PASS"},
            {
                "status": "PASS",
                "result": "positive",
                "metric": "acc",
                "value": "+0.1",
                "confidence": 0.9,
                "analysis": "component_a mattered",
                "method_context": "context",
                "follow_up_required": False,
            },
        ],
    }

    phase_payload = agent._phase_summary_payload(plan_payload, step_result)

    assert phase_payload["status"] == "FAIL"
    assert phase_payload["ready_for_next_phase"] is False
    assert "component_b" in phase_payload["next_worker_input"]
    assert "component_a" in phase_payload["science_component_results"]


def test_final_ablation_materialization_uses_reviewer_evidence(tmp_path):
    _write_idea_json(tmp_path / "idea.json")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_dir))
    _write_json(
        paths["science_reviewer"],
        _pass_review(
            "science",
            science_component_results={
                "component_a": {
                    "result": "positive",
                    "metric": "acc",
                    "value": "+0.1",
                    "confidence": 0.9,
                    "analysis": "strong effect",
                    "method_context": "ignored by deterministic materializer",
                    "follow_up_required": False,
                }
            },
            summary={"feasible": True, "confidence": 0.9, "key_findings": ["works"]},
        ),
    )

    result = write_ablation_results_artifacts(str(tmp_path), str(project_dir), generated_by="prefinish_hook")
    assert result["valid"] is True
    payload = load_workspace_state(str(tmp_path))["ablation_results"]
    assert payload["components"]["component_a"]["method_context"] == "demo explanation"
    report = load_workspace_state(str(tmp_path))["ablation_materialization_report"]
    assert report["generated_by"] == "prefinish_hook"
    assert os.path.exists(paths["ablation_results_manifest"])


def test_final_ablation_materialization_accepts_inconclusive_result(tmp_path):
    _write_idea_json(tmp_path / "idea.json")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_dir))
    _write_json(
        paths["science_reviewer"],
        _pass_review(
            "science",
            science_component_results={
                "component_a": {
                    "result": "inconclusive",
                    "metric": "acc",
                    "value": "+0.0",
                    "confidence": 0.2,
                    "analysis": "completed comparison without directional conclusion",
                    "method_context": "ignored by deterministic materializer",
                    "follow_up_required": False,
                }
            },
            summary={"feasible": True, "confidence": 0.2, "key_findings": ["component_a was inconclusive"]},
        ),
    )

    result = write_ablation_results_artifacts(str(tmp_path), str(project_dir))

    assert result["valid"] is True
    payload = load_workspace_state(str(tmp_path))["ablation_results"]
    assert payload["components"]["component_a"]["result"] == "inconclusive"


def test_final_ablation_materialization_rejects_non_pass_reviewer(tmp_path):
    _write_idea_json(tmp_path / "idea.json")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_dir))
    _write_json(paths["science_reviewer"], {"status": "FAIL", "scope": "science"})

    result = write_ablation_results_artifacts(str(tmp_path), str(project_dir))
    assert result["valid"] is False


def test_final_ablation_materialization_requires_phase_component_results(tmp_path):
    _write_idea_json(tmp_path / "idea.json")
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_dir))
    _write_json(
        paths["science_reviewer"],
        _pass_review(
            "science",
            summary={"feasible": True, "confidence": 0.9, "key_findings": ["works"]},
        ),
    )
    step_review_path = (
        tmp_path
        / "agent_reports"
        / "science"
        / "review"
        / "without_component_a"
        / "latest.json"
    )
    _write_json(
        str(step_review_path),
        {
            "disabled_components": ["component_a"],
            "result": "positive",
            "metric": "acc",
            "value": "+0.1",
            "confidence": 0.9,
            "analysis": "strong effect",
            "follow_up_required": False,
        },
    )

    result = write_ablation_results_artifacts(str(tmp_path), str(project_dir))

    assert result["valid"] is False
    assert "science phase report" in result["blocker"]
    assert "science_component_results" in result["blocker"]
    assert str(step_review_path) not in result["source_evidence_files"]


def test_prepare_plan_order_feedback_uses_artifact_hook(tmp_path):
    (tmp_path / "project").mkdir()
    wrong_plan = {
        "stages": [
            {
                "stage_id": "env",
                "goal": "",
                "input_paths": {},
                "repos_policy": "reference_or_copy",
                "project_must_be_self_contained": True,
                "research_required": True,
                "acquisition_required": True,
                "existing_local_hints": [],
                "done_condition": "ok",
                "artifact_ids": ["prepare.env"],
            },
            {
                "stage_id": "repos",
                "goal": "",
                "input_paths": {},
                "repos_policy": "reference_or_copy",
                "project_must_be_self_contained": True,
                "research_required": True,
                "acquisition_required": True,
                "existing_local_hints": [],
                "done_condition": "ok",
                "artifact_ids": ["prepare.repos"],
            },
        ],
        "summary": "bad",
        "usage_notes": "",
    }
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="prepare.plan",
            stage="prepare_planner",
            path="agent_reports/prepare/plan/latest.json",
            kind="json",
            schema_name="prepare_plan",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="prepare_planner", step_id="plan", attempt=1)}
    context = ToolExecutionContext(
        cwd=tmp_path,
        metadata=metadata,
    )
    tools = {tool.name: tool for tool in artifact_tools()}

    write_result = asyncio.run(
        tools["write_artifact"].execute(
            tools["write_artifact"].input_model(
                artifact_id="prepare.plan",
                json_content={"stages": wrong_plan["stages"]},
            ),
            context,
        )
    )
    assert write_result.is_error is True
    assert "ordered exactly" in write_result.output
    assert "repos" in write_result.output
    assert "Repair template" in write_result.output
    assert "agent_reports/prepare/artifacts/discovery.json" in write_result.output
    assert '"stage_id": "synthesis"' in write_result.output


def test_science_plan_protocol_feedback_uses_artifact_hook(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_idea_json(tmp_path / "idea.json")
    plan_path = tmp_path / "agent_reports" / "science" / "plan" / "latest.json"
    _write_json(str(plan_path), {"stages": []})
    bad_step = _reference_step(tmp_path, project_dir, artifact_paths(str(tmp_path), str(project_dir)))
    for key in (
        "run_level",
        "setup_rationale",
        "source_basis",
        "runtime_probe_summary",
        "training_protocol",
        "evaluation_protocol",
    ):
        bad_step.pop(key, None)
    registry = ArtifactRegistry(workspace_root=str(tmp_path))
    registry.add(
        ArtifactSpec(
            artifact_id="science.plan",
            stage="science_planner",
            path="agent_reports/science/plan/latest.json",
            kind="json",
            schema_name="science_plan",
        )
    )
    metadata = {"xcientist_artifact_context": registry.to_context(stage="science_planner", step_id="plan", attempt=1)}
    context = ToolExecutionContext(cwd=tmp_path, metadata=metadata)
    tools = {tool.name: tool for tool in artifact_tools()}

    write_result = asyncio.run(
        tools["write_artifact"].execute(
            tools["write_artifact"].input_model(
                artifact_id="science.plan",
                json_content={"stages": [bad_step], "summary": "bad", "usage_notes": ""},
            ),
            context,
        )
    )

    assert write_result.is_error is True
    assert "run_level" in write_result.output
    assert "training_protocol" in write_result.output
    assert "Repair template" in write_result.output
    assert '"condition_id": "full_all_components"' in write_result.output
    assert '"component_result"' not in write_result.output
    assert load_json_file(str(plan_path)) == {"stages": []}


def test_science_plan_rejects_multi_component_disabled_condition(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_json(
        str(tmp_path / "idea.json"),
        {
            "title": "demo",
            "components": [
                {"component": "component_a", "explanation": "first"},
                {"component": "component_b", "explanation": "second"},
            ],
        },
    )
    paths = artifact_paths(str(tmp_path), str(project_dir))
    reference = _reference_step(tmp_path, project_dir, paths)
    reference["enabled_components"] = ["component_a", "component_b"]
    disabled = _component_disabled_step(tmp_path, project_dir, paths)
    disabled["condition_id"] = "without_component_a_b"
    disabled["enabled_components"] = []
    disabled["disabled_components"] = ["component_a", "component_b"]
    disabled["output_dir"] = "results/science/without_component_a_b"
    disabled["raw_evidence"] = ["results/science/without_component_a_b/metrics.json"]
    disabled["artifact_ids"] = ["science.without_component_a_b.evidence"]
    issues = validate_science_condition_plan(
        {"stages": [reference, disabled]},
        project_dir=str(project_dir),
        workspace_root=str(tmp_path),
    )
    assert any("disable exactly one" in issue for issue in issues)


def test_science_plan_requires_exact_reference_plus_one_ablation_per_component(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_json(
        str(tmp_path / "idea.json"),
        {
            "title": "demo",
            "components": [
                {"component": "component_a", "explanation": "first"},
                {"component": "component_b", "explanation": "second"},
            ],
        },
    )
    paths = artifact_paths(str(tmp_path), str(project_dir))
    reference = _reference_step(tmp_path, project_dir, paths)
    reference["enabled_components"] = ["component_a", "component_b"]
    without_a = _component_disabled_step(tmp_path, project_dir, paths)
    without_a["enabled_components"] = ["component_b"]
    without_a["disabled_components"] = ["component_a"]

    issues = validate_science_condition_plan(
        {"stages": [reference, without_a]},
        project_dir=str(project_dir),
        workspace_root=str(tmp_path),
    )

    joined = "\n".join(issues)
    assert "1 + len(idea.json.components)" in joined
    assert "missing=component_b" in joined
    assert "Expected science plan pattern" in joined


def test_science_plan_rejects_multiple_all_component_references(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_idea_json(tmp_path / "idea.json")
    paths = artifact_paths(str(tmp_path), str(project_dir))
    first_reference = _reference_step(tmp_path, project_dir, paths)
    second_reference = _reference_step(tmp_path, project_dir, paths)
    second_reference["condition_id"] = "full_component_a_repeat"
    second_reference["output_dir"] = "results/science/full_component_a_repeat"
    second_reference["raw_evidence"] = ["results/science/full_component_a_repeat/metrics.json"]
    second_reference["artifact_ids"] = ["science.full_component_a_repeat.evidence"]
    without_a = _component_disabled_step(tmp_path, project_dir, paths)

    issues = validate_science_condition_plan(
        {"stages": [first_reference, second_reference, without_a]},
        project_dir=str(project_dir),
        workspace_root=str(tmp_path),
    )

    assert any("exactly one all-components reference" in issue for issue in issues)


def test_code_plan_schema_repair_guide_includes_full_step_template():
    from src.agents.experiment_agent.agents.code.planner import _code_step_schema

    guide = artifact_schema_repair_guide(
        "code_plan",
        artifact_id="code.plan",
        path="agent_reports/code/plan/latest.json",
    )
    template = _first_json_fence(guide)
    issues = validate_json_schema_fragment(
        template,
        planner_output_schema(step_schema=_code_step_schema()),
    )

    assert issues == []
    assert "Schema: `code_plan`" in guide
    assert "Repair template" in guide
    assert '"step_id": "final_integration_smoke"' in guide
    assert '"code_artifacts"' in guide
    assert '"component_disable_hooks"' in guide
    assert "code.final_integration_smoke.evidence" in guide


def test_prepare_plan_schema_repair_guide_matches_planner_schema():
    from src.agents.experiment_agent.agents.prepare.planner import _prepare_step_schema

    guide = artifact_schema_repair_guide(
        "prepare_plan",
        artifact_id="prepare.plan",
        path="agent_reports/prepare/plan/latest.json",
    )
    template = _first_json_fence(guide)
    issues = validate_json_schema_fragment(
        template,
        planner_output_schema(step_schema=_prepare_step_schema()),
    )

    assert issues == []


def test_materialize_executable_plan_does_not_overwrite_managed_latest(tmp_path):
    guide = artifact_schema_repair_guide(
        "prepare_plan",
        artifact_id="prepare.plan",
        path="agent_reports/prepare/plan/latest.json",
    )
    plan_payload = _first_json_fence(guide)
    latest_path = tmp_path / "agent_reports" / "prepare" / "plan" / "latest.json"
    latest_path.parent.mkdir(parents=True)
    original_text = json.dumps(plan_payload, ensure_ascii=False, separators=(",", ":"))
    latest_path.write_text(original_text, encoding="utf-8")

    paths = materialize_executable_plan(
        workspace_root=str(tmp_path),
        scope="prepare",
        plan_payload=plan_payload,
        planner_report={"scope": "prepare", "summary": "prepared", "usage_notes": "use artifacts"},
    )

    assert latest_path.read_text(encoding="utf-8") == original_text
    assert os.path.exists(paths["executable"])
    assert os.path.exists(paths["planner_report"])


def test_materialize_executable_plan_rejects_invalid_managed_plan(tmp_path):
    latest_path = tmp_path / "agent_reports" / "prepare" / "plan" / "latest.json"
    latest_path.parent.mkdir(parents=True)
    plan_payload = {"stages": [], "summary": "bad", "usage_notes": ""}
    latest_path.write_text(json.dumps(plan_payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed executable plan contract"):
        materialize_executable_plan(
            workspace_root=str(tmp_path),
            scope="prepare",
            plan_payload=plan_payload,
            planner_report={"scope": "prepare"},
        )

    assert not (tmp_path / "agent_reports" / "prepare" / "plan" / "executable.json").exists()


def test_science_plan_schema_repair_guide_matches_planner_schema():
    from src.agents.experiment_agent.agents.science.planner import _science_step_schema

    guide = artifact_schema_repair_guide(
        "science_plan",
        artifact_id="science.plan",
        path="agent_reports/science/plan/latest.json",
    )
    template = _first_json_fence(guide)
    issues = validate_json_schema_fragment(
        template,
        planner_output_schema(step_schema=_science_step_schema()),
    )

    assert issues == []
    first_step = template["stages"][0]
    assert isinstance(first_step["train_dataset_binding"], dict)
    assert isinstance(first_step["evaluation_dataset_bindings"][0], dict)
    assert isinstance(first_step["metric_bindings"][0], dict)


def test_science_prefinish_blocks_invalid_component_result_enum(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_idea_json(tmp_path / "idea.json")
    steps = [_reference_step(tmp_path, project_dir, artifact_paths(str(tmp_path), str(project_dir))), _component_disabled_step(tmp_path, project_dir, artifact_paths(str(tmp_path), str(project_dir)))]

    async def call_worker(step, _previous_review, artifact_context, prefinish_gate=None):
        condition_id = step["condition_id"]
        artifact_id = f"science.{condition_id}.evidence"
        await _run_artifact_command(
            artifact_context,
            artifact_id,
            _science_evidence_command(step),
        )
        worker_payload = _worker_report(artifact_id)
        assert prefinish_gate is not None
        result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        if condition_id == "full_component_a":
            assert result.blocked is False
        else:
            assert result.blocked is True
            assert "result" in result.reason
            assert "structured_findings.component_result.result" in result.reason
            assert '"component_result"' in result.reason
        return worker_payload

    async def call_reviewer(step, _worker_payload, _artifact_context):
        condition_id = step["condition_id"]
        disabled = step.get("disabled_components") or []
        return [
            _matrix_review(
                "statistical_interpretation",
                checked_artifact=f"results/science/{condition_id}/metrics.json",
                condition_id=condition_id,
                enabled_components=step.get("enabled_components") or [],
                disabled_components=disabled,
                reference_condition_id=step.get("reference_condition_id"),
                component_result={
                    "result": "custom_free_text" if disabled else "positive",
                    "metric": "acc",
                    "value": "+0.1",
                    "confidence": 0.9,
                    "analysis": "interpretation",
                    "method_context": "context",
                    "follow_up_required": False,
                },
            )
        ]

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=steps,
            scope="science",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "FAIL"
    latest = load_json_file(str(tmp_path / "agent_reports" / "science" / "review" / "without_component_a" / "science_result_contract" / "latest.json"))
    assert latest["status"] == "FAIL"
    assert "custom_free_text" in latest["issues"][0]["message"]
    assert "expected_component_result" in latest["structured_findings"]["science_result_contract"]


def test_science_prefinish_blocks_follow_up_required_component_result(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_idea_json(tmp_path / "idea.json")
    steps = [_reference_step(tmp_path, project_dir, artifact_paths(str(tmp_path), str(project_dir))), _component_disabled_step(tmp_path, project_dir, artifact_paths(str(tmp_path), str(project_dir)))]

    async def call_worker(step, _previous_review, artifact_context, prefinish_gate=None):
        condition_id = step["condition_id"]
        artifact_id = f"science.{condition_id}.evidence"
        await _run_artifact_command(
            artifact_context,
            artifact_id,
            _science_evidence_command(step),
        )
        worker_payload = _worker_report(artifact_id)
        assert prefinish_gate is not None
        result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        if condition_id == "full_component_a":
            assert result.blocked is False
        else:
            assert result.blocked is True
            assert "follow_up_required" in result.reason
        return worker_payload

    async def call_reviewer(step, _worker_payload, _artifact_context):
        condition_id = step["condition_id"]
        disabled = step.get("disabled_components") or []
        return [
            _matrix_review(
                "statistical_interpretation",
                checked_artifact=f"results/science/{condition_id}/metrics.json",
                condition_id=condition_id,
                enabled_components=step.get("enabled_components") or [],
                disabled_components=disabled,
                reference_condition_id=step.get("reference_condition_id"),
                component_result={
                    "result": "inconclusive" if disabled else "positive",
                    "metric": "acc",
                    "value": "+0.1",
                    "confidence": 0.5 if disabled else 0.9,
                    "analysis": "interpretation",
                    "method_context": "context",
                    "follow_up_required": bool(disabled),
                },
            )
        ]

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=steps,
            scope="science",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "FAIL"
    latest = load_json_file(str(tmp_path / "agent_reports" / "science" / "review" / "without_component_a" / "science_result_contract" / "latest.json"))
    assert latest["status"] == "FAIL"
    messages = "\n".join(issue["message"] for issue in latest["issues"])
    assert "follow_up_required" in messages


def test_science_prefinish_accepts_inconclusive_without_followup(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_idea_json(tmp_path / "idea.json")
    steps = [_reference_step(tmp_path, project_dir, artifact_paths(str(tmp_path), str(project_dir))), _component_disabled_step(tmp_path, project_dir, artifact_paths(str(tmp_path), str(project_dir)))]

    async def call_worker(step, _previous_review, artifact_context, prefinish_gate=None):
        condition_id = step["condition_id"]
        artifact_id = f"science.{condition_id}.evidence"
        await _run_artifact_command(
            artifact_context,
            artifact_id,
            _science_evidence_command(step),
        )
        worker_payload = _worker_report(artifact_id)
        assert prefinish_gate is not None
        result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        assert result.blocked is False
        return worker_payload

    async def call_reviewer(step, _worker_payload, _artifact_context):
        condition_id = step["condition_id"]
        disabled = step.get("disabled_components") or []
        return [
            _matrix_review(
                "statistical_interpretation",
                checked_artifact=f"results/science/{condition_id}/metrics.json",
                condition_id=condition_id,
                enabled_components=step.get("enabled_components") or [],
                disabled_components=disabled,
                reference_condition_id=step.get("reference_condition_id"),
                component_result={
                    "result": "inconclusive" if disabled else "positive",
                    "metric": "acc",
                    "value": "+0.0",
                    "confidence": 0.2 if disabled else 0.9,
                    "analysis": "completed comparison without directional conclusion",
                    "method_context": "context",
                    "follow_up_required": False,
                },
            )
        ]

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=steps,
            scope="science",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "PASS"
    latest = load_json_file(str(tmp_path / "agent_reports" / "science" / "review" / "without_component_a" / "science_result_contract" / "latest.json"))
    assert latest["status"] == "PASS"


def test_science_component_result_must_come_from_statistical_reviewer(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    _write_idea_json(tmp_path / "idea.json")
    steps = [_reference_step(tmp_path, project_dir, artifact_paths(str(tmp_path), str(project_dir))), _component_disabled_step(tmp_path, project_dir, artifact_paths(str(tmp_path), str(project_dir)))]

    async def call_worker(step, _previous_review, artifact_context, prefinish_gate=None):
        condition_id = step["condition_id"]
        artifact_id = f"science.{condition_id}.evidence"
        await _run_artifact_command(
            artifact_context,
            artifact_id,
            _science_evidence_command(step),
        )
        worker_payload = _worker_report(artifact_id)
        assert prefinish_gate is not None
        result = await prefinish_gate({"assistant_text": json.dumps(worker_payload)})
        if condition_id == "full_component_a":
            assert result.blocked is False
        else:
            assert result.blocked is True
            assert "statistical" in result.reason or "component_result.result" in result.reason
        return worker_payload

    async def call_reviewer(step, _worker_payload, _artifact_context):
        condition_id = step["condition_id"]
        disabled = step.get("disabled_components") or []
        component_result = {
            "condition_id": condition_id,
            "enabled_components": step.get("enabled_components") or [],
            "disabled_components": disabled,
            "reference_condition_id": step.get("reference_condition_id"),
            "component_result": {
                "result": "positive",
                "metric": "acc",
                "value": "+0.1",
                "confidence": 0.9,
                "analysis": "interpretation",
                "method_context": "context",
                "follow_up_required": False,
            },
        }
        return [
            _matrix_review(
                "idea_alignment",
                checked_artifact=f"results/science/{condition_id}/metrics.json",
                **component_result,
            ),
            _matrix_review(
                "statistical_interpretation",
                checked_artifact=f"results/science/{condition_id}/metrics.json",
            ),
        ]

    result = asyncio.run(
        execute_step_with_prefinish_review(
            steps=steps,
            scope="science",
            workspace_root=str(tmp_path),
            call_worker=call_worker,
            call_reviewer=call_reviewer,
        )
    )

    assert result["status"] == "FAIL"
    latest = load_json_file(str(tmp_path / "agent_reports" / "science" / "review" / "without_component_a" / "science_result_contract" / "latest.json"))
    assert latest["status"] == "FAIL"
    assert "component_result.result" in latest["issues"][0]["message"]


def test_prepare_plan_rejects_contracts_instead_of_runtime_normalizing(tmp_path):
    from src.agents.experiment_agent.runtime.prepare_contracts import validate_prepare_plan

    stage = {
        "stage_id": "repos",
        "goal": "verify repos",
        "input_paths": {},
        "repos_policy": "symlink_reference_only",
        "project_must_be_self_contained": False,
        "research_required": False,
        "acquisition_required": False,
        "existing_local_hints": [],
        "done_condition": "repos verified",
        "artifact_ids": ["prepare.repos"],
    }

    issues = validate_prepare_plan({"stages": [stage]})

    assert any("ordered exactly" in issue for issue in issues)
    assert any("repos_policy must be exactly `reference_or_copy`" in issue for issue in issues)
    assert any("project_must_be_self_contained must be true" in issue for issue in issues)
    assert any("artifact_ids must be exactly" in issue for issue in issues)
    assert any("artifact tools" in issue for issue in issues)


def test_prepare_plan_requires_acquisition_order_and_artifact_ids():
    from src.agents.experiment_agent.runtime.prepare_contracts import validate_prepare_plan

    stages = []
    for stage_id, artifact_ids in [
        ("repos", ["prepare.discovery", "prepare.repos"]),
        ("dataset", ["prepare.dataset"]),
        ("model", ["prepare.model"]),
        ("env", ["prepare.env"]),
        ("synthesis", ["prepare.idea", "prepare.target_inventory"]),
    ]:
        stages.append(
            {
                "stage_id": stage_id,
                "goal": f"{stage_id} goal",
                "input_paths": {},
                "repos_policy": "reference_or_copy",
                "project_must_be_self_contained": True,
                "research_required": True,
                "acquisition_required": True,
                "existing_local_hints": [],
                "done_condition": " ".join(
                    [
                        "Must use Xcientist artifact tools.",
                        "The artifact ledger is the proof.",
                        "agent_reports/prepare/artifacts/discovery.json",
                        "agent_reports/prepare/artifacts/repos.json",
                        "agent_reports/prepare/artifacts/dataset.json",
                        "agent_reports/prepare/artifacts/model.json",
                        "agent_reports/prepare/artifacts/env.json",
                        "agent_reports/prepare/artifacts/idea.md",
                        "agent_reports/prepare/artifacts/target_inventory.json",
                        "## Canonical Idea Components",
                        "target_inventory.json maps every idea.json component to concrete implementation targets.",
                    ]
                ),
                "artifact_ids": artifact_ids,
            }
        )

    assert validate_prepare_plan({"stages": stages}) == []
    bad = {"stages": [stages[0], stages[3], stages[1], stages[2], stages[4]]}
    assert any("ordered exactly" in issue for issue in validate_prepare_plan(bad))


def test_prepare_failure_does_not_runtime_backfill_worker_artifacts(tmp_path, monkeypatch):
    from src.agents.experiment_agent.agents.prepare import planner as prepare_planner_module

    workspace = tmp_path / "workspace"
    project_dir = workspace / "project"
    workspace.mkdir()
    project_dir.mkdir()
    _write_idea_json(str(workspace / "idea.json"))

    paths = artifact_paths(str(workspace), str(project_dir))

    monkeypatch.setattr(
        prepare_planner_module,
        "ensure_experiment_dirs",
        lambda experiment_id: {
            "workspace_dir": str(workspace),
            "project_dir": str(project_dir),
            "repos_dir": str(workspace / "repos"),
            "dataset_dir": str(workspace / "dataset_candidate"),
            "model_dir": str(workspace / "model_candidate"),
            "results_dir": str(workspace / "results"),
            "reports_dir": str(workspace / "agent_reports"),
        },
    )
    prepare_plan_payload = {
        "stages": [
            {
                "stage_id": stage_id,
                "goal": f"{stage_id} goal",
                "input_paths": {},
                "repos_policy": "reference_or_copy",
                "project_must_be_self_contained": True,
                "research_required": True,
                "acquisition_required": True,
                "existing_local_hints": [],
                "done_condition": " ".join(
                    [
                        "Must use Xcientist artifact tools.",
                        "The artifact ledger is the proof.",
                        "agent_reports/prepare/artifacts/discovery.json",
                        "agent_reports/prepare/artifacts/repos.json",
                        "agent_reports/prepare/artifacts/dataset.json",
                        "agent_reports/prepare/artifacts/model.json",
                        "agent_reports/prepare/artifacts/env.json",
                        "agent_reports/prepare/artifacts/idea.md",
                        "agent_reports/prepare/artifacts/target_inventory.json",
                        "## Canonical Idea Components",
                        "target_inventory.json maps every idea.json component to concrete implementation targets.",
                    ]
                ),
                "artifact_ids": artifact_ids,
            }
            for stage_id, artifact_ids in [
                ("repos", ["prepare.discovery", "prepare.repos"]),
                ("dataset", ["prepare.dataset"]),
                ("model", ["prepare.model"]),
                ("env", ["prepare.env"]),
                ("synthesis", ["prepare.idea", "prepare.target_inventory"]),
            ]
        ],
        "summary": "plan",
        "usage_notes": "",
    }

    async def fake_run(self, *args, **kwargs):
        if kwargs.get("agent_name") == "prepare-planner":
            await _write_artifact_json(self.artifact_context, "prepare.plan", prepare_plan_payload)
            gate = (kwargs.get("extra_tool_metadata") or {}).get("xcientist_prefinish_gate")
            assert gate is not None
            gate_result = await gate({"assistant_text": json.dumps(prepare_plan_payload)})
            assert gate_result.blocked is False
        return {"output": prepare_plan_payload}

    async def fake_execute_step_with_prefinish_review(*args, **kwargs):
        return {
            "status": "FAIL",
            "failed_review_report": {
                "status": "FAIL",
                "evidence_summary": "missing synthesis artifacts",
                "blocking_issues": ["synthesis artifacts missing"],
                "required_followup": ["write prepare.idea and prepare.target_inventory with artifact tools"],
                "terminal_blocker": False,
                "next_worker_input": "write the missing managed artifacts and finish again",
            },
            "step_reports": [],
        }

    monkeypatch.setattr(PrepareAgent, "run", fake_run)
    monkeypatch.setattr(
        prepare_planner_module,
        "execute_step_with_prefinish_review",
        fake_execute_step_with_prefinish_review,
    )

    agent = PrepareAgent(verbose=False, workspace_root=str(workspace))
    report = asyncio.run(agent.prepare_workspace("demo"))

    assert report.idea_md_path == os.path.realpath(paths["idea"])
    assert os.path.exists(paths["prepare_reviewer"])
    assert not os.path.exists(paths["idea"])
    assert not os.path.exists(paths["prepare_target_inventory"])
