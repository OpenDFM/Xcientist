import asyncio
import importlib.util
import json
import os
import sys

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

HAS_EXPERIMENT_DEPS = (
    importlib.util.find_spec("omegaconf") is not None
    and importlib.util.find_spec("openhands") is not None
)

if HAS_EXPERIMENT_DEPS:
    from src.agents.experiment_agent.agents.science.planner import AblationScienceAgent
    from src.agents.experiment_agent.agents.master.entry import Decision, MasterAgent
    from src.agents.experiment_agent.agents.reporting.entry import AblationReportIntegratorAgent
    from src.agents.experiment_agent.agents.code import EXPERIMENT_CODE_PLANNER
    from src.agents.experiment_agent.agents.science import (
        EXPERIMENT_ABLATION_SCIENCE_PLANNER,
        EXPERIMENT_STANDARD_SCIENCE_PLANNER,
    )
    from src.agents.experiment_agent.runtime.ablation_results import (
        build_ablation_results_artifacts,
    )
    from src.agents.experiment_agent.runtime.manifests import artifact_paths, load_workspace_state
    from src.agents.experiment_agent.tools.bounded_io import _parent_env_export_chunks

pytestmark = pytest.mark.skipif(
    not HAS_EXPERIMENT_DEPS,
    reason="experiment-agent test dependencies are not installed in this environment",
)


def _write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _pass_report(scope: str, **extra):
    payload = {
        "status": "PASS",
        "scope": scope,
        "checked_artifacts": [],
        "findings": [],
        "required_fixes": [],
        "evidence_summary": "ok",
    }
    payload.update(extra)
    return payload


def _write_idea_json(path: str, components=None):
    payload = {
        "title": "demo",
        "components": components
        or [
            {
                "component": "component_a",
                "explanation": "demo explanation",
            }
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def test_master_gate_and_materializer_follow_current_ablation_contract(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(
        tmp_path / "idea.json",
        components=[
            {"component": "component_a", "description": "component A description"},
            {"component": "component_b", "explanation": "component B explanation"},
        ],
    )
    project_root = tmp_path / "project"
    project_root.mkdir()

    paths = artifact_paths(str(tmp_path))
    _write_json(paths["prepare_validator"], _pass_report("prepare"))
    _write_json(paths["code_validator"], _pass_report("code"))
    _write_json(paths["standard_science_validator"], _pass_report("standard_science"))
    _write_json(
        paths["ablation_science_validator"],
        _pass_report(
            "ablation_science",
            ablation_components={
                "component_a": {
                    "result": "positive",
                    "metric": "accuracy",
                    "value": "+0.10",
                    "confidence": 0.9,
                    "analysis": "evidence-backed",
                    "method_context": "ignored by runtime",
                    "follow_up_required": False,
                },
                "component_b": {
                    "result": "negative",
                    "metric": "accuracy",
                    "value": "-0.03",
                    "confidence": 0.82,
                    "analysis": "small drop without component",
                    "method_context": "ignored by runtime",
                    "follow_up_required": False,
                },
            },
            summary={
                "feasible": True,
                "confidence": 0.86,
                "key_findings": ["component_a matters", "component_b is complementary"],
            },
        ),
    )

    agent = MasterAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        verbose=False,
    )

    payload = agent._compute_gate_payload()
    assert payload["decision"] == Decision.CONVERGED
    assert agent._materialize_ablation_results() is True

    ablation_results = load_workspace_state(str(tmp_path))["ablation_results"]
    assert set(ablation_results.keys()) == {"components", "summary"}
    assert list(ablation_results["components"].keys()) == ["component_a", "component_b"]
    assert ablation_results["components"]["component_a"]["method_context"] == "component A description"
    assert ablation_results["components"]["component_b"]["method_context"] == "component B explanation"
    assert "metadata" not in ablation_results
    assert "evidence_paths" not in ablation_results
    assert set(ablation_results["summary"].keys()) == {"feasible", "confidence", "key_findings"}
    assert os.path.exists(paths["final_artifact_contract"])


def test_master_gate_uses_generic_phase_completion_not_just_status(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()

    paths = artifact_paths(str(tmp_path))
    _write_json(paths["prepare_validator"], _pass_report("prepare"))
    _write_json(paths["code_validator"], _pass_report("code"))
    _write_json(
        paths["standard_science_validator"],
        {
            "verdict": "PASS",
            "scope": "standard_science",
            "checked_artifacts": [],
            "findings": ["runs exist"],
            "required_fixes": [],
            "required_fixes_for_complete_evaluation": [
                {"priority": "high", "issue": "missing coverage", "fix": "run more evidence"}
            ],
            "evidence_summary": "usable but incomplete",
        },
    )

    agent = MasterAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        verbose=False,
    )

    payload = agent._compute_gate_payload()
    assert payload["decision"] == Decision.STANDARD_EXP_NEEDED
    assert payload["phase_completion_status"] == "partial"
    assert payload["ready_for_next_phase"] is False
    assert payload["blocking_issues"]
    assert "missing coverage" in payload["blocking_issues"][0]


def test_master_routes_back_to_code_when_project_is_not_self_contained(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "runner.py").write_text(
        "import sys\nsys.path.insert(0, '../repos/demo')\n",
        encoding="utf-8",
    )

    paths = artifact_paths(str(tmp_path))
    _write_json(paths["prepare_validator"], _pass_report("prepare"))

    agent = MasterAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        verbose=False,
    )

    payload = agent._compute_gate_payload()
    assert payload["decision"] == Decision.CODE_NEEDED
    assert payload["self_contained_project"] is False
    assert payload["blocking_issues"]
    assert os.path.exists(paths["self_contained_report"])


def test_terminal_parent_env_export_includes_external_variables(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_AGENT_TEST_PARENT_ENV", "visible_value")
    chunks = _parent_env_export_chunks()
    rendered = "\n".join(chunks)
    assert "EXPERIMENT_AGENT_TEST_PARENT_ENV" in rendered
    assert "visible_value" in rendered


def test_ablation_materialization_rejects_non_phase_result_or_partial_phase(tmp_path):
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = artifact_paths(str(tmp_path))
    _write_json(
        paths["ablation_science_validator"],
        {
            "status": "PASS",
            "phase_completion_status": "partial",
            "artifact_role": "smoke_check",
            "scope": "ablation_science",
            "checked_artifacts": [],
            "findings": [],
            "required_fixes": [],
            "evidence_summary": "smoke only",
            "summary": {"feasible": True, "confidence": 0.9, "key_findings": ["placeholder"]},
            "ablation_components": {
                "component_a": {
                    "result": "positive",
                    "metric": "accuracy",
                    "value": "+0.1",
                    "confidence": 0.9,
                    "analysis": "evidence-backed",
                    "method_context": "ignored by runtime",
                    "follow_up_required": False,
                }
            },
        },
    )

    result = build_ablation_results_artifacts(str(tmp_path), str(project_root))
    assert result["valid"] is False
    assert "phase_result" in result["blocker"] or "not marked complete" in result["blocker"]


def test_integrator_prefers_deterministic_materialization(tmp_path, monkeypatch):
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = artifact_paths(str(tmp_path))
    _write_json(paths["ablation_science_validator"], _pass_report(
        "ablation_science",
        ablation_components={
            "component_a": {
                "result": "positive",
                "metric": "accuracy",
                "value": "+0.1",
                "confidence": 0.9,
                "analysis": "evidence-backed",
                "method_context": "ignored by runtime",
            }
        },
        summary={"feasible": True, "confidence": 0.9, "key_findings": ["component_a matters"]},
    ))

    agent = AblationReportIntegratorAgent(
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        verbose=False,
    )

    async def fail_run(*args, **kwargs):
        raise AssertionError("LLM fallback should not run when deterministic integration succeeds")

    monkeypatch.setattr(agent, "run", fail_run)
    result = asyncio.run(agent.execute())

    assert result["valid"] is True
    assert result["mode"] == "deterministic"


def test_ablation_science_agent_no_longer_calls_integrator(tmp_path, monkeypatch):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()

    agent = AblationScienceAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        project_root=str(project_root),
        workspace_root=str(tmp_path),
        plan="run ablations",
        verbose=False,
    )

    async def fake_base_execute(self):
        return {"report": "ok", "status": "completed"}

    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.science.planner._BaseSciencePlanner.execute",
        fake_base_execute,
    )

    result = asyncio.run(agent.execute())

    assert result == {"report": "ok", "status": "completed"}


def test_master_runtime_deduplicates_adjacent_phase_dispatches(tmp_path, monkeypatch):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()

    agent = MasterAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        max_iterations=6,
        verbose=False,
    )

    gate_sequence = iter(
        [
            {"decision": Decision.CODE_NEEDED, "phase": "code", "reasons": ["code"], "evidence_files": []},
            {"decision": Decision.STANDARD_EXP_NEEDED, "phase": "standard_science", "reasons": ["standard"], "evidence_files": []},
            {"decision": Decision.STANDARD_EXP_NEEDED, "phase": "standard_science", "reasons": ["standard"], "evidence_files": []},
            {"decision": Decision.ABLATION_NEEDED, "phase": "ablation_science", "reasons": ["ablation"], "evidence_files": []},
            {"decision": Decision.ABLATION_NEEDED, "phase": "ablation_science", "reasons": ["ablation"], "evidence_files": []},
            {"decision": Decision.CONVERGED, "phase": "complete", "reasons": ["done"], "evidence_files": []},
        ]
    )
    calls = []

    def fake_gate():
        return next(gate_sequence)

    async def fake_run_planner_task(subagent_type: str, description: str, planner_prompt: str):
        calls.append(subagent_type)
        return {"status": "completed"}

    monkeypatch.setattr(agent, "_compute_gate_payload", fake_gate)
    monkeypatch.setattr(agent, "_run_planner_task", fake_run_planner_task)
    monkeypatch.setattr(agent, "_materialize_ablation_results", lambda: True)
    monkeypatch.setattr(agent, "_materialize_results_summary", lambda: str(tmp_path / "master_summary.md"))

    result = asyncio.run(agent.run_orchestration())

    assert result["converged"] is True
    assert result["decision"] == Decision.CONVERGED
    assert calls == [
        EXPERIMENT_CODE_PLANNER,
        EXPERIMENT_STANDARD_SCIENCE_PLANNER,
        EXPERIMENT_ABLATION_SCIENCE_PLANNER,
    ]
