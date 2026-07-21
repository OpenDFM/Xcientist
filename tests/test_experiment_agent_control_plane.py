import asyncio
import hashlib
import json
import os
import sys
from types import SimpleNamespace

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

from src.agents.experiment_agent import config as experiment_config
from src.agents.experiment_agent import main as experiment_main
from src.agents.experiment_agent.agents.base.agent import (
    BaseAgent,
)
from src.agents.experiment_agent.agents.master.entry import Decision, MasterAgent
from src.agents.experiment_agent.agents.finalization.entry import (
    run_finalization_agent,
)
from src.agents.experiment_agent.runtime.finalization_hooks import (
    run_final_science_prefinish_hooks,
)
from src.agents.experiment_agent.runtime.manifests import (
    artifact_paths,
    load_json_file,
    load_workspace_state,
)
from src.agents.experiment_agent.runtime.artifacts import ArtifactLedger
from src.agents.experiment_agent.runtime.openharness_runner import (
    OpenHarnessAgentRunner,
    _build_structured_output_prefinish_gate,
    _structured_output_fallback_payload,
    _structured_output_fallback_text_from_metadata,
    ensure_openharness_runtime_env,
)
from src.agents.experiment_agent.runtime.report_layout import artifact_rel, step_report_abs_paths
from src.pipeline.contracts import (
    EXPERIMENT_ABLATION_RESULTS_REL,
    EXPERIMENT_SYMBOLIC_MEMORY_RECEIPT_REL,
    validate_experiment_contract,
)
from src.pipeline.run_loop import _symbolic_memory_receipt_path
from openharness.api.client import ApiMessageCompleteEvent, ApiMessageRequest, ApiRetryEvent
from openharness.api.openai_client import OpenAICompatibleClient
from openharness.engine.messages import ConversationMessage


def _write_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _pass_report(scope: str, **extra):
    payload = {
        "status": "PASS",
        "evidence_summary": "ok",
        "terminal_blocker": False,
        "next_worker_input": "",
        "checked_artifacts": ["agent_reports/evidence.json"],
        "review_scope": ["contract", "code", "experiments", "logs", "artifacts", "evidence", "safety"],
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


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_ablation_payload(component_name: str = "component_a"):
    return {
        "components": {
            component_name: {
                "result": "positive",
                "metric": "acc",
                "value": "+0.1",
                "confidence": 0.9,
                "analysis": "component improved the metric",
                "method_context": "demo component",
            }
        },
        "summary": {
            "feasible": True,
            "confidence": 0.9,
            "key_findings": ["component mattered"],
        },
    }


def _write_symbolic_memory_receipt(workspace_root, ablation_path, *, status="PASS", blocker=""):
    memory_dir = workspace_root / "symbolic_memory"
    memory_file = memory_dir / "symbolic_memory.json"
    _write_json(
        str(memory_file),
        {
            "next_id": 1,
            "fidmap2mid": {"0": "sym_component_a"},
            "records": {
                "0": {
                    "id": "sym_component_a",
                    "component": "component_a",
                    "component_family": "component.demo",
                    "result": "positive",
                    "metric": "acc",
                    "value": "+0.1",
                    "analysis": "component improved the metric",
                    "method_context": "demo component",
                    "confidence": 0.9,
                }
            },
            "config": {"upsert_threshold": 0.82},
        },
    )
    receipt_path = workspace_root / EXPERIMENT_SYMBOLIC_MEMORY_RECEIPT_REL
    _write_json(
        str(receipt_path),
        {
            "status": status,
            "hook": "final_science_prefinish",
            "experiment_id": "demo",
            "ablation_results_path": str(ablation_path),
            "symbolic_memory_path": str(memory_dir),
            "symbolic_memory_file_path": str(memory_file),
            "component_count": 1,
            "records_created": 1 if status == "PASS" else 0,
            "record_ids": ["sym_component_a"] if status == "PASS" else [],
            "blocker": blocker,
        },
    )
    return receipt_path


def _write_converged_phase_reports(workspace_root, project_root):
    paths = artifact_paths(str(workspace_root), str(project_root))
    _write_json(paths["prepare_reviewer"], _pass_report("prepare"))
    _write_json(paths["code_reviewer"], _pass_report("code"))
    science_steps = [
        {
            "condition_id": "full_component_a",
            "enabled_components": ["component_a"],
            "disabled_components": [],
            "reference_condition_id": "",
            "command": "python project/run.py --output results/science/full_component_a",
            "output_dir": "results/science/full_component_a",
        },
        {
            "condition_id": "without_component_a",
            "enabled_components": [],
            "disabled_components": ["component_a"],
            "reference_condition_id": "full_component_a",
            "command": "python project/run.py --disable-component-a --output results/science/without_component_a",
            "output_dir": "results/science/without_component_a",
        },
    ]
    _write_json(
        paths["science_executable_plan"],
        {
            "stages": science_steps,
            "summary": "science planned",
            "usage_notes": "run full reference and component-disabled ablation",
        },
    )
    for step in science_steps:
        condition_id = step["condition_id"]
        output_dir = workspace_root / step["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_rel = f"results/science/{condition_id}/metrics.json"
        log_rel = f"results/science/{condition_id}/run.log"
        _write_json(str(workspace_root / metrics_rel), {"accuracy": 0.9})
        (workspace_root / log_rel).write_text("formal science run\n", encoding="utf-8")
        evidence_rel = artifact_rel("science", condition_id, "evidence.json")
        evidence_path = str(workspace_root / evidence_rel)
        _write_json(
            evidence_path,
            {
                "condition_id": condition_id,
                "enabled_components": step["enabled_components"],
                "disabled_components": step["disabled_components"],
                "reference_condition_id": step["reference_condition_id"],
                "run_level": "full",
                "command": step["command"],
                "returncode": 0,
                "output_dir": step["output_dir"],
                "raw_outputs": [metrics_rel],
                "logs": [log_rel],
                "metrics_files": [metrics_rel],
                "dataset_bindings": {"train": "dataset_candidate/train.json", "evaluation": ["dataset_candidate/eval.json"]},
                "model_bindings": {"backend": "none_required_for_test"},
                "duration_sec": 1.0,
            },
        )
        ArtifactLedger(str(workspace_root)).append(
            {
                "event": "write_artifact",
                "artifact_id": f"science.{condition_id}.evidence",
                "path": evidence_path,
                "kind": "json",
                "sha256": _sha256_file(evidence_path),
                "schema_name": "science_evidence",
                "schema_issues": [],
                "stage": "science",
                "step_id": condition_id,
                "attempt": 1,
                "agent": "science_worker",
            }
        )
        step_paths = step_report_abs_paths(str(workspace_root), "science", condition_id)
        worker_payload = {
            "summary": "science condition completed",
            "artifact_ids_touched": [f"science.{condition_id}.evidence"],
            "remaining_blockers": [],
        }
        review_payload = _pass_report(
            "science",
            checked_artifacts=[evidence_rel, metrics_rel, log_rel],
            condition_id=condition_id,
            enabled_components=step["enabled_components"],
            disabled_components=step["disabled_components"],
            reference_condition_id=step["reference_condition_id"],
            result="positive" if step["disabled_components"] else "",
            metric="accuracy" if step["disabled_components"] else "",
            value="+0.10" if step["disabled_components"] else "",
            confidence=0.9 if step["disabled_components"] else 0.0,
            analysis="evidence-backed" if step["disabled_components"] else "",
            method_context="component_a disabled" if step["disabled_components"] else "",
            follow_up_required=False,
            prefinish_contract={"status": "PASS", "hook": "step_prefinish_contract", "issues": []},
            review_matrix={"status": "PASS", "reports": [], "structured_findings": {"reviewer_statuses": {"aggregate": "PASS"}}},
        )
        hook_payload = {
            "scope": "science",
            "step_id": condition_id,
            "attempt": 1,
            "status": "PASS",
            "review_status": "PASS",
            "worker_report_path": step_paths["worker_report_path"],
            "review_report_path": step_paths["review_report_path"],
            "returned_to_worker": False,
            "prefinish_contract": {"status": "PASS", "hook": "step_prefinish_contract", "issues": []},
            "review_matrix": {"status": "PASS", "reports": [], "structured_findings": {"reviewer_statuses": {"aggregate": "PASS"}}},
        }
        _write_json(step_paths["worker_report_path"], worker_payload)
        _write_json(step_paths["review_report_path"], review_payload)
        _write_json(step_paths["hook_report_path"], hook_payload)
    _write_json(
        paths["science_reviewer"],
        _pass_report(
            "science",
            science_component_results={
                "component_a": {
                    "result": "positive",
                    "metric": "accuracy",
                    "value": "+0.10",
                    "confidence": 0.9,
                    "analysis": "evidence-backed",
                    "method_context": "ignored by deterministic materializer",
                    "follow_up_required": False,
                    "condition_id": "without_component_a",
                    "reference_condition_id": "full_component_a",
                }
            },
            summary={
                "feasible": True,
                "confidence": 0.86,
                "key_findings": ["component_a matters"],
            },
        ),
    )
    return paths


def test_experiment_config_defaults_to_openharness_backend():
    assert experiment_config.get_backend_name() == "openharness"
    harness_cfg = experiment_config.get_openharness_config()
    execution_cfg = experiment_config.get_execution_config()
    assert harness_cfg["default_model"]
    assert harness_cfg["role_models"]["planner"]
    assert harness_cfg["role_models"]["worker"]
    assert harness_cfg["role_models"]["reviewer"]
    assert harness_cfg["runtime_dir_name"] == ".openharness_runtime"
    assert harness_cfg["max_turns"] == 0
    assert execution_cfg["planner_max_turns"] == 0
    assert execution_cfg["worker_max_turns"] == 0
    ok, errors = experiment_config.validate_config()
    assert ok, errors


def test_experiment_workspace_ignores_legacy_codeagent_env(tmp_path, monkeypatch):
    root = tmp_path / "workspace_root"
    legacy = tmp_path / "legacy_codeagent"
    monkeypatch.delenv("EXPERIMENT_AGENT_WORKSPACE_DIR", raising=False)
    monkeypatch.setenv("CODEAGENT_WORKSPACES_DIR", str(legacy))
    monkeypatch.setattr(
        experiment_config,
        "get_workspace_config",
        lambda: {
            "root": str(root),
            "prepare_clone_depth": 1,
            "model_candidate_seed": "",
            "tavily_enabled": True,
            "tavily_api_key": "",
            "tavily_remote_url_template": "https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}",
        },
    )

    assert experiment_config.get_workspace_dir("demo") == os.path.realpath(root / "demo")


def test_ensure_openharness_runtime_env_is_workspace_local(tmp_path):
    workspace = tmp_path / "workspace"
    updates = ensure_openharness_runtime_env(str(workspace))
    assert updates["OPENHARNESS_CONFIG_DIR"].endswith(".openharness_runtime/config")
    assert updates["OPENHARNESS_DATA_DIR"].endswith(".openharness_runtime/data")
    assert updates["OPENHARNESS_LOGS_DIR"].endswith(".openharness_runtime/logs")
    for path in updates.values():
        assert os.path.isdir(path)
        assert str(workspace) in path


def test_openharness_runner_worker_and_reviewer_tools_are_separated(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiment_config,
        "get_openharness_config",
        lambda: {
            "api_key": "test-key",
            "base_url": "",
            "default_model": "gpt-test",
            "role_models": {},
            "timeout_seconds": 30,
            "max_tokens": 1024,
            "max_turns": 4,
            "context_window_tokens": None,
            "auto_compact_threshold_tokens": None,
            "runtime_dir_name": ".openharness_runtime",
            "max_budget_usd": 0,
        },
    )
    worker = OpenHarnessAgentRunner(model="gpt-test", workspace_root=str(tmp_path), verbose=False)
    reviewer = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
        reviewer_mode=True,
    )

    worker_tools = {tool.name for tool in worker._build_registry().list_tools()}
    reviewer_tools = {tool.name for tool in reviewer._build_registry().list_tools()}
    assert {"bash", "write_file", "edit_file"}.issubset(worker_tools)
    assert "read_file" in reviewer_tools
    assert not {"bash", "write_file", "edit_file"} & reviewer_tools

    worker_settings = worker._build_settings()
    reviewer_settings = reviewer._build_settings()
    assert worker_settings.permission.mode.value == "full_auto"
    assert reviewer_settings.permission.mode.value == "plan"
    assert set(reviewer_settings.permission.denied_tools) == {"bash", "write_file", "edit_file"}
    assert worker_settings.memory.enabled is False
    assert reviewer_settings.allow_project_plugins is False


def test_openharness_runner_treats_zero_max_turns_as_unlimited(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiment_config,
        "get_openharness_config",
        lambda: {
            "api_key": "test-key",
            "base_url": "",
            "default_model": "gpt-test",
            "role_models": {},
            "timeout_seconds": 30,
            "max_tokens": 1024,
            "max_turns": 0,
            "context_window_tokens": None,
            "auto_compact_threshold_tokens": None,
            "runtime_dir_name": ".openharness_runtime",
            "max_budget_usd": 0,
        },
    )

    runner = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
    )
    settings = runner._build_settings()

    assert runner.max_turns is None
    assert settings.max_turns == 0


def test_openharness_runner_registers_mcp_tools_only_for_enabled_workers(tmp_path, monkeypatch):
    from openharness.mcp.types import McpToolInfo

    class FakeMcpManager:
        def list_tools(self):
            return [
                McpToolInfo(
                    server_name="tavily",
                    name="search",
                    description="search",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                )
            ]

        def list_resources(self):
            return []

        async def call_tool(self, server_name, tool_name, arguments):
            return "ok"

        async def read_resource(self, server_name, uri):
            return "resource"

    monkeypatch.setattr(
        experiment_config,
        "get_openharness_config",
        lambda: {
            "api_key": "test-key",
            "base_url": "",
            "default_model": "gpt-test",
            "role_models": {},
            "timeout_seconds": 30,
            "max_tokens": 1024,
            "max_turns": 4,
            "context_window_tokens": None,
            "auto_compact_threshold_tokens": None,
            "runtime_dir_name": ".openharness_runtime",
            "max_budget_usd": 0,
        },
    )
    worker = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
        enable_mcp=True,
    )
    reviewer = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
        reviewer_mode=True,
        enable_mcp=True,
    )

    worker_tools = {tool.name for tool in worker._build_registry(FakeMcpManager()).list_tools()}
    reviewer_tools = {tool.name for tool in reviewer._build_registry(FakeMcpManager()).list_tools()}
    assert "mcp__tavily__search" in worker_tools
    assert "list_mcp_resources" in worker_tools
    assert "mcp__tavily__search" not in reviewer_tools


def test_openharness_runner_records_mcp_unavailable_status(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(
        experiment_config,
        "get_openharness_config",
        lambda: {
            "api_key": "test-key",
            "base_url": "",
            "default_model": "gpt-test",
            "role_models": {},
            "timeout_seconds": 30,
            "max_tokens": 1024,
            "max_turns": 4,
            "context_window_tokens": None,
            "auto_compact_threshold_tokens": None,
            "runtime_dir_name": ".openharness_runtime",
            "max_budget_usd": 0,
        },
    )
    monkeypatch.setattr(
        experiment_config,
        "get_workspace_config",
        lambda: {
            "root": str(tmp_path),
            "prepare_clone_depth": 1,
            "model_candidate_seed": "",
            "tavily_enabled": True,
            "tavily_api_key": "",
            "tavily_remote_url_template": "https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}",
        },
    )
    runner = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
        enable_mcp=True,
    )

    manager = asyncio.run(runner._build_mcp_manager())

    assert manager is None
    status = load_json_file(str(tmp_path / "agent_reports" / "_runtime" / "mcp_status.json"))
    assert status["enabled"] is True
    assert status["connected"] is False
    assert status["status"] == "unavailable"
    assert "Tavily" in status["reason"]


def test_openharness_runner_retries_empty_structured_response(tmp_path, monkeypatch):
    calls = {"count": 0}

    async def fake_run_text(self, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("Model returned an empty assistant message.")
        return '{"status": "PASS"}'

    monkeypatch.setattr(OpenHarnessAgentRunner, "run_text", fake_run_text)
    runner = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
    )
    payload = asyncio.run(
        runner.run_json(
            system_prompt="",
            user_prompt="return json",
            output_schema={"type": "object"},
        )
    )

    assert payload == {"status": "PASS"}
    assert calls["count"] == 2


def test_openharness_runner_extracts_single_structured_response_from_prose(tmp_path, monkeypatch):
    calls = {"count": 0}

    async def fake_run_text(self, **kwargs):
        calls["count"] += 1
        return 'Here is the JSON: {"status": "PASS"}'

    monkeypatch.setattr(OpenHarnessAgentRunner, "run_text", fake_run_text)
    runner = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
    )

    payload = asyncio.run(
        runner.run_json(
            system_prompt="",
            user_prompt="return json",
            output_schema={"type": "object"},
        )
    )

    assert payload == {"status": "PASS"}
    assert calls["count"] == 1


def test_openharness_runner_rejects_multiple_structured_response_candidates(tmp_path, monkeypatch):
    async def fake_run_text(self, **kwargs):
        del self, kwargs
        return 'First {"status": "PASS"} second {"status": "FAIL"}'

    monkeypatch.setattr(OpenHarnessAgentRunner, "run_text", fake_run_text)
    runner = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
    )

    try:
        asyncio.run(
            runner.run_json(
                system_prompt="",
                user_prompt="return json",
                output_schema={"type": "object"},
            )
        )
    except ValueError as exc:
        assert "exactly one unambiguous JSON object" in str(exc)
    else:
        raise AssertionError("run_json must reject multiple JSON object candidates")


def test_structured_output_prefinish_gate_returns_schema_and_reviewer_id_feedback():
    gate = _build_structured_output_prefinish_gate(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reviewer_id": {"type": "string"},
                "status": {"type": "string", "enum": ["PASS", "FAIL"]},
            },
            "required": ["reviewer_id", "status"],
        },
        expected_reviewer_id="idea_alignment",
    )

    result = asyncio.run(
        gate({"assistant_text": '{"reviewer_id":"wrong","status":"MAYBE","extra":true}'})
    )

    assert result.blocked is True
    assert "Return JSON shaped like this" in result.reason
    assert "Expected structured output schema" in result.reason
    assert "$ contains unsupported fields: extra" in result.reason
    assert "$.status must be one of" in result.reason
    assert "$.reviewer_id must be exactly 'idea_alignment'" in result.reason


def test_structured_output_reviewer_fallback_is_schema_valid_fail_report():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reviewer_id": {"type": "string"},
            "reviewer_kind": {"type": "string", "enum": ["deterministic", "agent"]},
            "status": {"type": "string", "enum": ["PASS", "FAIL"]},
            "blocking": {"type": "boolean"},
            "summary": {"type": "string"},
            "checked_artifacts": {"type": "array", "items": {"type": "string"}},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "code": {"type": "string"},
                        "message": {"type": "string"},
                        "required_fix": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["code", "message", "required_fix", "evidence"],
                },
            },
            "structured_findings": {"type": "object"},
        },
        "required": [
            "reviewer_id",
            "reviewer_kind",
            "status",
            "blocking",
            "summary",
            "checked_artifacts",
            "issues",
            "structured_findings",
        ],
    }

    fallback = _structured_output_fallback_payload(
        schema,
        expected_reviewer_id="statistical_interpretation",
        reason="too many hook repairs",
    )
    text_before_limit = _structured_output_fallback_text_from_metadata(
        {
            "xcientist_structured_output_max_hook_blocks": 3,
            "xcientist_structured_output_fallback_json": fallback,
        },
        block_count=2,
    )
    text_at_limit = _structured_output_fallback_text_from_metadata(
        {
            "xcientist_structured_output_max_hook_blocks": 3,
            "xcientist_structured_output_fallback_json": fallback,
        },
        block_count=3,
    )

    assert text_before_limit is None
    assert text_at_limit is not None
    parsed = json.loads(text_at_limit)
    assert parsed["reviewer_id"] == "statistical_interpretation"
    assert parsed["reviewer_kind"] == "agent"
    assert parsed["status"] == "FAIL"
    assert parsed["blocking"] is True
    assert parsed["issues"][0]["code"] == "reviewer_output_schema_invalid"
    assert "prefinish reviewers" in parsed["issues"][0]["required_fix"]


def test_run_json_installs_structured_output_prefinish_gate(tmp_path, monkeypatch):
    seen_metadata = []

    async def fake_run_text(self, **kwargs):
        del kwargs
        seen_metadata.append(dict(self.extra_tool_metadata))
        gate = self.extra_tool_metadata["xcientist_prefinish_gate"]
        blocked = await gate({"assistant_text": '{"reviewer_id":"wrong","status":"PASS"}'})
        assert blocked.blocked is True
        assert "$.reviewer_id must be exactly 'idea_alignment'" in blocked.reason
        return '{"reviewer_id":"idea_alignment","status":"PASS"}'

    monkeypatch.setattr(OpenHarnessAgentRunner, "run_text", fake_run_text)
    runner = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
        extra_tool_metadata={"xcientist_expected_reviewer_id": "idea_alignment"},
    )

    payload = asyncio.run(
        runner.run_json(
            system_prompt="",
            user_prompt="return review json",
            output_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reviewer_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["PASS", "FAIL"]},
                },
                "required": ["reviewer_id", "status"],
            },
        )
    )

    assert payload == {"reviewer_id": "idea_alignment", "status": "PASS"}
    assert seen_metadata
    assert "xcientist_prefinish_gate" in seen_metadata[0]
    assert seen_metadata[0]["xcientist_structured_output_max_hook_blocks"] == 6
    assert (
        seen_metadata[0]["xcientist_structured_output_fallback_json"]["reviewer_id"]
        == "idea_alignment"
    )
    assert runner.extra_tool_metadata == {"xcientist_expected_reviewer_id": "idea_alignment"}


def test_openai_client_retries_xunfei_transient_provider_errors(monkeypatch):
    from openharness.api import openai_client as client_mod

    calls = {"count": 0}
    sleeps = []

    class FakeStream:
        def __init__(self):
            self._sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._sent:
                raise StopAsyncIteration
            self._sent = True
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content='{"status": "PASS"}',
                            reasoning_content="",
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
            )

    class FakeCompletions:
        async def create(self, **_params):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError(
                    "Xunfei request failed code: 10010, "
                    "msg: RecvFromEngineError:Engine Busy"
                )
            if calls["count"] == 2:
                raise RuntimeError(
                    "xunfei response error code: 11210, "
                    "msg: NotEnoughCvError"
                )
            return FakeStream()

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

        async def close(self):
            return None

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def collect_events():
        client = OpenAICompatibleClient(
            "test-key",
            max_retries=3,
            retry_base_delay=0.01,
            retry_max_delay=0.01,
        )
        request = ApiMessageRequest(
            model="astron-code-latest",
            messages=[ConversationMessage.from_user_text("hello")],
            max_tokens=32,
        )
        return [event async for event in client.stream_message(request)]

    monkeypatch.setattr(client_mod, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(client_mod.asyncio, "sleep", fake_sleep)

    events = asyncio.run(collect_events())

    assert calls["count"] == 3
    assert sleeps == [0.01, 0.01]
    assert isinstance(events[0], ApiRetryEvent)
    assert isinstance(events[1], ApiRetryEvent)
    assert isinstance(events[-1], ApiMessageCompleteEvent)
    assert events[-1].message.text == '{"status": "PASS"}'


def test_openharness_runner_prints_model_heartbeat(tmp_path, monkeypatch, capsys):
    from src.agents.experiment_agent.runtime import openharness_runner as runner_mod

    class FakeAssistantTextDelta:
        def __init__(self, text):
            self.text = text

    class FakeMessage:
        text = '{"status": "PASS"}'

    class FakeUsage:
        input_tokens = 3
        output_tokens = 4
        total_tokens = 7

    class FakeAssistantTurnComplete:
        def __init__(self):
            self.message = FakeMessage()
            self.usage = FakeUsage()

    class FakeEngine:
        async def submit_message(self, _prompt):
            await asyncio.sleep(0.12)
            yield FakeAssistantTextDelta('{"status": "PASS"}')
            yield FakeAssistantTurnComplete()

    monkeypatch.setenv("XCIENTIST_OPENHARNESS_HEARTBEAT_SECONDS", "0.05")
    monkeypatch.setattr(
        OpenHarnessAgentRunner,
        "_build_engine",
        lambda self, **_kwargs: FakeEngine(),
    )
    monkeypatch.setattr(
        runner_mod,
        "_load_openharness_symbols",
        lambda: {
            "AssistantTextDelta": FakeAssistantTextDelta,
            "AssistantTurnComplete": FakeAssistantTurnComplete,
            "ErrorEvent": type("FakeErrorEvent", (), {}),
            "StatusEvent": type("FakeStatusEvent", (), {}),
            "ToolExecutionStarted": type("FakeToolExecutionStarted", (), {}),
            "ToolExecutionCompleted": type("FakeToolExecutionCompleted", (), {}),
        },
    )

    runner = OpenHarnessAgentRunner(
        model="gpt-test",
        workspace_root=str(tmp_path),
        verbose=False,
    )
    payload = asyncio.run(
        runner.run_json(
            system_prompt="",
            user_prompt="return json",
            output_schema={"type": "object"},
            agent_name="demo_agent",
        )
    )

    output = capsys.readouterr().out
    assert payload == {"status": "PASS"}
    assert "model" in output
    assert "start" in output
    assert "waiting" in output
    assert "done" in output
    assert "demo agent" in output


def test_base_agent_routes_prefinish_review_to_read_only_runner(tmp_path, monkeypatch):
    calls = []

    async def fake_run_json(self, **kwargs):
        calls.append({"reviewer_mode": self.reviewer_mode, **kwargs})
        return {
            "status": "PASS",
            "evidence_summary": "ok",
            "terminal_blocker": False,
            "next_worker_input": "",
            "checked_artifacts": ["a"],
            "review_scope": ["contract"],
        }

    monkeypatch.setattr(OpenHarnessAgentRunner, "run_json", fake_run_json)

    agent = BaseAgent(agent_type="demo", model="gpt-test", workspace_root=str(tmp_path), verbose=False)
    result = asyncio.run(
        agent.run(
            user_prompt="review",
            output_schema={"type": "object"},
            purpose="prefinish_review",
        )
    )
    assert result["output"]["status"] == "PASS"
    assert calls[0]["reviewer_mode"] is True


def test_ensure_experiment_dirs_creates_openharness_runtime(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    path_payload = {
        "workspace_dir": str(workspace_root),
        "project_dir": str(workspace_root / "project"),
        "logs_dir": str(workspace_root / "logs"),
        "cache_dir": str(workspace_root / ".cache"),
        "dataset_dir": str(workspace_root / "dataset_candidate"),
        "model_dir": str(workspace_root / "model_candidate"),
        "results_dir": str(workspace_root / "results"),
        "reports_dir": str(workspace_root / "agent_reports"),
        "repos_dir": str(workspace_root / "repos"),
        "model_candidate_seed": "",
    }
    monkeypatch.setattr(experiment_config, "get_path_config", lambda _experiment_id: path_payload)

    paths = experiment_config.ensure_experiment_dirs("demo")
    assert os.path.isdir(workspace_root / ".openharness_runtime" / "config")
    assert os.path.isdir(workspace_root / "project")
    assert os.path.isdir(workspace_root / "results" / "science")
    assert paths["model_share_dir"].endswith("model_candidate/model_share")


def test_master_gate_converges_without_materializing_final_contract(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = _write_converged_phase_reports(tmp_path, project_root)

    agent = MasterAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        verbose=False,
    )

    payload = agent._compute_gate_payload()
    assert payload["decision"] == Decision.CONVERGED
    state = load_workspace_state(str(tmp_path))
    assert state["ablation_results"] is None
    assert not os.path.exists(paths["ablation_results_manifest"])


def test_master_gate_uses_reviewer_phase_completion_not_just_status(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()

    paths = artifact_paths(str(tmp_path), str(project_root))
    _write_json(paths["prepare_reviewer"], _pass_report("prepare"))
    _write_json(paths["code_reviewer"], _pass_report("code"))
    _write_json(
        paths["science_reviewer"],
        _pass_report(
            "science",
            blocking_issues=["missing coverage"],
            phase_completion_status="partial",
            ready_for_next_phase=False,
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
    assert payload["decision"] == Decision.RUN_SCIENCE
    assert payload["phase_completion_status"] == "partial"
    assert payload["ready_for_next_phase"] is False


def test_master_runs_code_then_science_without_decision_loop(tmp_path, monkeypatch):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_root))
    _write_json(paths["prepare_reviewer"], _pass_report("prepare"))

    calls = []

    async def fake_code_agent(**kwargs):
        calls.append("code")
        _write_json(paths["code_reviewer"], _pass_report("code"))
        return {"status": "completed"}

    async def fake_science_agent(**kwargs):
        calls.append("science")
        _write_json(paths["science_reviewer"], _pass_report("science"))
        return {"status": "completed"}

    monkeypatch.setattr("src.agents.experiment_agent.agents.master.entry.run_code_agent", fake_code_agent)
    monkeypatch.setattr("src.agents.experiment_agent.agents.master.entry.run_science_agent", fake_science_agent)

    agent = MasterAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        verbose=False,
    )

    result = asyncio.run(agent.run_orchestration())

    assert calls == ["code", "science"]
    assert result["converged"] is True
    assert result["stopped_due_to_iteration_limit"] is False
    assert result["iterations"] == 2
    assert os.path.exists(paths["iteration_status"])


def test_master_returns_blocked_payload_when_prepare_gate_missing(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = artifact_paths(str(tmp_path), str(project_root))

    agent = MasterAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        verbose=False,
    )

    result = asyncio.run(agent.run_orchestration())

    assert result["blocked"] is True
    assert result["decision"] == "PREPARE_BLOCKED"
    decision = load_json_file(paths["master_decision"])
    assert decision["decision"] == "PREPARE_BLOCKED"
    assert load_json_file(paths["runtime_phase_state"])["active_phase"] == "prepare"


def test_final_prefinish_hook_writes_ablation_results_and_symbolic_memory_receipt(tmp_path, monkeypatch):
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = _write_converged_phase_reports(tmp_path, project_root)

    def fake_convert(**kwargs):
        assert kwargs["ablation_path"] == paths["ablation_results"]
        memory_file = os.path.join(kwargs["symbolic_memory_path"], "symbolic_memory.json")
        _write_json(
            memory_file,
            {
                "next_id": 1,
                "fidmap2mid": {"0": "component_a"},
                "records": {
                    "0": {
                        "id": "component_a",
                        "component": "component_a",
                        "component_family": "component.demo",
                        "result": "positive",
                        "metric": "accuracy",
                        "value": "+0.10",
                        "analysis": "evidence-backed",
                        "method_context": "demo component",
                        "confidence": 0.9,
                    }
                },
                "config": {"upsert_threshold": 0.82},
            },
        )
        return [{"id": "component_a", "component": "component_a"}]

    monkeypatch.setattr(
        "src.pipeline.experiment_to_symbolic.convert_ablation_to_symbolic_memory",
        fake_convert,
    )
    memory_path = tmp_path / "symbolic"
    monkeypatch.setenv("XCIENTIST_SYMBOLIC_MEMORY_PATH", str(memory_path))

    receipt = run_final_science_prefinish_hooks(
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        experiment_id="demo",
    )

    assert receipt["status"] == "PASS"
    assert receipt["records_created"] == 1
    assert receipt["ablation_results_path"] == paths["ablation_results"]
    assert receipt["symbolic_memory_path"] == str(memory_path)
    assert os.path.exists(paths["ablation_results"])
    assert load_json_file(paths["symbolic_memory_receipt"])["status"] == "PASS"
    assert load_json_file(paths["ablation_materialization_report"])["generated_by"] == "prefinish_hook"


def test_final_prefinish_hook_real_symbolic_writeback_is_contract_aligned(tmp_path, monkeypatch):
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = _write_converged_phase_reports(tmp_path, project_root)
    memory_path = tmp_path / "symbolic"
    monkeypatch.setenv("XCIENTIST_SYMBOLIC_MEMORY_PATH", str(memory_path))

    receipt = run_final_science_prefinish_hooks(
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        experiment_id="demo",
    )

    memory_file = memory_path / "symbolic_memory.json"
    assert receipt["status"] == "PASS"
    assert receipt["symbolic_memory_file_path"] == str(memory_file)
    assert receipt["component_count"] == 1
    assert receipt["records_created"] == 1
    assert memory_file.exists()
    memory_payload = load_json_file(str(memory_file))
    assert {
        record["component"]
        for record in memory_payload["records"].values()
        if isinstance(record, dict)
    } == {"component_a"}
    report = validate_experiment_contract(tmp_path, {})
    assert report.valid is True
    assert report.artifacts["ablation_results"] == paths["ablation_results"]


def test_final_prefinish_hook_preserves_final_artifacts_when_symbolic_memory_fails(tmp_path, monkeypatch):
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = _write_converged_phase_reports(tmp_path, project_root)

    def fail_convert(**kwargs):
        assert kwargs["ablation_path"] == paths["ablation_results"]
        raise RuntimeError("symbolic memory backend unavailable")

    monkeypatch.setattr(
        "src.pipeline.experiment_to_symbolic.convert_ablation_to_symbolic_memory",
        fail_convert,
    )
    monkeypatch.setenv("XCIENTIST_SYMBOLIC_MEMORY_PATH", str(tmp_path / "symbolic"))

    receipt = run_final_science_prefinish_hooks(
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        experiment_id="demo",
    )

    assert receipt["status"] == "FAIL"
    assert receipt["ablation_results_path"] == paths["ablation_results"]
    assert receipt["symbolic_memory_path"] == str(tmp_path / "symbolic")
    assert "symbolic memory backend unavailable" in receipt["blocker"]
    assert "pipeline.default_macro_roles" in receipt["repair_instructions"]
    assert os.path.exists(paths["ablation_results"])
    assert os.path.exists(paths["ablation_materialization_report"])
    assert os.path.exists(paths["ablation_results_manifest"])


def test_final_prefinish_hook_rejects_science_phase_without_step_lineage(tmp_path):
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = _write_converged_phase_reports(tmp_path, project_root)
    missing_hook = step_report_abs_paths(
        str(tmp_path),
        "science",
        "without_component_a",
    )["hook_report_path"]
    os.remove(missing_hook)

    receipt = run_final_science_prefinish_hooks(
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        experiment_id="demo",
    )

    assert receipt["status"] == "FAIL"
    assert receipt["blocker"] == "science lineage verification failed"
    assert "without_component_a" in receipt["repair_instructions"]
    assert receipt["science_lineage"]["status"] == "FAIL"
    assert not os.path.exists(paths["ablation_results"])


def test_finalization_agent_runs_final_hook_as_openharness_prefinish_gate(tmp_path, monkeypatch):
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = _write_converged_phase_reports(tmp_path, project_root)
    calls = {"final_hook": 0, "gate": 0}

    def fake_final_hook(**kwargs):
        calls["final_hook"] += 1
        assert kwargs["workspace_root"] == str(tmp_path)
        receipt = {
            "status": "PASS",
            "hook": "final_science_prefinish",
            "experiment_id": kwargs["experiment_id"],
            "ablation_results_path": paths["ablation_results"],
            "symbolic_memory_path": str(tmp_path / "symbolic"),
            "records_created": 1,
            "record_ids": ["component_a"],
            "blocker": "",
        }
        _write_json(paths["symbolic_memory_receipt"], receipt)
        return receipt

    async def fake_run_text(self, **kwargs):
        gate = self.extra_tool_metadata.get("xcientist_prefinish_gate")
        assert callable(gate)
        assert self.artifact_context["stage"] == "finalization"
        assert any(
            spec["artifact_id"] == "final.ablation_results" and spec["writer"] == "runtime"
            for spec in self.artifact_context["artifact_specs"]
        )
        calls["gate"] += 1
        hook_result = await gate(
            {
                "assistant_text": json.dumps(
                    {
                        "status": "READY_FOR_FINALIZATION",
                        "checked_inputs": ["idea.json", "agent_reports/science/phase.json"],
                        "repair_summary": "none",
                        "notes": "ready",
                    }
                )
            }
        )
        assert hook_result.blocked is False
        return json.dumps(
            {
                "status": "READY_FOR_FINALIZATION",
                "checked_inputs": ["idea.json", "agent_reports/science/phase.json"],
                "repair_summary": "none",
                "notes": "ready",
            }
        )

    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.finalization.entry.run_final_science_prefinish_hooks",
        fake_final_hook,
    )
    monkeypatch.setattr(OpenHarnessAgentRunner, "run_text", fake_run_text)

    receipt = asyncio.run(
        run_finalization_agent(
            experiment_id="demo",
            workspace_root=str(tmp_path),
            project_root=str(project_root),
            verbose=False,
        )
    )

    assert receipt["status"] == "PASS"
    assert receipt["ablation_results_path"] == paths["ablation_results"]
    assert calls == {"final_hook": 1, "gate": 1}


def test_main_reuses_valid_prepare_and_runs_finalization_prefinish_agent_on_convergence(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    project_root = workspace_root / "project"
    project_root.mkdir(parents=True)
    paths = artifact_paths(str(workspace_root), str(project_root))
    _write_json(paths["prepare_reviewer"], _pass_report("prepare"))

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("prepare should be reused, not rerun")

    async def fake_master(**kwargs):
        return {
            "iterations": 1,
            "final_path": "master.md",
            "converged": True,
            "stopped_due_to_iteration_limit": False,
        }

    async def fake_finalization_agent(**kwargs):
        assert kwargs["workspace_root"] == str(workspace_root)
        return {
            "status": "PASS",
            "ablation_results_path": paths["ablation_results"],
        }

    monkeypatch.setattr(experiment_main, "print_config", lambda: None)
    monkeypatch.setattr(experiment_main, "print_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment_main,
        "ensure_experiment_dirs",
        lambda experiment_id: {
            "workspace_dir": str(workspace_root),
            "project_dir": str(project_root),
            "results_dir": str(workspace_root / "results"),
            "model_dir": str(workspace_root / "model_candidate"),
            "reports_dir": str(workspace_root / "agent_reports"),
            "cache_dir": str(workspace_root / ".cache"),
        },
    )
    monkeypatch.setattr(experiment_main, "copy_prepared_data_to_workspace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_main, "write_workspace_env_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_main, "run_prepare", fail_prepare)
    monkeypatch.setattr(experiment_main, "run_master", fake_master)
    monkeypatch.setattr(experiment_main, "run_finalization_agent", fake_finalization_agent)
    monkeypatch.setattr(experiment_main, "get_idea_input_path", lambda experiment_id: str(workspace_root / "idea.md"))

    result = asyncio.run(
        experiment_main.run_experiment_once(
            experiment_id="demo",
            workspace_root=str(workspace_root),
            verbose=False,
        )
    )

    assert result["ok"] is True
    assert result["converged"] is True
    assert result["ablation_results_path"].endswith("agent_reports/ablation/final/ablation_results.json")


def test_main_restores_runtime_config_env_after_default_run(tmp_path, monkeypatch):
    monkeypatch.delenv("XCIENTIST_CONFIG", raising=False)
    workspace_root = tmp_path / "workspace"
    project_root = workspace_root / "project"
    project_root.mkdir(parents=True)
    paths = artifact_paths(str(workspace_root), str(project_root))
    _write_json(paths["prepare_reviewer"], _pass_report("prepare"))

    monkeypatch.setattr(experiment_main, "print_config", lambda: None)
    monkeypatch.setattr(experiment_main, "print_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment_main, "print_kv_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment_main,
        "ensure_experiment_dirs",
        lambda experiment_id: {
            "workspace_dir": str(workspace_root),
            "project_dir": str(project_root),
            "results_dir": str(workspace_root / "results"),
            "model_dir": str(workspace_root / "model_candidate"),
            "reports_dir": str(workspace_root / "agent_reports"),
            "cache_dir": str(workspace_root / ".cache"),
        },
    )
    monkeypatch.setattr(experiment_main, "copy_prepared_data_to_workspace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_main, "write_workspace_env_file", lambda *_args, **_kwargs: None)

    result = asyncio.run(
        experiment_main.run_experiment_once(
            experiment_id="demo",
            workspace_root=str(workspace_root),
            prepare_only=True,
            verbose=False,
        )
    )

    assert result["ok"] is True
    assert "XCIENTIST_CONFIG" not in os.environ


def test_pipeline_experiment_contract_uses_ablation_final_path(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _write_idea_json(str(workspace_root / "idea.json"))
    ablation_path = workspace_root / EXPERIMENT_ABLATION_RESULTS_REL
    _write_json(str(ablation_path), _valid_ablation_payload())
    _write_symbolic_memory_receipt(workspace_root, ablation_path)

    report = validate_experiment_contract(workspace_root, {})

    assert report.valid is True
    assert EXPERIMENT_ABLATION_RESULTS_REL.as_posix() == "agent_reports/ablation/final/ablation_results.json"
    assert report.artifacts["ablation_results"].endswith("agent_reports/ablation/final/ablation_results.json")
    assert report.artifacts["symbolic_memory_receipt"].endswith(
        "agent_reports/ablation/final/symbolic_memory_receipt.json"
    )


def test_pipeline_experiment_contract_requires_symbolic_memory_receipt_pass(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _write_idea_json(str(workspace_root / "idea.json"))
    ablation_path = workspace_root / EXPERIMENT_ABLATION_RESULTS_REL
    _write_json(str(ablation_path), _valid_ablation_payload())
    _write_symbolic_memory_receipt(
        workspace_root,
        ablation_path,
        status="FAIL",
        blocker="symbolic backend unavailable",
    )

    report = validate_experiment_contract(workspace_root, {})

    assert report.valid is False
    assert any(issue.code == "symbolic_memory_writeback_failed" for issue in report.issues)


def test_pipeline_experiment_contract_rejects_malformed_final_ablation_payload(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _write_idea_json(str(workspace_root / "idea.json"))
    ablation_path = workspace_root / EXPERIMENT_ABLATION_RESULTS_REL
    payload = _valid_ablation_payload()
    payload["components"]["component_a"]["result"] = "custom"
    payload["summary"]["key_findings"] = []
    _write_json(str(ablation_path), payload)

    report = validate_experiment_contract(workspace_root, {})

    assert report.valid is False
    assert any("result must be one of" in issue.message for issue in report.issues)


def test_pipeline_reads_symbolic_memory_receipt_from_ablation_final_path(tmp_path):
    workspace_root = tmp_path / "workspace"

    assert (
        EXPERIMENT_SYMBOLIC_MEMORY_RECEIPT_REL.as_posix()
        == "agent_reports/ablation/final/symbolic_memory_receipt.json"
    )
    assert _symbolic_memory_receipt_path(workspace_root) == workspace_root / EXPERIMENT_SYMBOLIC_MEMORY_RECEIPT_REL


def test_main_returns_blocked_when_master_does_not_converge(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    project_root = workspace_root / "project"
    project_root.mkdir(parents=True)
    paths = artifact_paths(str(workspace_root), str(project_root))
    _write_json(paths["prepare_reviewer"], _pass_report("prepare"))

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("prepare should be reused, not rerun")

    async def fake_master(**kwargs):
        return {
            "iterations": 1,
            "final_path": paths["master_report"],
            "converged": False,
            "stopped_due_to_iteration_limit": False,
            "decision": "RUN_CODE",
            "blocking_issues": ["code phase incomplete"],
        }

    async def fail_finalization_agent(**kwargs):
        raise AssertionError("finalization agent must not run before master convergence")

    monkeypatch.setattr(experiment_main, "print_config", lambda: None)
    monkeypatch.setattr(experiment_main, "print_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment_main, "print_kv_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment_main,
        "ensure_experiment_dirs",
        lambda experiment_id: {
            "workspace_dir": str(workspace_root),
            "project_dir": str(project_root),
            "results_dir": str(workspace_root / "results"),
            "model_dir": str(workspace_root / "model_candidate"),
            "reports_dir": str(workspace_root / "agent_reports"),
            "cache_dir": str(workspace_root / ".cache"),
        },
    )
    monkeypatch.setattr(experiment_main, "copy_prepared_data_to_workspace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_main, "write_workspace_env_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_main, "run_prepare", fail_prepare)
    monkeypatch.setattr(experiment_main, "run_master", fake_master)
    monkeypatch.setattr(experiment_main, "run_finalization_agent", fail_finalization_agent)
    monkeypatch.setattr(experiment_main, "get_idea_input_path", lambda experiment_id: str(workspace_root / "idea.md"))

    result = asyncio.run(
        experiment_main.run_experiment_once(
            experiment_id="demo",
            workspace_root=str(workspace_root),
            verbose=False,
        )
    )

    assert result["ok"] is False
    assert result["converged"] is False
    assert result["decision"] == "RUN_CODE"
    assert result["blocking_issues"] == ["code phase incomplete"]


def test_main_returns_blocked_when_finalization_prefinish_agent_fails(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    project_root = workspace_root / "project"
    project_root.mkdir(parents=True)
    paths = artifact_paths(str(workspace_root), str(project_root))
    _write_json(paths["prepare_reviewer"], _pass_report("prepare"))

    async def fail_prepare(*args, **kwargs):
        raise AssertionError("prepare should be reused, not rerun")

    async def fake_master(**kwargs):
        return {
            "iterations": 1,
            "final_path": "master.md",
            "converged": True,
            "stopped_due_to_iteration_limit": False,
        }

    async def fake_finalization_agent(**kwargs):
        _write_json(
            paths["symbolic_memory_receipt"],
            {
                "status": "FAIL",
                "hook": "final_science_prefinish",
                "blocker": "missing approved component-disabled evidence",
            },
        )
        return {
            "status": "FAIL",
            "ablation_results_path": "",
            "blocker": "missing approved component-disabled evidence",
        }

    monkeypatch.setattr(experiment_main, "print_config", lambda: None)
    monkeypatch.setattr(experiment_main, "print_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment_main, "print_kv_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment_main,
        "ensure_experiment_dirs",
        lambda experiment_id: {
            "workspace_dir": str(workspace_root),
            "project_dir": str(project_root),
            "results_dir": str(workspace_root / "results"),
            "model_dir": str(workspace_root / "model_candidate"),
            "reports_dir": str(workspace_root / "agent_reports"),
            "cache_dir": str(workspace_root / ".cache"),
        },
    )
    monkeypatch.setattr(experiment_main, "copy_prepared_data_to_workspace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_main, "write_workspace_env_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_main, "run_prepare", fail_prepare)
    monkeypatch.setattr(experiment_main, "run_master", fake_master)
    monkeypatch.setattr(experiment_main, "run_finalization_agent", fake_finalization_agent)
    monkeypatch.setattr(experiment_main, "get_idea_input_path", lambda experiment_id: str(workspace_root / "idea.md"))

    result = asyncio.run(
        experiment_main.run_experiment_once(
            experiment_id="demo",
            workspace_root=str(workspace_root),
            verbose=False,
        )
    )

    assert result["ok"] is False
    assert result["finalization_status"] == "FAIL"
    assert result["symbolic_memory_receipt_path"] == paths["symbolic_memory_receipt"]
    assert "missing approved component-disabled evidence" in result["finalization_blocker"]


def test_main_stops_when_prepare_review_gate_fails(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    project_root = workspace_root / "project"
    project_root.mkdir(parents=True)
    paths = artifact_paths(str(workspace_root), str(project_root))

    async def fail_prepare(*args, **kwargs):
        _write_json(
            paths["prepare_reviewer"],
            _pass_report(
                "prepare",
                status="FAIL",
                phase_completion_status="partial",
                ready_for_next_phase=False,
                blocking_issues=["fix prepare handoff"],
            ),
        )
        return type(
            "PrepareReport",
            (),
            {
                "idea_md_path": paths["idea"],
                "project_dir": str(project_root),
                "repos_dir": str(workspace_root / "repos"),
                "dataset_dir": str(workspace_root / "dataset_candidate"),
                "model_dir": str(workspace_root / "model_candidate"),
                "results_dir": str(workspace_root / "results"),
                "reports_dir": str(workspace_root / "agent_reports"),
            },
        )()

    async def fail_master(**kwargs):
        raise AssertionError("master must not run without approved prepare handoff")

    monkeypatch.setattr(experiment_main, "print_config", lambda: None)
    monkeypatch.setattr(experiment_main, "print_phase", lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment_main, "print_kv_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment_main,
        "ensure_experiment_dirs",
        lambda experiment_id: {
            "workspace_dir": str(workspace_root),
            "project_dir": str(project_root),
            "results_dir": str(workspace_root / "results"),
            "model_dir": str(workspace_root / "model_candidate"),
            "reports_dir": str(workspace_root / "agent_reports"),
            "cache_dir": str(workspace_root / ".cache"),
        },
    )
    monkeypatch.setattr(experiment_main, "copy_prepared_data_to_workspace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_main, "write_workspace_env_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(experiment_main, "run_prepare", fail_prepare)
    monkeypatch.setattr(experiment_main, "run_master", fail_master)

    result = asyncio.run(
        experiment_main.run_experiment_once(
            experiment_id="demo",
            workspace_root=str(workspace_root),
            verbose=False,
        )
    )

    assert result["ok"] is False
    assert result["prepare_status"] == "FAIL"
    assert result["prepare_reviewer_path"] == paths["prepare_reviewer"]
