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
    from src.agents.experiment_agent.config import (
        CODE_VALIDATION_FEEDBACK_ROUNDS,
        PREPARE_VALIDATION_FEEDBACK_ROUNDS,
        SCIENCE_VALIDATION_FEEDBACK_ROUNDS,
        ensure_experiment_dirs,
        ensure_minimax_no_proxy_env,
        get_model_config,
        get_model_share_dir,
        get_workspace_dir,
    )
    from src.agents.experiment_agent.agents.prepare import run_prepare
    from src.agents.experiment_agent.agents.master import run_master
    from src.agents.experiment_agent.agents.base.subagents import create_phase_subagent
    from src.agents.experiment_agent.agents.code import (
        EXPERIMENT_CODE_PLANNER,
        CodeAgent,
        register_experiment_code_planner,
        run_code_agent,
    )
    from src.agents.experiment_agent.agents.code.planner import (
        create_experiment_code_planner_agent,
    )
    from src.agents.experiment_agent.agents.master.entry import Decision, MasterAgent
    from src.agents.experiment_agent.agents.prepare.entry import PrepareAgent
    from src.agents.experiment_agent.agents.science import (
        EXPERIMENT_ABLATION_SCIENCE_PLANNER,
        EXPERIMENT_STANDARD_SCIENCE_PLANNER,
        AblationScienceAgent,
        ScienceAgent,
        StandardScienceAgent,
        register_science_planners,
        run_ablation_science_agent,
        run_science_agent,
        run_standard_science_agent,
    )
    from src.agents.experiment_agent.agents.science.planner import (
        create_standard_science_planner_agent,
    )
    from src.agents.experiment_agent.agents.reporting.integrator import (
        create_ablation_report_integrator_agent,
    )
    from src.agents.experiment_agent.runtime.manifests import artifact_paths, load_workspace_state
    from src.agents.experiment_agent.skills import (
        get_code_agent_context,
        get_prepare_agent_context,
        get_worker_agent_context,
    )
    from src.agents.experiment_agent.tools import enable_experiment_tool_overrides
    from src.agents.experiment_agent.tools.bounded_io import BoundedFileEditorAction
    from src.agents.experiment_agent.tools.resource_tools import _resolve_local_dir
    from src.agents.experiment_agent.tools.openhands import SecurityValidator
    from openhands.sdk.tool.registry import get_tool_module_qualnames
else:
    CODE_VALIDATION_FEEDBACK_ROUNDS = 1
    PREPARE_VALIDATION_FEEDBACK_ROUNDS = 1
    SCIENCE_VALIDATION_FEEDBACK_ROUNDS = 1

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


def test_runtime_artifact_paths_focus_on_validator_backed_outputs(tmp_path):
    paths = artifact_paths(str(tmp_path))

    assert "ablation_results" in paths
    assert "idea_json" in paths
    assert paths["results_dir"].endswith("results")
    assert paths["agent_reports_dir"].endswith("agent_reports")
    assert paths["standard_summary"].endswith(os.path.join("agent_reports", "standard_science_summary.md"))
    assert paths["ablation_summary"].endswith(os.path.join("agent_reports", "ablation_science_summary.md"))
    assert paths["results_summary"].endswith(os.path.join("agent_reports", "master_summary.md"))
    assert paths["ablation_results"].endswith("ablation_results.json")
    assert paths["final_artifact_contract"].endswith(os.path.join("agent_reports", "final_artifact_contract.json"))
    assert not paths["ablation_results"].endswith(os.path.join("results", "ablation_results.json"))
    assert "prepare_validator" in paths
    assert "self_contained_report" in paths
    assert "prepare_repo_validator" in paths
    assert "prepare_model_worker" in paths
    assert "prepare_model_validator" in paths
    assert paths["model_dir"].endswith("model_candidate")
    assert paths["model_share_dir"].endswith(os.path.join("model_candidate", "model_share"))
    assert "code_validator" in paths
    assert "standard_science_validator" in paths
    assert "prepare_manifest" not in paths
    assert "benchmark_results" not in paths


def test_execution_feedback_round_constants_are_loaded():
    assert PREPARE_VALIDATION_FEEDBACK_ROUNDS >= 1
    assert CODE_VALIDATION_FEEDBACK_ROUNDS >= 1
    assert SCIENCE_VALIDATION_FEEDBACK_ROUNDS >= 1


def test_curated_contexts_include_project_agents_skill():
    prepare_context = get_prepare_agent_context()
    code_context = get_code_agent_context()
    science_worker_context = get_worker_agent_context("science_worker")

    prepare_skill_names = {skill.name for skill in prepare_context.skills}
    code_skill_names = {skill.name for skill in code_context.skills}
    science_worker_skill_names = {skill.name for skill in science_worker_context.skills}

    assert "agents" in prepare_skill_names
    assert "agents" in code_skill_names
    assert "bounded-tool-use" in prepare_skill_names
    assert "bounded-tool-use" in code_skill_names
    assert "bounded-tool-use" in science_worker_skill_names
    assert "resource-acquisition" in prepare_skill_names
    assert "code-enablement" in code_skill_names
    assert "science-execution" in science_worker_skill_names
    assert "component-coverage" in science_worker_skill_names


def test_new_agent_entrypoints_are_importable():
    assert callable(run_prepare)
    assert callable(run_master)
    assert callable(run_code_agent)
    assert callable(run_science_agent)
    assert callable(run_standard_science_agent)
    assert callable(run_ablation_science_agent)
    assert ScienceAgent is StandardScienceAgent


def test_prepare_prompt_uses_phase_local_worker_and_validator(tmp_path):
    agent = PrepareAgent(verbose=False, workspace_root=str(tmp_path))
    canonical_components = [
        {"component": "component_a", "explanation": "first", "index": "1"}
    ]
    prompt = agent._build_user_prompt(
        experiment_id="demo",
        idea_json_path=str(tmp_path / "idea.json"),
        workspace_dir=str(tmp_path),
        project_dir=str(tmp_path / "project"),
        repos_dir=str(tmp_path / "repos"),
        dataset_dir=str(tmp_path / "dataset_candidate"),
        model_dir=str(tmp_path / "model_candidate"),
        results_dir=str(tmp_path / "results"),
        reports_dir=str(tmp_path / "agent_reports"),
        idea_summary={},
        canonical_components=canonical_components,
        force=False,
        clone_depth=1,
        skip_repos=False,
        skip_datasets=False,
        candidate_env_vars=[],
    )
    assert "`task`" in prompt
    assert "prepare_step_executor" in prompt
    assert "prepare_repo_worker" in prompt
    assert "prepare_model_worker" in prompt
    assert "prepare_validator" in prompt
    assert "dataset_candidate/" in prompt
    assert "model_candidate/" in prompt
    assert "model_candidate/model_share" in prompt
    assert "prepare_target_inventory.json" in prompt
    assert "OPENAI_API_KEY" in prompt
    assert "OPENAI_API_BASE" in prompt
    assert "results_dir" in prompt
    assert "agent_reports_dir" in prompt
    assert "## Canonical Idea Components" in prompt
    assert "component_a" in prompt
    assert "prepare_manifest.json" not in prompt


def test_prepare_planner_tools_are_capability_isolated(tmp_path):
    agent = PrepareAgent(verbose=False, workspace_root=str(tmp_path))
    tool_names = {tool.name for tool in agent._get_tools()}

    assert "task_tool_set" in tool_names
    assert "task_tracker" in tool_names
    assert "file_editor" in tool_names
    assert "web_search" in tool_names
    assert "terminal" not in tool_names


def test_experiment_tool_overrides_replace_openhands_defaults():
    enable_experiment_tool_overrides()
    module_qualnames = get_tool_module_qualnames()
    assert module_qualnames["file_editor"] == "src.agents.experiment_agent.tools.bounded_io"
    assert module_qualnames["terminal"] == "src.agents.experiment_agent.tools.bounded_io"


def test_bounded_file_editor_action_exposes_targeted_commands():
    schema = BoundedFileEditorAction.to_mcp_schema()
    command_schema = schema["properties"]["command"]
    assert "search" in command_schema["enum"]
    assert "read_json" in command_schema["enum"]
    assert "stat" in command_schema["enum"]


def test_experiment_agents_disable_filesystem_mcp_by_default():
    assert PrepareAgent.PREPARE_DEFAULT_MCP_SERVERS == ["fetch", "github"]
    assert CodeAgent.CODE_DEFAULT_MCP_SERVERS == []
    assert StandardScienceAgent.SCIENCE_DEFAULT_MCP_SERVERS == []
    assert AblationScienceAgent.SCIENCE_DEFAULT_MCP_SERVERS == []
    assert MasterAgent.MASTER_DEFAULT_MCP_SERVERS == []


def test_experiment_agent_mcp_configs_exclude_thinking_server(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()

    prepare_agent = PrepareAgent(verbose=False, workspace_root=str(tmp_path))
    code_agent = CodeAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        project_root=str(project_root),
        workspace_root=str(tmp_path),
        plan="plan",
        verbose=False,
    )
    master_agent = MasterAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        verbose=False,
    )

    prepare_mcp = prepare_agent._build_mcp_config()["mcpServers"]
    assert sorted(prepare_mcp.keys()) == ["fetch", "github"]
    assert "thinking" not in prepare_mcp
    assert "thinking" not in code_agent._build_mcp_config()["mcpServers"]
    assert "thinking" not in master_agent._build_mcp_config()["mcpServers"]


def test_phase_and_planner_subagent_factories_force_builtin_think_and_optional_prepare_mcp(monkeypatch):
    captured = {}

    class DummyAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.base.subagents.Agent",
        DummyAgent,
    )
    create_phase_subagent(
        llm=object(),
        role="prepare_worker",
        tool_names=["terminal"],
        system_prompt="test",
    )
    assert captured["include_default_tools"] == ["FinishTool", "ThinkTool"]
    assert "mcp_config" not in captured

    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.code.planner.Agent",
        DummyAgent,
    )
    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.code.planner._register_code_subagents",
        lambda: None,
    )
    captured.clear()
    create_experiment_code_planner_agent(object())
    assert captured["include_default_tools"] == ["FinishTool", "ThinkTool"]
    assert "mcp_config" not in captured

    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.science.planner.Agent",
        DummyAgent,
    )
    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.science.planner._register_science_subagents",
        lambda: None,
    )
    captured.clear()
    create_standard_science_planner_agent(object())
    assert captured["include_default_tools"] == ["FinishTool", "ThinkTool"]
    assert "mcp_config" not in captured

    monkeypatch.setattr(
        "src.agents.experiment_agent.agents.reporting.integrator.Agent",
        DummyAgent,
    )
    captured.clear()
    create_ablation_report_integrator_agent(object())
    assert captured["include_default_tools"] == ["FinishTool", "ThinkTool"]
    assert "mcp_config" not in captured


def test_register_helpers_always_register_phase_step_executors(monkeypatch):
    from src.agents.experiment_agent.agents.code import planner as code_planner
    from src.agents.experiment_agent.agents.science import planner as science_planner

    registered = []

    def fake_register_agent(*, name, factory_func, description):
        registered.append(name)

    monkeypatch.setattr(code_planner, "register_agent", fake_register_agent)
    monkeypatch.setattr(science_planner, "register_agent", fake_register_agent)

    monkeypatch.setattr(code_planner, "_CODE_PLANNER_REGISTERED", True)
    monkeypatch.setattr(code_planner, "_CODE_SUBAGENTS_REGISTERED", False)
    monkeypatch.setattr(science_planner, "_SCIENCE_PLANNERS_REGISTERED", True)
    monkeypatch.setattr(science_planner, "_SCIENCE_SUBAGENTS_REGISTERED", False)

    code_planner.register_experiment_code_planner()
    science_planner.register_science_planners()

    assert "code_step_executor" in registered
    assert "standard_science_step_executor" in registered
    assert "ablation_science_step_executor" in registered


def test_prepare_stage_specs_use_phase_local_subagents():
    observed_triplets = [
        (spec["executor_type"], spec["worker_type"], spec["validator_type"])
        for spec in PrepareAgent.PREPARE_STAGE_SPECS
    ]
    assert observed_triplets == [
        ("prepare_step_executor", "prepare_repo_worker", "prepare_validator"),
        ("prepare_step_executor", "prepare_env_worker", "prepare_validator"),
        ("prepare_step_executor", "prepare_dataset_worker", "prepare_validator"),
        ("prepare_step_executor", "prepare_model_worker", "prepare_validator"),
        ("prepare_step_executor", "prepare_synthesis_worker", "prepare_validator"),
    ]


def test_code_agent_prompt_requires_worker_validator_loop(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()

    agent = CodeAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        project_root=str(project_root),
        workspace_root=str(tmp_path),
        plan="enable missing conditions",
        verbose=False,
    )

    prompt = agent._build_user_prompt()
    assert "`task`" in prompt
    assert "code_step_executor" in prompt
    assert "code_worker" in prompt
    assert "code_validator" in prompt
    assert "prepare_validator_report.json" in prompt
    assert str(project_root) in prompt
    assert os.path.join(str(tmp_path), "results") in prompt
    assert "formal science result artifact" in prompt
    assert "exact component names" in prompt


def test_science_prompts_split_standard_and_ablation_roles(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(
        tmp_path / "idea.json",
        components=[
            {"component": "component_a", "explanation": "first"},
            {"component": "component_b", "explanation": "second"},
        ],
    )
    project_root = tmp_path / "project"
    project_root.mkdir()

    standard_agent = StandardScienceAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        project_root=str(project_root),
        workspace_root=str(tmp_path),
        plan="run standard science",
        verbose=False,
    )
    ablation_agent = AblationScienceAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        project_root=str(project_root),
        workspace_root=str(tmp_path),
        plan="run ablation science",
        verbose=False,
    )

    standard_prompt = standard_agent._build_user_prompt()
    ablation_prompt = ablation_agent._build_user_prompt()

    assert "science_step_executor" in standard_prompt
    assert "science_worker" in standard_prompt
    assert "science_validator" in standard_prompt
    assert os.path.join(str(tmp_path), "results", "standard") in standard_prompt
    assert "benchmark_results.json" not in standard_prompt
    assert "idea_json_path" in standard_prompt
    assert "science_step_executor" in ablation_prompt
    assert "science_worker" in ablation_prompt
    assert "science_validator" in ablation_prompt
    assert os.path.join(str(tmp_path), "results", "ablation") in ablation_prompt
    assert "Do not write `ablation_results.json` yourself" in ablation_prompt
    assert "component_a" in ablation_prompt
    assert "component_b" in ablation_prompt
    assert "same order" in ablation_prompt


def test_master_agent_tools_include_task_tool(tmp_path):
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
        verbose=False,
    )

    tool_names = {tool.name for tool in agent._get_tools()}
    assert "task_tool_set" in tool_names
    assert "terminal" in tool_names


def test_planner_registration_helpers_are_callable():
    register_experiment_code_planner()
    register_science_planners()
    assert EXPERIMENT_CODE_PLANNER == "experiment_code_planner"
    assert EXPERIMENT_STANDARD_SCIENCE_PLANNER == "experiment_standard_science_planner"
    assert EXPERIMENT_ABLATION_SCIENCE_PLANNER == "experiment_ablation_science_planner"


def test_master_gate_is_validator_driven_and_writes_ablation_results(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    (tmp_path / "code_summary.md").write_text("code ok", encoding="utf-8")
    project_root = tmp_path / "project"
    project_root.mkdir()

    paths = artifact_paths(str(tmp_path))
    os.makedirs(paths["results_dir"], exist_ok=True)
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
                    "value": "+0.1",
                    "confidence": 0.9,
                    "analysis": "evidence-backed",
                    "method_context": "ablated variant",
                    "follow_up_required": False,
                }
            },
            summary={
                "feasible": True,
                "confidence": 0.9,
                "key_findings": ["component_a matters"],
            },
            evidence_paths=["results/ablation/run_001"],
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
    summary_path = agent._materialize_results_summary()
    assert summary_path == paths["results_summary"]
    ablation_results = load_workspace_state(str(tmp_path))["ablation_results"]
    assert set(ablation_results.keys()) == {"components", "summary"}
    assert ablation_results["components"]["component_a"]["result"] == "positive"
    assert ablation_results["components"]["component_a"]["method_context"] == "demo explanation"
    assert ablation_results["summary"]["feasible"] is True
    assert "evidence_paths" not in ablation_results
    assert "metadata" not in ablation_results


def test_master_requires_prepare_handoff_before_running(tmp_path):
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
        verbose=False,
    )

    with pytest.raises(ValueError, match="prepare handoff"):
        agent._ensure_prepare_ready()


def test_master_accepts_partial_prepare_verdict_when_ready_to_proceed(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    project_root = tmp_path / "project"
    project_root.mkdir()

    paths = artifact_paths(str(tmp_path))
    _write_json(
        paths["prepare_validator"],
        {
            "status": "PARTIAL",
            "scope": "prepare",
            "checked_artifacts": [],
            "findings": [],
            "required_fixes": [],
            "evidence_summary": "usable with caveats",
            "ready_to_proceed": True,
        },
    )

    agent = MasterAgent(
        experiment_id="demo",
        idea_path=str(idea_path),
        workspace_root=str(tmp_path),
        project_root=str(project_root),
        verbose=False,
    )

    agent._ensure_prepare_ready()
    payload = agent._compute_gate_payload()
    assert payload["decision"] == Decision.CODE_NEEDED


def test_master_requires_ablation_follow_up_to_clear(tmp_path):
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("# idea", encoding="utf-8")
    _write_idea_json(tmp_path / "idea.json")
    (tmp_path / "code_summary.md").write_text("code ok", encoding="utf-8")
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
                    "result": "inconclusive",
                    "metric": "accuracy",
                    "value": "0.0",
                    "confidence": 0.4,
                    "analysis": "need more evidence",
                    "method_context": "ablated variant",
                    "follow_up_required": True,
                }
            },
            summary={
                "feasible": True,
                "confidence": 0.4,
                "key_findings": ["need more evidence"],
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
    assert payload["decision"] == Decision.ABLATION_NEEDED
    assert any("follow-up" in reason.lower() for reason in payload["reasons"])


def test_master_runs_code_then_standard_then_ablation_until_converged(tmp_path, monkeypatch):
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
            {"decision": Decision.CODE_NEEDED, "phase": "code", "reasons": ["Code validator has not passed."], "evidence_files": []},
            {"decision": Decision.STANDARD_EXP_NEEDED, "phase": "standard_science", "reasons": ["Standard science validator has not passed."], "evidence_files": []},
            {"decision": Decision.STANDARD_EXP_NEEDED, "phase": "standard_science", "reasons": ["Standard science validator has not passed."], "evidence_files": []},
            {"decision": Decision.ABLATION_NEEDED, "phase": "ablation_science", "reasons": ["Ablation science validator has not passed."], "evidence_files": []},
            {"decision": Decision.ABLATION_NEEDED, "phase": "ablation_science", "reasons": ["Ablation science validator has not passed."], "evidence_files": []},
            {"decision": Decision.CONVERGED, "phase": "complete", "reasons": ["All validator-backed gates passed."], "evidence_files": []},
        ]
    )
    calls = []

    def fake_gate():
        return next(gate_sequence)

    async def fake_run_planner_task(subagent_type: str, description: str, planner_prompt: str):
        calls.append(subagent_type)
        if subagent_type == EXPERIMENT_CODE_PLANNER:
            (tmp_path / "code_summary.md").write_text("code ok", encoding="utf-8")
        else:
            results_dir = tmp_path / "results"
            results_dir.mkdir(exist_ok=True)
            if subagent_type == EXPERIMENT_STANDARD_SCIENCE_PLANNER:
                standard_dir = results_dir / "standard"
                standard_dir.mkdir(exist_ok=True)
                (standard_dir / "summary.md").write_text("science ok", encoding="utf-8")
            else:
                ablation_dir = results_dir / "ablation"
                ablation_dir.mkdir(exist_ok=True)
                (ablation_dir / "summary.md").write_text("ablation ok", encoding="utf-8")
        return f"{subagent_type} ok"

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


def test_ensure_minimax_no_proxy_env_adds_minimax_host(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("no_proxy", "localhost")

    ensure_minimax_no_proxy_env("https://api.minimaxi.com/v1")

    assert "api.minimaxi.com" in os.environ["NO_PROXY"].split(",")
    assert "api.minimaxi.com" in os.environ["no_proxy"].split(",")


def test_experiment_model_config_uses_unified_code_agent_field():
    model_cfg = get_model_config()

    assert model_cfg["code"]["agent"]
    assert "architect" not in model_cfg["code"]


def test_workspace_dir_prefers_new_experiment_workspace_env(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_AGENT_WORKSPACE_DIR", "/tmp/new-workspace")
    monkeypatch.setenv("CODEAGENT_WORKSPACES_DIR", "/tmp/old-workspace")

    assert get_workspace_dir("demo") == "/tmp/new-workspace"


def test_workspace_dir_falls_back_to_legacy_workspace_env(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_AGENT_WORKSPACE_DIR", raising=False)
    monkeypatch.setenv("CODEAGENT_WORKSPACES_DIR", "/tmp/old-workspace")

    assert get_workspace_dir("demo") == "/tmp/old-workspace"


def test_model_share_dir_helper_nests_under_model_candidate(monkeypatch):
    monkeypatch.setenv("EXPERIMENT_AGENT_WORKSPACE_DIR", "/tmp/demo-workspace")
    assert get_model_share_dir("demo") == "/tmp/demo-workspace/model_candidate/model_share"


def test_resolve_local_dir_rejects_model_share_writes(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "model_candidate" / "model_share").mkdir(parents=True)
    with pytest.raises(ValueError, match="model_candidate/model_share"):
        _resolve_local_dir("model_candidate/model_share/demo-model", str(workspace))


def test_security_validator_allows_model_share_lexical_reads(tmp_path):
    workspace = tmp_path / "workspace"
    mounted = workspace / "model_candidate" / "model_share" / "demo-model"
    mounted.mkdir(parents=True)
    assert SecurityValidator.validate_path(str(mounted), str(workspace)) is True


def test_ensure_experiment_dirs_creates_model_share_mount(tmp_path, monkeypatch):
    workspace_root = tmp_path / "workspaces"
    seed_root = tmp_path / "seed_models"
    seed_root.mkdir()
    monkeypatch.setenv("EXPERIMENT_AGENT_WORKSPACE_DIR", str(workspace_root / "demo"))

    original = os.environ.get("EXPERIMENT_AGENT_WORKSPACE_DIR")
    from src.agents.experiment_agent import config as experiment_config
    original_root = experiment_config.BASE_WORKSPACES_DIR
    original_cfg = experiment_config.get_workspace_config

    def fake_workspace_cfg():
        cfg = original_cfg()
        cfg["root"] = str(workspace_root)
        cfg["model_candidate_seed"] = str(seed_root)
        return cfg

    monkeypatch.setattr(experiment_config, "get_workspace_config", fake_workspace_cfg)
    monkeypatch.setattr(experiment_config, "BASE_WORKSPACES_DIR", str(workspace_root))

    paths = ensure_experiment_dirs("demo")
    assert os.path.isdir(paths["model_dir"])
    assert os.path.islink(paths["model_share_dir"])
    assert os.path.realpath(paths["model_share_dir"]) == os.path.realpath(str(seed_root))
