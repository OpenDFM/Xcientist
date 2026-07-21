"""Pipeline runner: Survey -> Idea -> Experiment loop with unified workspace and resume."""

import json
import os
import re
import shutil
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from src.agents.survey_agent.utils.topic_survey_storage import (
    apply_topic_survey_paths,
    build_survey_artifact_paths,
)
from src.config import load_config
from .contracts import (
    EXPERIMENT_ABLATION_RESULTS_REL,
    EXPERIMENT_SYMBOLIC_MEMORY_RECEIPT_REL,
)


def _generate_pipeline_name(topic: str) -> str:
    """Generate pipeline name from topic."""
    # Remove special chars, convert spaces to underscores, lowercase
    name = re.sub(r'[^a-zA-Z0-9\s]', '', topic)
    name = re.sub(r'\s+', '_', name.strip())
    return name.lower()


def _get_pipeline_workspace(workspace_root: str, pipeline_name: str) -> str:
    """Get the pipeline workspace directory."""
    return os.path.join(workspace_root, pipeline_name)


def _ensure_dirs(base_dir: str) -> None:
    """Ensure directory exists."""
    os.makedirs(base_dir, exist_ok=True)


# Pipeline State Management
def _load_pipeline_state(pipeline_workspace: str, state_filename: str) -> Optional[Dict]:
    """Load pipeline state from pipeline.yaml."""
    state_file = os.path.join(pipeline_workspace, state_filename)
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    return None


def _save_pipeline_state(pipeline_workspace: str, state: Dict, state_filename: str) -> None:
    """Save pipeline state to pipeline.yaml."""
    state_file = os.path.join(pipeline_workspace, state_filename)
    conf = OmegaConf.create(state)
    with open(state_file, 'w') as f:
        OmegaConf.save(conf, f)


def _init_pipeline_state(pipeline_workspace: str, topic: str, mature_idea: str, total_iterations: int) -> Dict:
    """Initialize new pipeline state."""
    return {
        "pipeline_name": os.path.basename(pipeline_workspace),
        "topic": topic,
        "mature_idea": mature_idea,
        "last_candidate_path": "",
        "last_ablation_results_dir": "",
        "total_iterations": total_iterations,
        "current_iteration": 0,
        "phases_completed": {
            "survey": False,
        },
        "last_updated": datetime.now().isoformat(),
    }


def _should_skip_phase(phase_name: str, state: Dict) -> bool:
    """Check if phase is already completed."""
    return state.get("phases_completed", {}).get(phase_name, False)


def _mark_phase_complete(phase_name: str, state: Dict) -> None:
    """Mark a phase as completed."""
    if "phases_completed" not in state:
        state["phases_completed"] = {}
    state["phases_completed"][phase_name] = True
    state["last_updated"] = datetime.now().isoformat()


def _get_subprocess_env() -> dict:
    """Get environment variables for subprocess, including .env values."""
    env = os.environ.copy()

    from dotenv import load_dotenv
    env_files = [
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[1] / "config" / ".env",
    ]
    all_keys = [
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "S2_API_KEY",
        "S2_API_TIMEOUT",
        "SEMANTIC_SCHOLAR_API_KEY",
        "SERPER_API_KEY",
        "MINIMAX_API_KEY",
        "JINA_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "HF_TOKEN",
        "http_proxy",
        "https_proxy",
        "OPENHANDS_MCP_TIMEOUT",
    ]
    for env_file in env_files:
        if not env_file.exists():
            continue
        load_dotenv(env_file, override=False)
        for key in all_keys:
            if key not in env or not env.get(key):
                val = os.environ.get(key)
                if val:
                    env[key] = val

    return env


def run_command(cmd: list, env: dict = None) -> int:
    """Run command with real-time output."""
    import sys

    # Use provided env or get from .env
    if env is None:
        env = _get_subprocess_env()

    # Use PIPE with unbuffered mode for real-time output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        env=env,
    )

    # Read and print output in real-time
    # Note: text=True already makes stdout a text stream, no need for TextIOWrapper
    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()
    return process.returncode


def run_survey(topic: str, output_root: str) -> bool:
    """Run Survey agent."""
    print("\n" + "=" * 50)
    print("Phase 1: Survey")
    print("=" * 50)

    project_root = os.getcwd()
    config_path = os.path.join(project_root, "src", "config")
    survey_script = os.path.join(
        project_root,
        "src",
        "agents",
        "survey_agent",
        "scripts",
        "run_deep_survey.py",
    )
    artifacts = build_survey_artifact_paths(topic, output_root=output_root)

    cmd = [
        sys.executable, survey_script,
        f"survey.BasicInfo.topic={topic}",
        f"++survey.BasicInfo.base_dir={artifacts.base_dir}",
        f"++survey.BasicInfo.save_path={artifacts.markdown_path}",
        f"++survey.BasicInfo.save_json_path={artifacts.json_path}",
        f"++survey.BasicInfo.evaluation_save_path={artifacts.evaluation_path}",
        "--config-path", config_path,
        "--config-name", "default",
    ]

    print(f"Running survey: {topic}")
    print(f"Output will be saved to: {artifacts.base_dir}")
    env = _get_subprocess_env()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(key, None)
    env["no_proxy"] = "58.210.177.113,localhost,127.0.0.1"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("MINERU_MODEL_SOURCE", "modelscope")
    returncode = run_command(cmd, env=env)
    return returncode == 0


def _stage_ablation_results_dir(experiment_dir: Path, ablation_path: Path) -> str:
    """Materialize a dedicated ablation-input directory containing one JSON file."""
    if not ablation_path.exists():
        return ""
    staged_dir = experiment_dir / "ligagent_ablation_input"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_path = staged_dir / ablation_path.name
    shutil.copy(ablation_path, staged_path)
    return str(staged_dir)


def _write_idea_runtime_config(
    config: Any,
    runtime_dir: str,
    topic: str,
    mature_idea: str,
    survey_output_root: str,
) -> str:
    """Materialize an idea-agent config that mirrors the active pipeline runtime."""
    runtime_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.update(runtime_config, "idea.run.topic", topic, merge=False)
    OmegaConf.update(runtime_config, "idea.run.mature_idea", mature_idea or "", merge=False)
    if survey_output_root:
        apply_topic_survey_paths(runtime_config.survey, topic, output_root=survey_output_root)
    runtime_config_path = os.path.join(runtime_dir, "idea_runtime_config.yaml")
    OmegaConf.save(runtime_config, runtime_config_path)
    return runtime_config_path


def _find_previous_ablation_results_dir(pipeline_workspace: str, iteration_index: int) -> str:
    """Best-effort lookup for the prior iteration's ablation input directory."""
    if iteration_index <= 0:
        return ""
    experiments_dir = Path(pipeline_workspace) / "experiments"
    if not experiments_dir.exists():
        return ""
    pattern = f"iter_{iteration_index - 1}_*"
    for experiment_dir in sorted(experiments_dir.glob(pattern), reverse=True):
        staged_dir = experiment_dir / "ligagent_ablation_input"
        if staged_dir.exists():
            return str(staged_dir)
        ablation_path = experiment_dir / EXPERIMENT_ABLATION_RESULTS_REL
        if ablation_path.exists():
            return _stage_ablation_results_dir(experiment_dir, ablation_path)
    return ""


def _idea_result_to_mature_idea_text(idea_result: Dict[str, Any]) -> str:
    title = str(idea_result.get("title") or "").strip()
    abstract = str(idea_result.get("abstract") or "").strip()
    method = str(idea_result.get("method") or "").strip()
    sections = []
    if title:
        sections.append(f"Title: {title}")
    if abstract:
        sections.append(f"Abstract: {abstract}")
    if method:
        sections.append(f"Method: {method}")
    return "\n".join(sections).strip()


def _symbolic_memory_receipt_path(experiment_dir: Path) -> Path:
    return experiment_dir / EXPERIMENT_SYMBOLIC_MEMORY_RECEIPT_REL


def _build_experiment_id(iteration_index: int, title: str, branch: str = "") -> str:
    slugified_title = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')
    slugified_title = slugified_title[:50] if slugified_title else f"experiment_{iteration_index}"
    if branch:
        return f"iter_{iteration_index}_{branch}_{slugified_title}"
    return f"iter_{iteration_index}_{slugified_title}"


def _run_experiment_branch(
    *,
    iteration_index: int,
    title: str,
    branch: str,
    idea_source_path: str,
    candidate_source_path: str,
    pipeline_workspace: str,
    idea_result_filename: str,
    experiment_result_filename: str,
    experiment_agent_iterations: int,
    config_path: str,
    resume_enabled: bool,
    state: Dict[str, Any],
    phase_name: str,
) -> Dict[str, str]:
    experiment_id = _build_experiment_id(iteration_index, title, branch=branch)
    exp_workspace = os.path.join(pipeline_workspace, "experiments", experiment_id)
    _ensure_dirs(exp_workspace)
    _ensure_dirs(os.path.join(exp_workspace, "project"))
    _ensure_dirs(os.path.join(exp_workspace, "results"))
    _ensure_dirs(os.path.join(exp_workspace, "checkpoints"))

    final_idea_file = os.path.join(exp_workspace, idea_result_filename)
    final_idea_json = os.path.join(exp_workspace, "idea.json")
    final_candidate_file = os.path.join(exp_workspace, "idea_candidate.json")
    shutil.copy(idea_source_path, final_idea_file)
    shutil.copy(final_idea_file, final_idea_json)
    if candidate_source_path:
        shutil.copy(candidate_source_path, final_candidate_file)

    success = run_experiment(
        experiment_id=experiment_id,
        workspace_root=exp_workspace,
        max_agent_iterations=experiment_agent_iterations,
        config_path=config_path,
        resume=resume_enabled,
    )
    if not success:
        raise RuntimeError(f"Experiment agent failed for {experiment_id}")

    _mark_phase_complete(phase_name, state)

    final_report_path = os.path.join(exp_workspace, "final_report.md")
    experiment_result_path = os.path.join(exp_workspace, experiment_result_filename)
    with open(experiment_result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "workspace": exp_workspace,
                "idea_path": final_idea_json,
                "idea_result_path": final_idea_file,
                "ablation_results_path": str(
                    Path(exp_workspace) / EXPERIMENT_ABLATION_RESULTS_REL
                ),
                "final_report_path": final_report_path if os.path.exists(final_report_path) else "",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "experiment_id": experiment_id,
        "workspace": exp_workspace,
        "idea_result_path": final_idea_file,
        "idea_path": final_idea_json,
        "candidate_path": final_candidate_file if candidate_source_path else "",
    }


def run_idea(
    topic: str,
    mature_idea: str,
    output_file: str,
    config: Any,
    runtime_dir: str,
    survey_output_root: str,
    ablation_results_path: str = "",
    previous_candidate_path: str = "",
) -> Dict[str, Any]:
    """Run Idea agent."""
    print("\n" + "=" * 50)
    print("Phase 2: Idea Generation")
    print("=" * 50)

    idea_agent_path = Path.cwd() / "src/agents/idea_agent/run.py"
    cmd = [sys.executable, str(idea_agent_path)]

    # Get environment with all required API keys
    env = _get_subprocess_env()
    env["IDEA_AGENT_CONFIG"] = _write_idea_runtime_config(
        config,
        runtime_dir,
        topic,
        mature_idea,
        survey_output_root,
    )
    if ablation_results_path and os.path.isdir(ablation_results_path):
        env["IDEA_AGENT_ABLATION_RESULTS_PATH"] = ablation_results_path
        print(f"Using prior ablation results dir: {ablation_results_path}")
    else:
        env.pop("IDEA_AGENT_ABLATION_RESULTS_PATH", None)
    if previous_candidate_path:
        env["IDEA_AGENT_PREVIOUS_CANDIDATE_PATH"] = previous_candidate_path
        print(f"Using prior idea candidate: {previous_candidate_path}")
    else:
        env.pop("IDEA_AGENT_PREVIOUS_CANDIDATE_PATH", None)

    # Capture output to parse result - real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # Note: text=True already makes stdout a text stream, no need for TextIOWrapper
    stdout_lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        stdout_lines.append(line)

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Idea agent failed: {process.returncode}")
    if any("❌ failed:" in line for line in stdout_lines):
        raise RuntimeError("Idea agent reported failure in stdout.")

    # Find and copy idea_result.json to output location
    idea_path = ""
    result_dir = ""
    for line in stdout_lines:
        if "✅ completed ->" in line:
            result_dir = line.split("->", 1)[1].strip()
        if "idea_result.json" in line:
            for part in line.strip().split():
                if "idea_result.json" in part:
                    idea_path = part
                    break

    if result_dir and not idea_path:
        candidate = Path(result_dir) / "idea_result.json"
        if candidate.exists():
            idea_path = str(candidate)

    if not idea_path:
        raise RuntimeError("Idea agent completed but did not report idea_result.json")

    if not os.path.isabs(idea_path):
        idea_path = os.path.join(os.getcwd(), idea_path)
    if not os.path.exists(idea_path):
        raise RuntimeError(f"Idea result not found: {idea_path}")

    # Copy to output location
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    shutil.copy(idea_path, output_file)
    print(f"Idea result saved to: {output_file}")

    with open(idea_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    replanned_idea_path = ""
    candidate_path = ""
    if result_dir:
        main_candidate = Path(result_dir) / "idea_candidate.json"
        if main_candidate.exists():
            candidate_path = str(main_candidate)
        candidate = Path(result_dir) / "replanned_idea_result.json"
        if candidate.exists():
            replanned_idea_path = str(candidate)

    return {
        "payload": payload,
        "idea_path": idea_path,
        "candidate_path": candidate_path,
        "run_dir": result_dir,
        "replanned_idea_path": replanned_idea_path,
    }


def run_experiment(
    experiment_id: str,
    workspace_root: str,
    max_agent_iterations: int,
    config_path: str,
    resume: bool = False,
    skip_prepare: bool = False,
) -> bool:
    """Run Experiment agent (prepare + execute)."""
    print("\n" + "=" * 50)
    print("Phase 3: Experiment")
    print("=" * 50)
    print(f"Experiment workspace: {workspace_root}")

    _ = max_agent_iterations, skip_prepare
    env = _get_subprocess_env()
    env["EXPERIMENT_AGENT_WORKSPACE_DIR"] = workspace_root
    env["XCIENTIST_CONFIG"] = config_path

    print("Running Experiment Agent (unified main)...")
    cmd = [
        sys.executable, "-m", "src.agents.experiment_agent.main",
        "--experiment", experiment_id,
        "--config", config_path,
        "--verbose",
    ]
    if resume:
        cmd.append("--resume")
    return run_command(cmd, env=env) == 0


def _find_latest_experiment_for_blog(
    pipeline_workspace: str,
    experiment_result_filename: str,
) -> tuple[str, str]:
    """Find the most recent completed experiment recorded by the pipeline."""
    experiments_dir = Path(pipeline_workspace) / "experiments"
    if not experiments_dir.exists():
        return "", ""

    result_paths = sorted(
        experiments_dir.glob(f"*/{experiment_result_filename}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for result_path in result_paths:
        if result_path.parent.name.startswith("temp_iter_"):
            continue
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        experiment_id = str(payload.get("experiment_id") or result_path.parent.name)
        workspace = str(payload.get("workspace") or result_path.parent)
        if experiment_id and os.path.exists(workspace):
            return experiment_id, workspace
    return "", ""


def run_blog(experiment_id: str, source_workspace: str, resume: bool = False) -> bool:
    """Run Blog agent after the Idea/Experiment loop completes."""
    print("\n" + "=" * 50)
    print("Phase 4: Blog")
    print("=" * 50)
    print(f"Blog experiment: {experiment_id}")
    print(f"Source workspace: {source_workspace}")

    blog_script = Path.cwd() / "run_blog.sh"
    if not blog_script.exists():
        raise FileNotFoundError(f"Blog runner not found: {blog_script}")

    env = _get_subprocess_env()
    env["BLOG_AGENT_SOURCE_WORKSPACE"] = source_workspace

    cmd = [str(blog_script), "--experiment", experiment_id]
    if resume:
        cmd.append("--resume")
    return run_command(cmd, env=env) == 0


def _apply_topic_override(config: Any, topic_override: Optional[str]) -> str:
    topic = str(topic_override or "").strip()
    if not topic:
        return ""
    OmegaConf.update(config, "idea.topic", topic, merge=False)
    OmegaConf.update(config, "idea.run.topic", topic, merge=False)
    OmegaConf.update(config, "survey.BasicInfo.topic", topic, merge=False)
    return topic


def main(config_path: str = "src/config/default.yaml", topic_override: Optional[str] = None):
    """Main entry point."""
    # Load config
    config = load_config(config_path)
    explicit_topic = _apply_topic_override(config, topic_override)

    # Get settings from config
    workspace_root = config.workspace.root
    topic = config.idea.topic
    mature_idea = config.idea.get("mature_idea", "")
    max_iterations = config.pipeline.iterate.max_iterations
    skip_survey = config.pipeline.get("skip_survey", False)
    resume_enabled = config.pipeline.get("resume_enabled", True)
    state_filename = str(config.pipeline.get("state_file", "pipeline.yaml"))
    experiment_agent_iterations = int(
        config.experiment.execution.get("max_iterations", 10)
    )

    # Get output roots from config
    pipeline_output_root = config.pipeline.output.root  # e.g., "pipeline_runs"
    survey_output_root = str(config.survey.get("output", {}).get("root_dir", "src/agents/survey_agent/outputs"))

    # Pipeline name
    pipeline_name_config = config.pipeline.get("name", "")
    pipeline_name = pipeline_name_config or _generate_pipeline_name(topic)

    # Pipeline workspace: workspace_root/pipeline_output_root/pipeline_name
    pipeline_workspace = os.path.join(workspace_root, pipeline_output_root, pipeline_name)

    if not topic:
        print("Error: idea.topic is required in config")
        return

    print("=" * 50)
    print("X-Scientist Pipeline")
    print("=" * 50)
    print(f"Topic: {topic}")
    print(f"Pipeline Name: {pipeline_name}")
    print(f"Workspace: {pipeline_workspace}")
    print(f"Mature Idea: {mature_idea[:50] + '...' if len(mature_idea) > 50 else mature_idea or 'None'}")
    print(f"Iterations: {max_iterations}")
    print(f"Skip Survey: {skip_survey}")
    print(f"Resume: {resume_enabled}")
    print()

    # Ensure workspace exists
    _ensure_dirs(pipeline_workspace)
    _ensure_dirs(os.path.join(pipeline_workspace, "experiments"))

    # Load or init state
    if resume_enabled:
        state = _load_pipeline_state(pipeline_workspace, state_filename)
        if not state:
            # No existing state, initialize new
            state = _init_pipeline_state(pipeline_workspace, topic, mature_idea, max_iterations)
            print("Starting new pipeline")
        else:
            print(f"Resuming pipeline from iteration {state.get('current_iteration', 0) + 1}")
            topic = explicit_topic or state.get("topic", topic)
            if explicit_topic:
                state["topic"] = explicit_topic
            mature_idea = state.get("mature_idea", mature_idea)
    else:
        state = _init_pipeline_state(pipeline_workspace, topic, mature_idea, max_iterations)

    # Phase 1: Survey
    if skip_survey or _should_skip_phase("survey", state):
        print("Skipping Survey phase")
    else:
        _ensure_dirs(survey_output_root)
        if run_survey(topic, survey_output_root):
            _mark_phase_complete("survey", state)
            _save_pipeline_state(pipeline_workspace, state, state_filename)
        else:
            print("Survey failed!")

    # Phase 2-3: Idea + Experiment loop
    start_iteration = state.get("current_iteration", 0)

    for i in range(start_iteration, max_iterations):
        print(f"\n=== Iteration {i + 1}/{max_iterations} ===")

        # 先运行 Idea agent 获取 title，再确定 experiment_id
        temp_exp_id = f"temp_iter_{i}"
        temp_exp_workspace = os.path.join(pipeline_workspace, "experiments", temp_exp_id)
        _ensure_dirs(temp_exp_workspace)

        state["current_iteration"] = i
        _save_pipeline_state(pipeline_workspace, state, state_filename)

        idea_result_filename = str(
            config.pipeline.output.get("idea_result_filename", "idea_result.json")
        )
        experiment_result_filename = config.pipeline.output.get(
            "experiment_result_filename", "experiment_result.json"
        )
        idea_output_file = os.path.join(temp_exp_workspace, idea_result_filename)
        prior_ablation_results_path = str(
            state.get("last_ablation_results_dir", "") or state.get("last_ablation_results_path", "") or ""
        )
        if not prior_ablation_results_path or not os.path.isdir(prior_ablation_results_path):
            prior_ablation_results_path = _find_previous_ablation_results_dir(pipeline_workspace, i)
        previous_candidate_path = str(state.get("last_candidate_path", "") or "")
        idea_run = run_idea(
            topic,
            mature_idea,
            idea_output_file,
            config=config,
            runtime_dir=temp_exp_workspace,
            survey_output_root=survey_output_root,
            ablation_results_path=prior_ablation_results_path,
            previous_candidate_path=previous_candidate_path,
        )
        idea_result = idea_run["payload"]

        main_experiment = _run_experiment_branch(
            iteration_index=i,
            title=idea_result.get("title", f"experiment_{i}"),
            branch="",
            idea_source_path=idea_output_file,
            candidate_source_path=str(idea_run.get("candidate_path") or ""),
            pipeline_workspace=pipeline_workspace,
            idea_result_filename=idea_result_filename,
            experiment_result_filename=experiment_result_filename,
            experiment_agent_iterations=experiment_agent_iterations,
            config_path=config_path,
            resume_enabled=resume_enabled,
            state=state,
            phase_name=f"experiment_{i}",
        )
        _save_pipeline_state(pipeline_workspace, state, state_filename)

        replanned_idea_path = str(idea_run.get("replanned_idea_path") or "")
        if replanned_idea_path:
            with open(replanned_idea_path, "r", encoding="utf-8") as f:
                replanned_idea_result = json.load(f)
            _run_experiment_branch(
                iteration_index=i,
                title=replanned_idea_result.get("title", f"replanned_experiment_{i}"),
                branch="replan",
                idea_source_path=replanned_idea_path,
                candidate_source_path="",
                pipeline_workspace=pipeline_workspace,
                idea_result_filename=idea_result_filename,
                experiment_result_filename=experiment_result_filename,
                experiment_agent_iterations=experiment_agent_iterations,
                config_path=config_path,
                resume_enabled=resume_enabled,
                state=state,
                phase_name=f"experiment_{i}_replan",
            )

        if os.path.exists(temp_exp_workspace):
            shutil.rmtree(temp_exp_workspace)

        exp_workspace = main_experiment["workspace"]
        state["last_candidate_path"] = main_experiment["candidate_path"]
        state["last_experiment_id"] = main_experiment["experiment_id"]
        state["last_experiment_workspace"] = exp_workspace
        mature_idea = _idea_result_to_mature_idea_text(idea_result)
        state["mature_idea"] = mature_idea
        state["current_iteration"] = i + 1
        _save_pipeline_state(pipeline_workspace, state, state_filename)

        ablation_path = str(Path(exp_workspace) / EXPERIMENT_ABLATION_RESULTS_REL)
        if os.path.exists(ablation_path):
            state["last_ablation_results_dir"] = _stage_ablation_results_dir(
                Path(exp_workspace),
                Path(ablation_path),
            )
            receipt_path = _symbolic_memory_receipt_path(Path(exp_workspace))
            if receipt_path.exists():
                print(f"✅ Experiment finalization wrote symbolic-memory receipt: {receipt_path}")
        else:
            state["last_ablation_results_dir"] = ""
        _save_pipeline_state(pipeline_workspace, state, state_filename)

    state["current_iteration"] = max_iterations
    _save_pipeline_state(pipeline_workspace, state, state_filename)

    experiment_result_filename = str(
        config.pipeline.output.get("experiment_result_filename", "experiment_result.json")
    )
    blog_experiment_id = str(state.get("last_experiment_id") or "")
    blog_source_workspace = str(state.get("last_experiment_workspace") or "")
    if not blog_experiment_id or not os.path.exists(blog_source_workspace):
        blog_experiment_id, blog_source_workspace = _find_latest_experiment_for_blog(
            pipeline_workspace,
            experiment_result_filename,
        )

    if _should_skip_phase("blog", state):
        print("Skipping Blog phase")
    elif not blog_experiment_id or not blog_source_workspace:
        raise RuntimeError("Blog agent cannot run because no completed experiment workspace was found.")
    elif run_blog(
        experiment_id=blog_experiment_id,
        source_workspace=blog_source_workspace,
        resume=resume_enabled,
    ):
        _mark_phase_complete("blog", state)
        _save_pipeline_state(pipeline_workspace, state, state_filename)
    else:
        raise RuntimeError(f"Blog agent failed for {blog_experiment_id}")

    print("\n" + "=" * 50)
    print("Pipeline Completed!")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X-Scientist pipeline runner")
    parser.add_argument(
        "--config",
        default="src/config/default.yaml",
        help="Path to unified config file",
    )
    parser.add_argument(
        "--topic",
        help="Override idea.topic, idea.run.topic, and survey.BasicInfo.topic",
    )
    args = parser.parse_args()
    main(config_path=args.config, topic_override=args.topic)
