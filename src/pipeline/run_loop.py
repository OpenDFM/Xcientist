"""Pipeline runner: Survey -> Idea -> Experiment loop with unified workspace and resume"""

import hashlib
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

from .experiment_to_symbolic import convert_ablation_to_symbolic_memory

# Import unified config
from src.config import load_config


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
    # Start with parent environment
    env = os.environ.copy()

    # Ensure .env variables are available in subprocess
    from dotenv import load_dotenv
    from pathlib import Path
    env_file = Path(__file__).parent.parent / "config" / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        # Update env with .env values (they may not be in parent env)
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
            "HF_TOKEN",
            "http_proxy",
            "https_proxy",
            "OPENHANDS_MCP_TIMEOUT",
        ]
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


def run_survey(topic: str, output_dir: str) -> bool:
    """Run Survey agent."""
    print("\n" + "=" * 50)
    print("Phase 1: Survey")
    print("=" * 50)

    project_root = os.getcwd()
    config_path = os.path.join(project_root, "src/agents/survey_agent/config")

    cmd = [
        sys.executable, "-m", "src.agents.survey_agent.scripts.run_deep_survey",
        f"BasicInfo.topic={topic}",
        "--config-path", config_path,
        "--config-name", "deep_survey",
    ]

    print(f"Running survey: {topic}")
    print(f"Output will be saved to: {output_dir}")
    returncode = run_command(cmd)
    return returncode == 0


def run_idea(topic: str, mature_idea: str, output_file: str) -> Dict[str, Any]:
    """Run Idea agent."""
    print("\n" + "=" * 50)
    print("Phase 2: Idea Generation")
    print("=" * 50)

    idea_agent_path = Path.cwd() / "src/agents/idea_agent/run.py"
 
    cmd = [sys.executable, str(idea_agent_path), "--topic", topic]

    # Add mature_idea if provided
    if mature_idea:
        cmd.extend(["--mature-idea", mature_idea])

    # Get environment with all required API keys
    env = _get_subprocess_env()

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
    stdout_output = "".join(stdout_lines)

    if process.returncode != 0:
        raise RuntimeError(f"Idea agent failed: {process.returncode}")

    # Find and copy idea_result.json to output location
    idea_path = "src/agents/idea_agent/idea_result.json"
    for line in stdout_lines:
        if "idea_result.json" in line:
            for part in line.strip().split():
                if "idea_result.json" in part:
                    idea_path = part
                    break

    if not os.path.isabs(idea_path):
        idea_path = os.path.join(os.getcwd(), idea_path)

    # Copy to output location
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    shutil.copy(idea_path, output_file)
    print(f"Idea result saved to: {output_file}")

    with open(idea_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(
    experiment_id: str,
    workspace_root: str,
    max_agent_iterations: int,
    resume: bool = False,
    skip_prepare: bool = False,
) -> bool:
    """Run Experiment agent (prepare + execute)."""
    print("\n" + "=" * 50)
    print("Phase 3: Experiment")
    print("=" * 50)
    print(f"Experiment workspace: {workspace_root}")

    # Set environment variable for workspace
    env = os.environ.copy()
    env["EXPERIMENT_AGENT_WORKSPACE_DIR"] = workspace_root

    print("Running Experiment Agent (unified main)...")
    cmd = [
        sys.executable, "-m", "src.agents.experiment_agent.main",
        "--experiment", experiment_id,
        "--max-iterations", str(max_agent_iterations),
        "--verbose",
    ]
    if resume:
        cmd.append("--resume")
    if skip_prepare:
        cmd.append("--skip-prepare")
    return run_command(cmd, env=env) == 0


def main(config_path: str = "src/config/default.yaml"):
    """Main entry point."""
    # Load config
    config = load_config(config_path)

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
    survey_output_base = config.survey.output.base_dir  # e.g., "/path/to/workspace/surveys"

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
            topic = state.get("topic", topic)
            mature_idea = state.get("mature_idea", mature_idea)
    else:
        state = _init_pipeline_state(pipeline_workspace, topic, mature_idea, max_iterations)

    # Phase 1: Survey
    if skip_survey or _should_skip_phase("survey", state):
        print("Skipping Survey phase")
    else:
        survey_output_dir = survey_output_base
        _ensure_dirs(survey_output_dir)
        if run_survey(topic, survey_output_dir):
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
        idea_output_file = os.path.join(temp_exp_workspace, idea_result_filename)
        idea_result = run_idea(topic, mature_idea, idea_output_file)

        # 使用 idea title 作为 experiment_id（slugify 处理）
        title = idea_result.get("title", f"experiment_{i}")
        slugified_title = re.sub(r'[^a-zA-Z0-9]+', '-', title.lower()).strip('-')
        # 限制长度避免路径过长，同时保留足够信息
        slugified_title = slugified_title[:50] if slugified_title else f"experiment_{i}"
        experiment_id = f"iter_{i}_{slugified_title}"

        # 创建正式的实验目录
        exp_workspace = os.path.join(pipeline_workspace, "experiments", experiment_id)
        _ensure_dirs(exp_workspace)
        _ensure_dirs(os.path.join(exp_workspace, "project"))
        _ensure_dirs(os.path.join(exp_workspace, "results"))
        _ensure_dirs(os.path.join(exp_workspace, "checkpoints"))

        # Move idea output into the canonical experiment workspace inputs.
        final_idea_file = os.path.join(exp_workspace, idea_result_filename)
        final_idea_json = os.path.join(exp_workspace, "idea.json")
        if os.path.exists(idea_output_file) and idea_output_file != final_idea_file:
            shutil.move(idea_output_file, final_idea_file)

        if os.path.exists(final_idea_file):
            shutil.copy(final_idea_file, final_idea_json)

        # 清理临时目录
        if os.path.exists(temp_exp_workspace) and temp_exp_workspace != exp_workspace:
            try:
                shutil.rmtree(temp_exp_workspace)
            except Exception:
                pass

        if os.path.exists(idea_output_file) and idea_output_file != os.path.join(exp_workspace, idea_result_filename):
            shutil.move(idea_output_file, os.path.join(exp_workspace, idea_result_filename))
        if os.path.exists(os.path.join(exp_workspace, idea_result_filename)):
            shutil.copy(
                os.path.join(exp_workspace, idea_result_filename),
                final_idea_json,
            )

        should_skip_prepare = resume_enabled and _should_skip_phase(f"experiment_{i}", state)
        success = run_experiment(
            experiment_id=experiment_id,
            workspace_root=exp_workspace,
            max_agent_iterations=experiment_agent_iterations,
            resume=resume_enabled,
            skip_prepare=should_skip_prepare,
        )
        if not success:
            raise RuntimeError(f"Experiment agent failed for {experiment_id}")

        _mark_phase_complete(f"experiment_{i}", state)
        state["current_iteration"] = i + 1
        _save_pipeline_state(pipeline_workspace, state, state_filename)

        experiment_result_filename = config.pipeline.output.get(
            "experiment_result_filename", "experiment_result.json"
        )
        experiment_result_path = os.path.join(exp_workspace, experiment_result_filename)
        final_report_path = os.path.join(exp_workspace, "final_report.md")
        with open(experiment_result_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_id": experiment_id,
                    "workspace": exp_workspace,
                    "idea_path": final_idea_json,
                    "idea_result_path": final_idea_file,
                    "ablation_results_path": os.path.join(exp_workspace, "ablation_results.json"),
                    "final_report_path": final_report_path if os.path.exists(final_report_path) else "",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        ablation_path = os.path.join(exp_workspace, "ablation_results.json")
        if os.path.exists(ablation_path):
            try:
                symbolic_memory_path = os.path.join(
                    workspace_root,
                    str(config.pipeline.get("symbolic_memory_path", "idea_skill_priors"))
                )
                convert_ablation_to_symbolic_memory(
                    ablation_path=ablation_path,
                    experiment_id=experiment_id,
                    symbolic_memory_path=symbolic_memory_path,
                    config=config,
                )
                print(f"✅ Converted ablation results to symbolic memory")
            except Exception as e:
                print(f"⚠️ Failed to convert ablation results: {e}")

    state["current_iteration"] = max_iterations
    _save_pipeline_state(pipeline_workspace, state, state_filename)

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
    args = parser.parse_args()
    main(config_path=args.config)
