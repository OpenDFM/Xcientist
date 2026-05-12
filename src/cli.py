from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.agents.survey_agent.utils.topic_survey_storage import (
    apply_topic_survey_paths,
    get_survey_output_root,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "src" / "config" / "default.yaml"
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
LEGACY_ENV_PATH = REPO_ROOT / "src" / "config" / ".env"
DEFAULT_MCP_WRAPPER_DIR = Path.home() / ".cache" / "researchagent_mcp" / "bin"


def _load_project_env() -> Path | None:
    loaded: Path | None = None
    for candidate in (DEFAULT_ENV_PATH, LEGACY_ENV_PATH):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            if loaded is None:
                loaded = candidate
    return loaded


def _base_env(*, config_path: Path | None = None) -> dict[str, str]:
    _load_project_env()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if config_path is not None:
        env["XCIENTIST_CONFIG_PATH"] = str(config_path.resolve())
    return env


def _run_command(cmd: Sequence[str], *, env: dict[str, str] | None = None) -> int:
    process = subprocess.run(
        list(cmd),
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    return int(process.returncode)


def _resolve_config_path(config: str | None) -> Path:
    path = Path(config).expanduser() if config else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _ensure_config_exists(config_path: Path) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")


def _survey_override_key(config_path: Path, key: str) -> str:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    prefix = "survey." if isinstance(config, dict) and "survey" in config else ""
    return f"{prefix}{key}"


def _temporary_config(base_config: Path, updates: Iterable[tuple[str, object]]) -> str | None:
    update_pairs = [(key, value) for key, value in updates if value is not None and value != ""]
    if not update_pairs:
        return None
    config = OmegaConf.load(base_config)
    for key, value in update_pairs:
        OmegaConf.update(config, key, value, merge=False)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        OmegaConf.save(config, handle.name)
        return handle.name


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Xcientist uv-friendly CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    survey = subparsers.add_parser("survey", help="Run Survey Agent")
    survey.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML")
    survey.add_argument("--topic", help="Override survey BasicInfo.topic")
    survey.add_argument("--base-dir", help="Override survey BasicInfo.base_dir")
    survey.add_argument("--save-path", help="Override survey BasicInfo.save_path")
    survey.add_argument("--save-json-path", help="Override survey BasicInfo.save_json_path")
    survey.add_argument(
        "--evaluation-save-path",
        help="Override survey.BasicInfo.evaluation_save_path",
    )
    survey.add_argument("overrides", nargs="*", help="Additional Hydra overrides")
    survey.set_defaults(func=_survey_command)

    idea = subparsers.add_parser("idea", help="Run Idea Agent")
    idea.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML")
    idea.add_argument("--topic", help="Override idea.run.topic")
    idea.add_argument("--input", help="Override idea.run.input")
    idea.add_argument("--mature-idea", help="Override idea.run.mature_idea")
    idea.add_argument("--refinement-scope", help="Override idea.run.refinement_scope")
    idea.add_argument("--output-root", help="Override idea.run.output_root")
    idea.add_argument(
        "--ablation-results-path",
        help="Set IDEA_AGENT_ABLATION_RESULTS_PATH",
    )
    idea.add_argument(
        "--previous-candidate-path",
        help="Set IDEA_AGENT_PREVIOUS_CANDIDATE_PATH",
    )
    idea.set_defaults(func=_idea_command)

    experiment = subparsers.add_parser("experiment", help="Run Experiment Agent")
    experiment.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML")
    experiment.add_argument("--experiment", required=True, help="Experiment ID")
    experiment.add_argument("--idea-json", required=True, help="Path to idea_result.json")
    experiment.add_argument("--prepare-only", action="store_true", help="Only run prepare phase")
    experiment.add_argument("--resume", action="store_true", help="Resume experiment execution")
    experiment.add_argument("--force", action="store_true", help="Force rerun prepare phase")
    experiment.add_argument("--skip-repos", action="store_true", help="Skip repo cloning in prepare")
    experiment.add_argument("--skip-datasets", action="store_true", help="Skip dataset downloads in prepare")
    experiment.add_argument("--clone-depth", type=int, default=1, help="git clone depth")
    experiment.add_argument("--max-iterations", type=int, help="Override science max iterations")
    experiment.add_argument(
        "--install-mcp-wrappers",
        action="store_true",
        help="Install local MCP wrapper scripts before launch",
    )
    experiment.set_defaults(func=_experiment_command)

    blog = subparsers.add_parser("blog", help="Run Blog Agent")
    blog.add_argument("--experiment", required=True, help="Experiment/project name")
    blog.add_argument("--resume", action="store_true", help="Resume from the last completed step")
    blog.add_argument(
        "--source-workspace",
        help="Set BLOG_AGENT_SOURCE_WORKSPACE for an experiment workspace outside the default source path",
    )
    blog.set_defaults(func=_blog_command)

    pipeline = subparsers.add_parser("pipeline", help="Run Survey -> Idea -> Experiment loop")
    pipeline.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML")
    pipeline.add_argument("--topic", help="Override pipeline research topic")
    pipeline.set_defaults(func=_pipeline_command)

    doctor = subparsers.add_parser("doctor", help="Check local runtime prerequisites")
    doctor.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML")
    doctor.set_defaults(func=_doctor_command)

    install = subparsers.add_parser("install-mcp-wrappers", help="Install local MCP wrapper scripts")
    install.set_defaults(func=_install_mcp_wrappers_command)

    return parser


def _survey_command(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args.config)
    _ensure_config_exists(config_path)
    override_key = lambda key: _survey_override_key(config_path, key)
    config = OmegaConf.load(config_path)
    env = _base_env(config_path=config_path)
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(key, None)
    env["no_proxy"] = "58.210.177.113,localhost,127.0.0.1"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("MINERU_MODEL_SOURCE", "modelscope")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "agents" / "survey_agent" / "scripts" / "run_deep_survey.py"),
        "--config-path",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
    ]
    derived_paths = None
    if args.topic and not any((args.base_dir, args.save_path, args.save_json_path, args.evaluation_save_path)):
        survey_config = config.get("survey") if hasattr(config, "get") and config.get("survey") is not None else config
        derived_paths = apply_topic_survey_paths(
            OmegaConf.create(OmegaConf.to_container(survey_config, resolve=False)),
            args.topic,
            output_root=get_survey_output_root(survey_config),
        )
    if args.topic:
        cmd.append(f"{override_key('BasicInfo.topic')}={args.topic}")
    if args.base_dir or derived_paths:
        cmd.append(f"++{override_key('BasicInfo.base_dir')}={args.base_dir or derived_paths.base_dir}")
    if args.save_path or derived_paths:
        cmd.append(f"++{override_key('BasicInfo.save_path')}={args.save_path or derived_paths.markdown_path}")
    if args.save_json_path or derived_paths:
        cmd.append(f"++{override_key('BasicInfo.save_json_path')}={args.save_json_path or derived_paths.json_path}")
    if args.evaluation_save_path or derived_paths:
        cmd.append(
            f"++{override_key('BasicInfo.evaluation_save_path')}="
            f"{args.evaluation_save_path or derived_paths.evaluation_path}"
        )
    cmd.extend(args.overrides)
    return _run_command(cmd, env=env)


def _idea_command(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args.config)
    _ensure_config_exists(config_path)
    runtime_config = _temporary_config(
        config_path,
        [
            ("idea.run.topic", args.topic),
            ("idea.run.input", args.input),
            ("idea.run.mature_idea", args.mature_idea),
            ("idea.run.refinement_scope", args.refinement_scope),
            ("idea.run.output_root", args.output_root),
        ],
    )
    env = _base_env(config_path=config_path)
    env["IDEA_AGENT_CONFIG"] = runtime_config or str(config_path)
    if args.ablation_results_path:
        env["IDEA_AGENT_ABLATION_RESULTS_PATH"] = str(Path(args.ablation_results_path).expanduser().resolve())
    if args.previous_candidate_path:
        env["IDEA_AGENT_PREVIOUS_CANDIDATE_PATH"] = str(
            Path(args.previous_candidate_path).expanduser().resolve()
        )
    try:
        return _run_command(
            [sys.executable, str(REPO_ROOT / "src" / "agents" / "idea_agent" / "run.py")],
            env=env,
        )
    finally:
        if runtime_config:
            Path(runtime_config).unlink(missing_ok=True)


def _experiment_command(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args.config)
    _ensure_config_exists(config_path)
    idea_json_path = Path(args.idea_json).expanduser().resolve()
    if not idea_json_path.exists():
        raise FileNotFoundError(f"Idea JSON not found: {idea_json_path}")

    if args.install_mcp_wrappers:
        rc = _install_mcp_wrappers_command(argparse.Namespace())
        if rc != 0:
            return rc

    from src.config import load_config

    config = load_config(str(config_path))
    workspace_root = Path(str(config.experiment.workspace.root)).expanduser()
    if not workspace_root.is_absolute():
        workspace_root = (REPO_ROOT / workspace_root).absolute()
    experiment_dir = workspace_root / args.experiment
    experiment_dir.mkdir(parents=True, exist_ok=True)

    target_idea_json = experiment_dir / "idea.json"
    target_idea_result_json = experiment_dir / "idea_result.json"
    if idea_json_path != target_idea_json.resolve():
        shutil.copy2(idea_json_path, target_idea_json)
    shutil.copy2(idea_json_path, target_idea_result_json)

    env = _base_env(config_path=config_path)
    env.setdefault("SHOW_LLM_REASONING", "1")
    env.setdefault("EXPERIMENT_AGENT_MEMORY_TOOL_LOGS", "0")
    env.setdefault("EXPERIMENT_AGENT_MEMORY_ENABLED", "0")
    env.setdefault("EXPERIMENT_AGENT_MEMORY_WRITEBACK", "0")
    env.setdefault("AGENT_BASH_TIMEOUT_SECONDS", "600000")
    env.setdefault("EXPERIMENT_AGENT_MCP_USE_WRAPPERS", "1")
    env.setdefault("EXPERIMENT_AGENT_MCP_WRAPPER_DIR", str(DEFAULT_MCP_WRAPPER_DIR))
    env["EXPERIMENT_AGENT_WORKSPACE_DIR"] = str(experiment_dir)

    cmd = [
        sys.executable,
        "-m",
        "src.agents.experiment_agent.main",
        "--experiment",
        args.experiment,
        "--verbose",
        "--clone-depth",
        str(args.clone_depth),
    ]
    if args.prepare_only:
        cmd.append("--prepare-only")
    if args.resume:
        cmd.append("--resume")
    if args.force:
        cmd.append("--force")
    if args.skip_repos:
        cmd.append("--skip-repos")
    if args.skip_datasets:
        cmd.append("--skip-datasets")
    if args.max_iterations is not None:
        cmd.extend(["--max-iterations", str(args.max_iterations)])
    return _run_command(cmd, env=env)


def _blog_command(args: argparse.Namespace) -> int:
    env = _base_env()
    agents_root = REPO_ROOT / "src" / "agents"
    env["PYTHONPATH"] = str(agents_root) + os.pathsep + env["PYTHONPATH"]
    if args.source_workspace:
        env["BLOG_AGENT_SOURCE_WORKSPACE"] = str(Path(args.source_workspace).expanduser().resolve())

    cmd = [
        sys.executable,
        "-m",
        "blog_agent.scripts.run",
        "--experiment",
        args.experiment,
    ]
    if args.resume:
        cmd.append("--resume")
    return _run_command(cmd, env=env)


def _pipeline_command(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args.config)
    _ensure_config_exists(config_path)
    env = _base_env(config_path=config_path)
    cmd = [
        sys.executable,
        "-m",
        "src.pipeline.run_loop",
        "--config",
        str(config_path),
    ]
    if args.topic:
        cmd.extend(["--topic", args.topic])
    return _run_command(cmd, env=env)


def _doctor_command(args: argparse.Namespace) -> int:
    config_path = _resolve_config_path(args.config)
    _ensure_config_exists(config_path)
    env_file = _load_project_env()

    from src.config import load_config

    config = load_config(str(config_path))
    checks: list[tuple[str, bool, str]] = []
    checks.append(("config", True, str(config_path)))
    checks.append((".env", env_file is not None, str(env_file or DEFAULT_ENV_PATH)))
    checks.append(("node", shutil.which("node") is not None, shutil.which("node") or "missing"))
    checks.append(("npx", shutil.which("npx") is not None, shutil.which("npx") or "missing"))
    checks.append(("uvx", shutil.which("uvx") is not None, shutil.which("uvx") or "missing"))
    checks.append(
        ("graph.db", (REPO_ROOT / "data" / "processed" / "graph.db").exists(), "data/processed/graph.db"),
    )
    checks.append(
        ("all-MiniLM-L6-v2", (REPO_ROOT / "models" / "all-MiniLM-L6-v2").exists(), "models/all-MiniLM-L6-v2")
    )
    checks.append(
        ("bge-m3", (REPO_ROOT / "models" / "bge-m3").exists(), "models/bge-m3"),
    )
    required_env = [
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "SEMANTIC_SCHOLAR_API_KEY",
    ]
    optional_env = [
        "MINIMAX_API_KEY",
        "SERPER_API_KEY",
        "GITHUB_AI_TOKEN",
        "JINA_API_KEY",
        "TAVILY_API_KEY",
        "HF_TOKEN",
    ]

    print("Xcientist doctor")
    print(f"repo: {REPO_ROOT}")
    print(f"workspace: {config.workspace.root}")
    for name, ok, detail in checks:
        status = "OK" if ok else "MISSING"
        print(f"[{status}] {name}: {detail}")

    print("\nEnvironment variables")
    for name in required_env:
        value = os.environ.get(name, "")
        print(f"[{'OK' if value else 'MISSING'}] {name}")
    for name in optional_env:
        value = os.environ.get(name, "")
        print(f"[{'OK' if value else 'OPTIONAL'}] {name}")

    failures = [
        not any(os.environ.get(name) for name in ("OPENAI_API_BASE", "OPENAI_BASE_URL")),
        not os.environ.get("OPENAI_API_KEY"),
        not os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        shutil.which("node") is None,
        shutil.which("npx") is None,
        not (REPO_ROOT / "data" / "processed" / "graph.db").exists(),
        not (REPO_ROOT / "models" / "all-MiniLM-L6-v2").exists(),
        not (REPO_ROOT / "models" / "bge-m3").exists(),
    ]
    return 1 if any(failures) else 0


def _install_mcp_wrappers_command(_: argparse.Namespace) -> int:
    env = _base_env()
    return _run_command(
        ["bash", str(REPO_ROOT / "scripts" / "install_mcp_wrappers.sh")],
        env=env,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_root_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def survey_main() -> int:
    return main(["survey", *sys.argv[1:]])


def idea_main() -> int:
    return main(["idea", *sys.argv[1:]])


def experiment_main() -> int:
    return main(["experiment", *sys.argv[1:]])


def blog_main() -> int:
    return main(["blog", *sys.argv[1:]])


def pipeline_main() -> int:
    return main(["pipeline", *sys.argv[1:]])


def doctor_main() -> int:
    return main(["doctor", *sys.argv[1:]])


def install_mcp_wrappers_main() -> int:
    return main(["install-mcp-wrappers", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
