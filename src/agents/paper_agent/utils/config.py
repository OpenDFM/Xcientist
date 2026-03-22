import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for key in str(path).split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
            continue
        if hasattr(cur, key):
            cur = getattr(cur, key)
            continue
        if hasattr(cur, "get"):
            cur = cur.get(key)
            continue
        return default
    return default if cur is None else cur


paper_cfg = None
experiment_cfg = None
survey_cfg = None
try:
    from src.config import load_config

    cfg_root = load_config()
    survey_cfg = _cfg_get(cfg_root, "survey", {})
    experiment_cfg = _cfg_get(cfg_root, "experiment", {})
    paper_cfg = _cfg_get(cfg_root, "paper", {})
except Exception:
    paper_cfg = {}
    experiment_cfg = {}
    survey_cfg = {}


PAPER_AGENT_ENABLE_TRACING: bool = os.environ.get(
    "PAPER_AGENT_ENABLE_TRACING", "0"
).strip().lower() in ("1", "true", "yes", "y", "on")


OPENAI_API_KEY: Optional[str] = os.environ.get(
    "OPENAI_API_KEY",
    str(_cfg_get(paper_cfg, "api.openai_api_key", "") or _cfg_get(experiment_cfg, "api.openai_api_key", "") or ""),
)
OPENAI_API_BASE: Optional[str] = os.environ.get(
    "OPENAI_API_BASE",
    str(
        _cfg_get(paper_cfg, "api.openai_api_base", "")
        or _cfg_get(experiment_cfg, "api.openai_api_base", "")
        or ""
    ),
)

MINIMAX_API_KEY: Optional[str] = os.environ.get(
    "MINIMAX_API_KEY",
    str(_cfg_get(paper_cfg, "api.minimax_api_key", "") or _cfg_get(experiment_cfg, "api.minimax_api_key", "") or ""),
)
MINIMAX_API_BASE: Optional[str] = str(
    os.environ.get(
        "MINIMAX_API_BASE",
        _cfg_get(paper_cfg, "api.minimax_api_base", "")
        or _cfg_get(experiment_cfg, "api.minimax_api_base", "")
        or "https://api.minimaxi.com/v1",
    )
)
MINIMAX_MODEL_EXTRA_BODY: dict = {"reasoning_split": True}
MINIMAX_MODELS: list = ["MiniMax-M2.1", "MiniMax-M2.5", "MiniMax-Text-01"]

LATEX_TEMPLATE_DIR = str(
    os.environ.get(
        "PAPER_AGENT_LATEX_TEMPLATE_DIR",
        _cfg_get(paper_cfg, "template_dir", "")
        or "/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent/src/agents/paper_agent/latex/ICML2025_Template",
    )
)
PAPER_COMPILE_DOCKER_IMAGE: Optional[str] = os.environ.get(
    "PAPER_COMPILE_DOCKER_IMAGE", "texlive/texlive:latest"
)


_paper_models = dict(_cfg_get(paper_cfg, "models", {}) or {})
_paper_default_model = str(_cfg_get(paper_cfg, "model", "") or _paper_models.get("default") or "MiniMax-M2.1")

PAPER_ARCHITECT_MODEL = str(_paper_models.get("architect") or _paper_default_model)
PAPER_WRITER_MODEL = str(_paper_models.get("writer") or _paper_default_model)
PAPER_REVIEWER_MODEL = str(_paper_models.get("review") or _paper_default_model)
PAPER_ANALYSIS_MODEL = str(_paper_models.get("analysis") or _paper_default_model)
PAPER_LITERATURE_MODEL = str(_paper_models.get("literature") or _paper_default_model)
PAPER_VIZ_MODEL = str(_paper_models.get("viz") or _paper_default_model)
PAPER_VLM_MODEL = str(_paper_models.get("vlm") or "gpt-4o")


def is_minimax_model(model_name: str) -> bool:
    if not model_name:
        return False
    return any(m.lower() in str(model_name).lower() for m in MINIMAX_MODELS)


def get_openai_config(model: Optional[str] = None) -> dict:
    model_name = str(model or "").strip()
    if not model_name:
        model_name = str(PAPER_WRITER_MODEL or "").strip() or "gpt-4o"

    if is_minimax_model(model_name):
        return {
            "api_key": MINIMAX_API_KEY,
            "model_name": model_name,
            "base_url": MINIMAX_API_BASE,
            "extra_body": MINIMAX_MODEL_EXTRA_BODY,
            "is_minimax": True,
        }
    cfg = {
        "api_key": OPENAI_API_KEY,
        "model_name": model_name,
        "is_minimax": False,
    }
    if OPENAI_API_BASE:
        cfg["base_url"] = OPENAI_API_BASE
    return cfg


def get_bash_timeout_seconds() -> int:
    try:
        return int(os.environ.get("PAPER_AGENT_BASH_TIMEOUT_SECONDS", "600"))
    except Exception:
        return 600


def get_semantic_scholar_config() -> Tuple[str, str]:
    base = os.environ.get(
        "SEMANTIC_SCHOLAR_API_BASE",
        str(_cfg_get(paper_cfg, "api.semantic_scholar_api_base", "") or "https://api.semanticscholar.org"),
    ).rstrip("/")
    key = os.environ.get(
        "SEMANTIC_SCHOLAR_API_KEY",
        str(
            _cfg_get(paper_cfg, "api.semantic_scholar_api_key", "")
            or _cfg_get(survey_cfg, "api.semantic_scholar_api_key", "")
            or ""
        ),
    ).strip()
    return base, key


def get_mineru_cmd() -> str:
    return str(os.environ.get("MINERU_CMD", "") or "mineru").strip()


@dataclass
class PaperAgentRunConfig:
    run_name: str
    paper_dir: str
    project_dir: str
    artifact_dir: str
    specs_dir: str
    model: str
    models: Dict[str, str] = field(default_factory=dict)
    experiment_id: str = ""
    experiment_workspace_dir: str = ""
    experiment_idea_md: str = ""
    experiment_specs_dir: str = ""
    experiment_spec_files: List[str] = field(default_factory=list)
    experiment_result_files: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PaperAgentRunConfig":
        return cls(
            run_name=str((d or {}).get("run_name", "") or ""),
            paper_dir=os.path.abspath(str((d or {}).get("paper_dir", "") or "")),
            project_dir=os.path.abspath(str((d or {}).get("project_dir", "") or "")),
            artifact_dir=os.path.abspath(str((d or {}).get("artifact_dir", "") or "")),
            specs_dir=os.path.abspath(str((d or {}).get("specs_dir", "") or "")),
            model=str((d or {}).get("model", "") or "gpt-5.2"),
            models=dict((d or {}).get("models", {}) or {}),
            experiment_id=str((d or {}).get("experiment_id", "") or ""),
            experiment_workspace_dir=os.path.abspath(
                str((d or {}).get("experiment_workspace_dir", "") or "")
            ),
            experiment_idea_md=os.path.abspath(
                str((d or {}).get("experiment_idea_md", "") or "")
            ),
            experiment_specs_dir=os.path.abspath(
                str((d or {}).get("experiment_specs_dir", "") or "")
            ),
            experiment_spec_files=list(
                (d or {}).get("experiment_spec_files", []) or []
            ),
            experiment_result_files=list(
                (d or {}).get("experiment_result_files", []) or []
            ),
        )


_RUNTIME_CFG: Optional[PaperAgentRunConfig] = None


def get_run_config_path() -> str:
    return os.environ.get("PAPER_AGENT_RUN_CONFIG_PATH", "").strip()


def set_runtime_config(cfg: PaperAgentRunConfig) -> None:
    global _RUNTIME_CFG
    _RUNTIME_CFG = cfg


def get_runtime_config() -> PaperAgentRunConfig:
    global _RUNTIME_CFG
    if _RUNTIME_CFG is not None:
        return _RUNTIME_CFG

    path = get_run_config_path()
    if not path:
        raise RuntimeError(
            "PAPER_AGENT_RUN_CONFIG_PATH is not set and runtime config is not initialized"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = PaperAgentRunConfig.from_dict(data)
    _RUNTIME_CFG = cfg
    return cfg


def write_run_config(path: str, cfg: PaperAgentRunConfig) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": cfg.run_name,
                "paper_dir": cfg.paper_dir,
                "project_dir": cfg.project_dir,
                "artifact_dir": cfg.artifact_dir,
                "specs_dir": cfg.specs_dir,
                "model": cfg.model,
                "models": cfg.models,
                "experiment_id": str(cfg.experiment_id or ""),
                "experiment_workspace_dir": str(cfg.experiment_workspace_dir or ""),
                "experiment_idea_md": str(cfg.experiment_idea_md or ""),
                "experiment_specs_dir": str(cfg.experiment_specs_dir or ""),
                "experiment_spec_files": cfg.experiment_spec_files,
                "experiment_result_files": cfg.experiment_result_files,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")


# Workspace roots
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PAPER_AGENT_DIR = os.path.dirname(_THIS_DIR)  # .../paper_agent
_AGENTS_DIR = os.path.dirname(_PAPER_AGENT_DIR)  # .../agents

# Paper agent workspaces (separate from experiment_agent workspaces)
BASE_WORKSPACES_DIR: str = os.environ.get(
    "PAPER_AGENT_WORKSPACES_DIR",
    str(_cfg_get(paper_cfg, "workspace.root", "") or os.path.join(_PAPER_AGENT_DIR, "workspaces")),
).strip()

# Experiment agent workspaces root (read-only input for paper_agent)
EXPERIMENT_WORKSPACES_DIR: str = os.environ.get(
    "EXPERIMENT_AGENT_WORKSPACES_DIR",
    str(
        _cfg_get(paper_cfg, "workspace.experiment_root", "")
        or _cfg_get(experiment_cfg, "workspace.root", "")
        or os.path.join(_AGENTS_DIR, "experiment_agent", "workspaces")
    ),
).strip()


def get_model_config() -> Dict[str, str]:
    return {
        "architect": PAPER_ARCHITECT_MODEL,
        "writer": PAPER_WRITER_MODEL,
        "review": PAPER_REVIEWER_MODEL,
        "analysis": PAPER_ANALYSIS_MODEL,
        "literature": PAPER_LITERATURE_MODEL,
        "viz": PAPER_VIZ_MODEL,
        "vlm": PAPER_VLM_MODEL,
    }


def setup_openai_api(model: Optional[str] = None, verbose: bool = False) -> bool:
    try:
        from httpx import Timeout

        cfg = get_openai_config(model=model)
        api_key = str(cfg.get("api_key", "") or "").strip()
        if not api_key:
            if verbose:
                print(
                    "✗ API key is not set (OPENAI_API_KEY / MINIMAX_API_KEY)"
                )
            return False

        client_kwargs = {
            "api_key": api_key,
            "timeout": Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0),
            "max_retries": 10,
        }
        base_url = str(cfg.get("base_url", "") or "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url
            if verbose:
                print(f"  Using API base: {base_url}")

        client = AsyncOpenAI(**client_kwargs)
        set_default_openai_client(client)
        set_default_openai_api("chat_completions")
        set_tracing_disabled(not PAPER_AGENT_ENABLE_TRACING)
        return True
    except Exception as e:
        if verbose:
            print(f"✗ Failed to setup OpenAI API: {type(e).__name__}: {e}")
        return False


def print_config(experiment_id: str = "") -> None:
    exp = str(experiment_id or "").strip()
    models = get_model_config()
    print("[PaperAgent Config]")
    if exp:
        print(f"  experiment_id: {exp}")
    print(f"  paper_workspaces_dir: {os.path.abspath(BASE_WORKSPACES_DIR)}")
    print(f"  experiment_workspaces_dir: {os.path.abspath(EXPERIMENT_WORKSPACES_DIR)}")
    print(
        f"  latex_template_dir: {LATEX_TEMPLATE_DIR or '(auto-generated minimal template)'}"
    )
    print("  models:")
    for k in [
        "architect",
        "writer",
        "analysis",
        "literature",
        "review",
        "viz",
        "vlm",
    ]:
        print(f"    - {k}: {models.get(k)}")
