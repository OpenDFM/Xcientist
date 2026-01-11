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


PAPER_AGENT_ENABLE_TRACING: bool = os.environ.get(
    "PAPER_AGENT_ENABLE_TRACING", "0"
).strip().lower() in ("1", "true", "yes", "y", "on")


# OpenAI Configuration
OPENAI_API_KEY: Optional[str] = os.environ.get(
    "OPENAI_API_KEY", "sk-BWZ0Kqbk3PvdF0zRFf69B63901B84e85A5B4D8B1AfE27e2e"
)
OPENAI_API_BASE: Optional[str] = os.environ.get(
    "OPENAI_API_BASE", "https://api.xi-ai.cn/v1"
)

MINIMAX_API_KEY: Optional[str] = os.environ.get(
    "MINIMAX_API_KEY",
    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLkvIEiLCJVc2VyTmFtZSI6IuS8gSIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTk4MjcyNjcyNjM0NTA3NTQ0IiwiUGhvbmUiOiIxODk4NTU0MDc2NiIsIkdyb3VwSUQiOiIxOTk4MjcyNjcyNjMwMzEzMjQwIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMTItMTAgMTQ6MTQ6NTMiLCJUb2tlblR5cGUiOjQsImlzcyI6Im1pbmltYXgifQ.hvIJx5NfyV-53iYcS7AMkwooAK4yLv00ZMW0CojFki_S0qXfBECOFozLVcSVcS_-Lbn1ttS6_ZQmuFOZLzZbMz679Svq_ffebftANne4fUQheFrdWMiI48JBvzVH5aDL85cxyLyLU4zfujrE1tpEkfOWddgASMpSzZmK-uiivOOPJqAoMQI76kyZbuVTIIMjXYmsTKsYpmj83ggnpHFT8E2pmXBnQyL_5IRwDRLyN4VKSRUjSRvjo8z4_QE_f1ubGLThJgnCeb0mS5nVtjg9rGcBHmRsvJoTwLKPSRv8lCaEvGTM9U8UVvOcMIt9Y3BgBT2tuUvDXJt-VGAnw3OfhA",
)
MINIMAX_API_BASE: Optional[str] = "https://api.minimaxi.com/v1"
MINIMAX_MODEL_EXTRA_BODY: dict = {"reasoning_split": True}
MINIMAX_MODELS: list = ["MiniMax-M2.1"]

XIAOMI_API_KEY: str = "sk-c8bwnop3bi1nahlzx98ga7o0kqgr8u9h0bpv6zri28hp2x20"
XIAOMI_API_BASE: str = "https://api.xiaomimimo.com/v1/"
XIAOMI_MODELS: list = ["mimo-v2-flash"]

LATEX_TEMPLATE_DIR = "/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent/src/agents/paper_agent/latex/ICML2025_Template"
PAPER_COMPILE_DOCKER_IMAGE: Optional[str] = os.environ.get(
    "PAPER_COMPILE_DOCKER_IMAGE", "texlive/texlive:latest"
)



PAPER_ARCHITECT_MODEL = "MiniMax-M2.1"
PAPER_WRITER_MODEL = "MiniMax-M2.1"
PAPER_REVIEWER_MODEL = "MiniMax-M2.1"
PAPER_ANALYSIS_MODEL = "MiniMax-M2.1"
PAPER_LITERATURE_MODEL = "MiniMax-M2.1"
PAPER_VIZ_MODEL = "MiniMax-M2.1"
PAPER_VLM_MODEL = "gpt-4o"


def is_minimax_model(model_name: str) -> bool:
    if not model_name:
        return False
    return any(m.lower() in str(model_name).lower() for m in MINIMAX_MODELS)


def is_xiaomi_model(model_name: str) -> bool:
    if not model_name:
        return False
    return any(m.lower() in str(model_name).lower() for m in XIAOMI_MODELS)


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
    if is_xiaomi_model(model_name):
        return {
            "api_key": XIAOMI_API_KEY,
            "model_name": model_name,
            "base_url": XIAOMI_API_BASE,
            "is_xiaomi": True,
        }
    cfg = {
        "api_key": OPENAI_API_KEY,
        "model_name": model_name,
        "is_minimax": False,
        "is_xiaomi": False,
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
        "SEMANTIC_SCHOLAR_API_BASE", "https://api.semanticscholar.org"
    ).rstrip("/")
    key = os.environ.get(
        "SEMANTIC_SCHOLAR_API_KEY", "1EzJeomTxpaiYyR5cJbCoaZThZTgFkph707DvYzJ"
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
    "PAPER_AGENT_WORKSPACES_DIR", os.path.join(_PAPER_AGENT_DIR, "workspaces")
).strip()

# Experiment agent workspaces root (read-only input for paper_agent)
EXPERIMENT_WORKSPACES_DIR: str = os.environ.get(
    "EXPERIMENT_AGENT_WORKSPACES_DIR",
    os.path.join(_AGENTS_DIR, "experiment_agent", "workspaces"),
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
                    "✗ API key is not set (OPENAI_API_KEY / MINIMAX_API_KEY / XIAOMI_API_KEY)"
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
