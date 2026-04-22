from __future__ import annotations

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer


_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _resolve_model_name_or_path(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value:
        return value

    value = value.replace("${repo_root}", str(_PROJECT_ROOT.resolve()))
    value = value.replace("${workspace}", str((_PROJECT_ROOT / "workspace").resolve()))

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    repo_candidate = (_PROJECT_ROOT / candidate).resolve()
    if repo_candidate.exists():
        return str(repo_candidate)
    return value


def _gpu_candidates_from_torch() -> List[Tuple[str, int]]:
    candidates: List[Tuple[str, int]] = []
    if not torch.cuda.is_available():
        return candidates
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
            candidates.append((f"cuda:{index}", int(free_bytes)))
        except Exception:
            continue
    return candidates


def _gpu_candidates_from_nvidia_smi() -> List[Tuple[str, int]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    candidates: List[Tuple[str, int]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            index_str, free_mib_str = [part.strip() for part in line.split(",", 1)]
            candidates.append((f"cuda:{int(index_str)}", int(free_mib_str) * 1024 * 1024))
        except Exception:
            continue
    return candidates


def get_gpu_candidates() -> List[str]:
    override = os.environ.get("SURVEY_AGENT_DEVICE", "").strip()
    if override:
        return [override]

    ranked = _gpu_candidates_from_torch() or _gpu_candidates_from_nvidia_smi()
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [device for device, _free_bytes in ranked]


def get_preferred_device() -> str:
    candidates = get_gpu_candidates()
    if candidates:
        return candidates[0]
    return "cpu"


def get_cuda_visible_device_value(device: str) -> str | None:
    if not device.startswith("cuda:"):
        return None
    return device.split(":", 1)[1]


@contextmanager
def temporary_cuda_visible_device(device: str):
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_value = get_cuda_visible_device_value(device)
    try:
        if visible_value is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_value
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def load_sentence_transformer_auto(model_name: str, logger=None) -> tuple[SentenceTransformer, str]:
    model_name_or_path = _resolve_model_name_or_path(model_name)
    candidates = get_gpu_candidates()
    attempted: List[str] = []
    last_error: Exception | None = None

    for device in candidates + ["cpu"]:
        attempted.append(device)
        try:
            try:
                model = SentenceTransformer(model_name_or_path, device=device)
            except TypeError:
                model = SentenceTransformer(model_name_or_path)
                model = model.to(device)
            if logger is not None:
                logger.info(f"Loaded SentenceTransformer on {device}")
            return model, device
        except Exception as exc:
            last_error = exc
            if logger is not None:
                logger.warning(f"Failed to load SentenceTransformer on {device}: {exc}")
            if device.startswith("cuda"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    attempted_str = ", ".join(attempted)
    if last_error is not None:
        raise RuntimeError(
            f"Failed to load SentenceTransformer on any candidate device ({attempted_str})"
        ) from last_error
    raise RuntimeError("Failed to load SentenceTransformer: no candidate devices available")
