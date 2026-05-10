"""
Shared MLIP helpers for experiment-agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .manifests import artifact_paths, load_json_file


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def mlip_reference_repo_md() -> str:
    return str(repo_root() / "workspace" / "mlp_seed" / "repos" / "repo.md")


def mlip_dataset_catalog_path() -> str:
    return str(repo_root() / "data" / "prepared" / "mlp" / "datasets.json")


def default_mlip_profile() -> Dict[str, Any]:
    return {
        "profile_id": "mlip_molecular_mechanism_validation",
        "domain": "mlip",
        "mode": "molecular_mechanism_validation",
        "train_datasets": ["QM9x"],
        "evaluation_datasets": ["rMD17", "ISO17", "3BPA"],
        "metric_bindings": [
            "energy_error",
            "force_error",
            "throughput",
            "memory_footprint",
        ],
        "baseline_families": ["MACE", "NequIP", "Allegro"],
        "claim_axes": ["long_range", "efficiency", "predictive_quality"],
        "soft_targets": {
            "policy": "soft_targets_only",
            "expectation": (
                "Mechanism validation should avoid clear regression against a strong local-equivariant "
                "baseline and should improve at least one of transfer, torsional robustness, or efficiency."
            ),
        },
    }


def resolve_mlip_profile(
    workspace_root: str,
    project_root: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    paths = artifact_paths(workspace_root, project_root)
    payload = load_json_file(paths["prepare_target_inventory"]) or {}
    profile = payload.get("mlip_profile") if isinstance(payload, dict) else None
    if isinstance(profile, dict) and profile.get("profile_id"):
        return profile
    return default_mlip_profile()


__all__ = [
    "default_mlip_profile",
    "mlip_dataset_catalog_path",
    "mlip_reference_repo_md",
    "repo_root",
    "resolve_mlip_profile",
]
