"""Preset definitions for tuning Idea Agent search preferences and weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


SCORE_WEIGHT_FIELDS: Tuple[str, ...] = (
    "alignment_weight",
    "complexity_weight",
    "novelty_weight",
    "impact_weight",
    "feasibility_weight",
    "clarity_weight",
    "conciseness_weight",
    "risk_weight",
    "protocol_weight",
)


@dataclass(frozen=True)
class IdeaTastePreset:
    mode: str
    label: str
    summary: str
    weights: Dict[str, float]


IDEA_TASTE_PRESETS: Dict[str, IdeaTastePreset] = {
    "moonshot_inventor": IdeaTastePreset(
        mode="moonshot_inventor",
        label="Moonshot Inventor",
        summary=(
            "Prioritize 0-to-1 mechanisms and outsized upside, while tolerating "
            "higher implementation risk and structural complexity."
        ),
        weights={
            "alignment_weight": 0.30,
            "complexity_weight": 0.10,
            "novelty_weight": 0.45,
            "impact_weight": 0.30,
            "feasibility_weight": 0.10,
            "clarity_weight": 0.10,
            "conciseness_weight": 0.05,
            "risk_weight": 0.10,
            "protocol_weight": 0.08,
        },
    ),
    "bridge_builder": IdeaTastePreset(
        mode="bridge_builder",
        label="Bridge Builder",
        summary=(
            "Favor cross-domain transfer: reusing strong ideas from domain A in "
            "domain B, with less emphasis on raw novelty and more on fit."
        ),
        weights={
            "alignment_weight": 0.45,
            "complexity_weight": 0.10,
            "novelty_weight": 0.10,
            "impact_weight": 0.22,
            "feasibility_weight": 0.28,
            "clarity_weight": 0.18,
            "conciseness_weight": 0.08,
            "risk_weight": 0.18,
            "protocol_weight": 0.24,
        },
    ),
    "steady_engineer": IdeaTastePreset(
        mode="steady_engineer",
        label="Steady Engineer",
        summary=(
            "Optimize for stable, highly feasible ideas that are easy to execute "
            "and less likely to fail."
        ),
        weights={
            "alignment_weight": 0.40,
            "complexity_weight": 0.22,
            "novelty_weight": 0.08,
            "impact_weight": 0.18,
            "feasibility_weight": 0.36,
            "clarity_weight": 0.20,
            "conciseness_weight": 0.16,
            "risk_weight": 0.32,
            "protocol_weight": 0.22,
        },
    ),
    "balanced_scout": IdeaTastePreset(
        mode="balanced_scout",
        label="Balanced Scout",
        summary=(
            "A general-purpose middle ground between novelty, usefulness, "
            "execution quality, and search discipline."
        ),
        weights={
            "alignment_weight": 0.20,
            "complexity_weight": 0.20,
            "novelty_weight": 0.30,
            "impact_weight": 0.25,
            "feasibility_weight": 0.20,
            "clarity_weight": 0.15,
            "conciseness_weight": 0.10,
            "risk_weight": 0.20,
            "protocol_weight": 0.15,
        },
    ),
    "evidence_first": IdeaTastePreset(
        mode="evidence_first",
        label="Evidence First",
        summary=(
            "Prefer ideas that are easy to validate, ablate, and defend with "
            "strong experimental evidence over flashy but brittle novelty."
        ),
        weights={
            "alignment_weight": 0.35,
            "complexity_weight": 0.18,
            "novelty_weight": 0.15,
            "impact_weight": 0.22,
            "feasibility_weight": 0.24,
            "clarity_weight": 0.18,
            "conciseness_weight": 0.12,
            "risk_weight": 0.25,
            "protocol_weight": 0.32,
        },
    ),
}


def list_idea_taste_presets() -> List[Dict[str, Any]]:
    return [
        {
            "mode": preset.mode,
            "label": preset.label,
            "summary": preset.summary,
            "weights": dict(preset.weights),
        }
        for preset in IDEA_TASTE_PRESETS.values()
    ]


def get_idea_taste_preset(mode: Optional[str]) -> Optional[IdeaTastePreset]:
    if mode is None:
        return None

    normalized = str(mode).strip().lower()
    if not normalized:
        return None

    preset = IDEA_TASTE_PRESETS.get(normalized)
    if preset is None:
        available = ", ".join(sorted(IDEA_TASTE_PRESETS))
        raise ValueError(
            f"Unknown mcts.idea_taste_mode '{mode}'. Available presets: {available}"
        )

    missing = [
        field_name for field_name in SCORE_WEIGHT_FIELDS if field_name not in preset.weights
    ]
    if missing:
        raise ValueError(
            f"Idea taste preset '{preset.mode}' is missing score weights: "
            f"{', '.join(missing)}"
        )

    return preset
