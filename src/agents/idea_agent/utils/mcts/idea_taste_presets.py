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
    skill_bias: Dict[str, float]
    instantiation_guidance: str


def _weight_total(weights: Dict[str, float]) -> float:
    return sum(float(weights.get(field_name, 0.0)) for field_name in SCORE_WEIGHT_FIELDS)


IDEA_TASTE_PRESETS: Dict[str, IdeaTastePreset] = {
    "moonshot_inventor": IdeaTastePreset(
        mode="moonshot_inventor",
        label="Moonshot Inventor",
        summary=(
            "Prioritize 0-to-1 mechanisms and outsized upside, while tolerating "
            "higher implementation risk and structural complexity."
        ),
        weights={
            "alignment_weight": 0.15,
            "complexity_weight": 0.07,
            "novelty_weight": 0.30,
            "impact_weight": 0.24,
            "feasibility_weight": 0.08,
            "clarity_weight": 0.06,
            "conciseness_weight": 0.03,
            "risk_weight": 0.05,
            "protocol_weight": 0.02,
        },
        skill_bias={
            "mechanism-commit-innovation": 1.0,
            "theory-transfer-injection": 0.9,
            "alternative-path-contrast": 0.4,
            "multi-scale-coordinator": 0.35,
            "hierarchical-decomposition": 0.35,
            "feedback-closed-loop": 0.2,
            "surgical-modularity": 0.25,
            "speculative-execution-with-repair": 0.3,
            "resource-aware-adaptive-path": 0.15,
        },
        instantiation_guidance=(
            "Prefer one bold mechanism-level move with outsized upside. Let the "
            "core novelty live in the task-solving path, not in extra scaffolding."
        ),
    ),
    "bridge_builder": IdeaTastePreset(
        mode="bridge_builder",
        label="Bridge Builder",
        summary=(
            "Favor cross-domain transfer: reusing strong ideas from domain A in "
            "domain B, with less emphasis on raw novelty and more on fit."
        ),
        weights={
            "alignment_weight": 0.17,
            "complexity_weight": 0.05,
            "novelty_weight": 0.19,
            "impact_weight": 0.20,
            "feasibility_weight": 0.14,
            "clarity_weight": 0.06,
            "conciseness_weight": 0.02,
            "risk_weight": 0.09,
            "protocol_weight": 0.08,
        },
        skill_bias={
            "theory-transfer-injection": 1.0,
            "multi-scale-coordinator": 0.65,
            "hierarchical-decomposition": 0.6,
            "alternative-path-contrast": 0.45,
            "feedback-closed-loop": 0.35,
            "mechanism-commit-innovation": 0.25,
            "surgical-modularity": 0.25,
            "resource-aware-adaptive-path": 0.2,
            "speculative-execution-with-repair": 0.15,
        },
        instantiation_guidance=(
            "Emphasize the transferable principle, adaptation point, and fit to the "
            "current domain. Make negative-transfer risks explicit."
        ),
    ),
    "steady_engineer": IdeaTastePreset(
        mode="steady_engineer",
        label="Steady Engineer",
        summary=(
            "Optimize for stable, highly feasible ideas that are easy to execute "
            "and less likely to fail."
        ),
        weights={
            "alignment_weight": 0.17,
            "complexity_weight": 0.04,
            "novelty_weight": 0.19,
            "impact_weight": 0.20,
            "feasibility_weight": 0.16,
            "clarity_weight": 0.08,
            "conciseness_weight": 0.02,
            "risk_weight": 0.08,
            "protocol_weight": 0.06,
        },
        skill_bias={
            "surgical-modularity": 1.0,
            "feedback-closed-loop": 0.75,
            "resource-aware-adaptive-path": 0.7,
            "hierarchical-decomposition": 0.45,
            "alternative-path-contrast": 0.35,
            "mechanism-commit-innovation": 0.3,
            "multi-scale-coordinator": 0.25,
            "speculative-execution-with-repair": 0.2,
            "theory-transfer-injection": 0.15,
        },
        instantiation_guidance=(
            "Favor minimal, well-scoped edits with clean interfaces and clear "
            "validation. Avoid avoidable architectural sprawl."
        ),
    ),
    "ambitious_realist": IdeaTastePreset(
        mode="ambitious_realist",
        label="Ambitious Realist",
        summary=(
            "Default search posture for high-upside ideas: strongly favor novelty "
            "and impact, while keeping enough feasibility, alignment, and risk "
            "control to avoid drifting into empty moonshots."
        ),
        weights={
            "alignment_weight": 0.16,
            "complexity_weight": 0.05,
            "novelty_weight": 0.25,
            "impact_weight": 0.22,
            "feasibility_weight": 0.10,
            "clarity_weight": 0.07,
            "conciseness_weight": 0.03,
            "risk_weight": 0.09,
            "protocol_weight": 0.03,
        },
        skill_bias={
            "mechanism-commit-innovation": 1.0,
            "theory-transfer-injection": 0.75,
            "surgical-modularity": 0.55,
            "multi-scale-coordinator": 0.45,
            "hierarchical-decomposition": 0.4,
            "alternative-path-contrast": 0.35,
            "feedback-closed-loop": 0.35,
            "speculative-execution-with-repair": 0.25,
            "resource-aware-adaptive-path": 0.2,
        },
        instantiation_guidance=(
            "Push for ambitious mechanisms with real upside, but keep the causal "
            "story implementable and defensible."
        ),
    ),
    "evidence_first": IdeaTastePreset(
        mode="evidence_first",
        label="Evidence First",
        summary=(
            "Prefer ideas that are easy to validate, ablate, and defend with "
            "strong experimental evidence over flashy but brittle novelty."
        ),
        weights={
            "alignment_weight": 0.16,
            "complexity_weight": 0.04,
            "novelty_weight": 0.19,
            "impact_weight": 0.18,
            "feasibility_weight": 0.15,
            "clarity_weight": 0.07,
            "conciseness_weight": 0.02,
            "risk_weight": 0.12,
            "protocol_weight": 0.07,
        },
        skill_bias={
            "surgical-modularity": 1.0,
            "feedback-closed-loop": 0.8,
            "alternative-path-contrast": 0.55,
            "resource-aware-adaptive-path": 0.45,
            "hierarchical-decomposition": 0.4,
            "multi-scale-coordinator": 0.35,
            "mechanism-commit-innovation": 0.25,
            "theory-transfer-injection": 0.2,
            "speculative-execution-with-repair": 0.2,
        },
        instantiation_guidance=(
            "Prefer the lightest mechanism that yields strong ablations, stress "
            "tests, and fair comparisons. Do not add novelty that cannot be cleanly validated."
        ),
    ),
}


def list_idea_taste_presets() -> List[Dict[str, Any]]:
    return [
        {
            "mode": preset.mode,
            "label": preset.label,
            "summary": preset.summary,
            "weights": dict(preset.weights),
            "skill_bias": dict(preset.skill_bias),
            "instantiation_guidance": preset.instantiation_guidance,
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

    if not isinstance(preset.skill_bias, dict):
        raise ValueError(f"Idea taste preset '{preset.mode}' must define skill_bias as a dict.")

    total = _weight_total(preset.weights)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Idea taste preset '{preset.mode}' must have score weights summing to 1.0; got {total:.6f}."
        )

    for skill_name, raw_bias in preset.skill_bias.items():
        try:
            bias = float(raw_bias)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Idea taste preset '{preset.mode}' has a non-numeric skill bias for '{skill_name}'."
            ) from exc
        if not 0.0 <= bias <= 1.0:
            raise ValueError(
                f"Idea taste preset '{preset.mode}' has out-of-range skill bias for '{skill_name}': {bias}"
            )

    if not str(preset.instantiation_guidance).strip():
        raise ValueError(
            f"Idea taste preset '{preset.mode}' must define instantiation_guidance."
        )

    return preset
