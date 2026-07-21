"""Canonical defect registry shared by MCTS evaluation and skill selection."""

from __future__ import annotations

from typing import Dict


DEFECT_REGISTRY: Dict[str, str] = {
    # mechanism-first innovation
    "stagnant_novelty": (
        "The idea lacks a genuinely new mechanism; contributions feel incremental "
        "or re-package known techniques without a clear novel insight."
    ),
    "unclear_mechanism": (
        "The core mechanism is vaguely described or under-specified, making it "
        "hard to reproduce, reason about, or evaluate the contribution."
    ),
    "validation_gap": (
        "The validation plan is missing key ablations, stress tests, or fair "
        "comparisons needed to support the claims."
    ),
    # alternative-path-contrast
    "brittle_single_path": (
        "The method assumes one dominant operating regime and lacks a structured "
        "fallback or contrastive treatment for rare regimes, failures, or recovery."
    ),
    "rare_regime_failure": (
        "Behavior under boundary conditions, overload, adversarial inputs, or "
        "other rare regimes is weak or unexamined."
    ),
    "weak_fallback_behavior": (
        "Fallback, recovery, or degraded-mode behavior is missing, underspecified, "
        "or ineffective when the primary path breaks down."
    ),
    # modular architecture
    "feature_dumping": (
        "Multiple components or features are added simultaneously without "
        "individual justification, making ablation or attribution impossible."
    ),
    "monolithic_design": (
        "Core responsibilities are trapped inside a single tightly coupled block, "
        "resisting modular analysis, replacement, or incremental improvement."
    ),
    "harder_to_ablate": (
        "Design choices make it difficult to isolate the effect of any single "
        "component through controlled ablation studies."
    ),
    # coordination across scales or layers
    "scale_mismatch": (
        "The solution operates at one scale, layer, or tier while the problem "
        "requires coordination across multiple scales or granularities."
    ),
    "coordination_failure": (
        "Multiple subsystems, branches, or layers fail to maintain a coherent "
        "decision rule, causing conflicts, redundancy, or information loss."
    ),
    "latency_bottleneck": (
        "A component, synchronization point, or coordination rule introduces "
        "unacceptable latency or throughput collapse."
    ),
    # hierarchical decomposition
    "responsibility_entanglement": (
        "Planning, control, execution, or analysis responsibilities are tangled "
        "together instead of being separated into explicit layers or roles."
    ),
    # feedback and adaptation
    "silent_failure": (
        "The system produces wrong or degraded outputs without exposing enough "
        "signal for downstream detection or correction."
    ),
    "drift": (
        "Performance degrades over time or across regimes as workloads, data, or "
        "operating conditions shift away from the design assumptions."
    ),
    "open_loop_fragility": (
        "The system acts in an open-loop manner and cannot adapt when outcomes, "
        "errors, or environment conditions change."
    ),
    # theory-guided reformulation
    "theory_gap": (
        "The approach lacks grounding in a transferable principle, invariant, or "
        "formal lens that could justify the mechanism design."
    ),
    "weak_generalization": (
        "Evidence that the approach transfers across domains, workloads, "
        "distributions, or operating regimes is insufficient or absent."
    ),
    # speculative execution
    "over_conservative_execution": (
        "The system pays full serialization, synchronization, or safety cost on "
        "every path because it lacks a principled optimistic fast path."
    ),
    "rollback_blindspot": (
        "The design lacks explicit detection, rollback, or repair when optimistic "
        "actions misfire or speculative assumptions are violated."
    ),
    # default fallback used when no context-specific defect is identified
    "unexplored_gap": (
        "No specific defect has been identified yet; the idea space is still "
        "being explored and requires targeted analysis."
    ),
}


def format_defect_registry() -> str:
    """Return a prompt-friendly listing of canonical defect tags."""
    lines = ["Canonical defect tag registry (use ONLY these tags in detected_defects):"]
    for tag, desc in DEFECT_REGISTRY.items():
        lines.append(f"  - {tag}: {desc}")
    return "\n".join(lines)
