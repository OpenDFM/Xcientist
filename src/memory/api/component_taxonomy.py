"""
Component Taxonomy: macro_role and component_family definition, extraction,
and structured representation.

macro_role is a stable abstraction of the functional role a component plays
within a method's architecture -- it describes the component's structural
position and purpose in the information flow or optimisation path, *not* its
concrete algorithmic form.

component_family = "{macro_role}.{sub_type}" is a refinement under a given
macro_role that captures the specific implementation style or mechanism
variant.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Macro Roles -- predefined, cross-domain stable semantic framework (10 roles)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MACRO_ROLES: Dict[str, str] = {
    "representation": (
        "Data / feature representation module: responsible for encoding raw "
        "inputs into internal representations, including embedding, encoding, "
        "feature extraction, tokenization, etc."
    ),
    "retrieval": (
        "Retrieval / selection module: selects relevant information from "
        "storage or context, including attention mechanism, memory access, "
        "information retrieval, lookup, etc."
    ),
    "reasoning": (
        "Reasoning / computation module: performs explicit reasoning or "
        "transformations in the representation space, including inference "
        "engine, message passing, graph convolution, chain-of-thought, etc."
    ),
    "objective": (
        "Objective / loss module: defines the optimisation target and "
        "training signal, including loss function, reward signal, contrastive "
        "objective, alignment target, etc."
    ),
    "constraint": (
        "Constraint / regularisation module: imposes structured constraints "
        "to guarantee certain properties, including regularisation, physical "
        "laws, domain invariants, consistency enforcement, etc."
    ),
    "controller": (
        "Scheduling / control module: dynamically decides the computation "
        "path or resource allocation, including gating mechanism, routing, "
        "scheduling, curriculum, conditional execution, etc."
    ),
    "evaluation": (
        "Evaluation / validation module: assesses quality of system outputs "
        "or intermediate results, including metric computation, self-checking, "
        "validation protocol, confidence estimation, etc."
    ),
    "adaptation": (
        "Adaptation / calibration module: adjusts model behaviour in response "
        "to environmental or data changes, including calibration, domain "
        "adaptation, correction, online learning, drift handling, etc."
    ),
    "aggregation": (
        "Aggregation / fusion module: integrates information from multiple "
        "sources or multiple scales, including multi-scale fusion, ensemble, "
        "pooling, cross-modal alignment, etc."
    ),
    "generation": (
        "Generation / output module: produces the final output or intermediate "
        "generation results, including decoder, generator, output head, "
        "prediction layer, sampler, etc."
    ),
}

MACRO_ROLE_NAMES: List[str] = sorted(MACRO_ROLES.keys())

# Keyword -> macro_role heuristic mapping (for inferring macro_role from
# free-text component descriptions).
_KEYWORD_TO_ROLE: Dict[str, str] = {
    # representation
    "embed": "representation", "encoding": "representation",
    "encoder": "representation", "feature": "representation",
    "tokeniz": "representation", "represent": "representation",
    "backbone": "representation", "pretrain": "representation",
    # retrieval
    "attention": "retrieval", "retriev": "retrieval",
    "lookup": "retrieval", "memory access": "retrieval",
    "select": "retrieval", "search": "retrieval",
    "cross-attention": "retrieval", "self-attention": "retrieval",
    # reasoning
    "reason": "reasoning", "infer": "reasoning",
    "message pass": "reasoning", "graph conv": "reasoning",
    "transform": "reasoning", "chain-of-thought": "reasoning",
    "propagat": "reasoning", "diffusion": "reasoning",
    # objective
    "loss": "objective", "objective": "objective",
    "reward": "objective", "contrastive": "objective",
    "alignment target": "objective", "criterion": "objective",
    "likelihood": "objective", "penalty": "objective",
    # constraint
    "regular": "constraint", "constraint": "constraint",
    "invariant": "constraint", "physical law": "constraint",
    "consistency": "constraint", "norm": "constraint",
    "spars": "constraint", "dropout": "constraint",
    # controller
    "gate": "controller", "rout": "controller",
    "schedul": "controller", "curriculum": "controller",
    "condition": "controller", "dispatch": "controller",
    "switch": "controller", "adaptive rate": "controller",
    # evaluation
    "metric": "evaluation", "evaluat": "evaluation",
    "valid": "evaluation", "check": "evaluation",
    "confidence": "evaluation", "monitor": "evaluation",
    # adaptation
    "adapt": "adaptation", "correct": "adaptation",
    "calibrat": "adaptation", "drift": "adaptation",
    "online learn": "adaptation", "fine-tun": "adaptation",
    "domain transfer": "adaptation",
    # aggregation
    "aggregat": "aggregation", "fus": "aggregation",
    "pool": "aggregation", "ensemble": "aggregation",
    "multi-scale": "aggregation", "concat": "aggregation",
    "merge": "aggregation", "cross-modal": "aggregation",
    # generation
    "decode": "generation", "generat": "generation",
    "output": "generation", "predict": "generation",
    "sampl": "generation", "reconstruct": "generation",
    "head": "generation",
}


def infer_macro_role(component_text: str) -> str:
    """Heuristically infer macro_role from a free-text component description.

    Returns the best-matching macro_role name; falls back to
    ``"representation"`` when no match is found (most components involve
    some form of representation processing).
    """
    text = (component_text or "").lower()
    if not text:
        return "representation"

    votes: Dict[str, float] = {}
    for keyword, role in _KEYWORD_TO_ROLE.items():
        if keyword in text:
            votes[role] = votes.get(role, 0.0) + 1.0

    if not votes:
        return "representation"
    return max(votes, key=votes.get)  # type: ignore[arg-type]


def build_component_family(macro_role: str, sub_type: str) -> str:
    """Build a ``component_family`` string: ``{macro_role}.{sub_type}``."""
    macro_role = (macro_role or "representation").strip().lower()
    sub_type = re.sub(r"[^a-z0-9_]", "_", (sub_type or "generic").strip().lower())
    sub_type = re.sub(r"_+", "_", sub_type).strip("_") or "generic"
    if macro_role not in MACRO_ROLES:
        macro_role = "representation"
    return f"{macro_role}.{sub_type}"


def parse_component_family(family: str) -> Tuple[str, str]:
    """Split ``component_family`` into ``(macro_role, sub_type)``."""
    parts = (family or "").split(".", 1)
    macro = parts[0].strip().lower() if parts else "representation"
    sub = parts[1].strip().lower() if len(parts) > 1 else "generic"
    if macro not in MACRO_ROLES:
        macro = "representation"
    return macro, sub


def extract_component_families(
    components: Sequence[str],
    method_text: str = "",
) -> List[Dict[str, str]]:
    """Extract component_family for each component from names and method text.

    The identification process analyses each component's role in the
    computational graph or method description -- checking whether it serves
    representation, retrieval, reasoning, constraint, control, or objective
    purposes -- then assigns a macro_role.  The sub_type is derived from the
    component's specific function, I/O types, connection position, or
    mechanism characteristics, yielding a structurally consistent yet
    expressively flexible component identity.

    Returns:
        ``[{"component": ..., "macro_role": ..., "sub_type": ...,
           "family": ...}, ...]``
    """
    method_lower = (method_text or "").lower()
    results: List[Dict[str, str]] = []

    for comp in components:
        comp_text = f"{comp} {method_lower}"
        macro = infer_macro_role(comp_text)
        # sub_type: normalised form of the component name itself
        raw_sub = re.sub(r"[^a-zA-Z0-9_\- ]", "", comp).strip()
        sub_type = re.sub(r"[\s\-]+", "_", raw_sub).lower() or "generic"
        family = build_component_family(macro, sub_type)
        results.append({
            "component": comp,
            "macro_role": macro,
            "sub_type": sub_type,
            "family": family,
        })

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Context Signature -- compressed query key extracted from parent-node state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Discretisation bucket thresholds
_COVERAGE_THRESHOLDS = (0.35, 0.65)   # low / medium / high
_STABILITY_THRESHOLDS = (0.30, 0.60)  # unstable / moderate / stable
_COST_THRESHOLDS = (0.40, 0.70)       # low / medium / high


def _discretize(value: float, thresholds: Tuple[float, float], labels: Tuple[str, str, str]) -> str:
    if value < thresholds[0]:
        return labels[0]
    if value < thresholds[1]:
        return labels[1]
    return labels[2]


@dataclass
class ContextSignature:
    """Structured query features compressed from the parent-node state.

    Fields are grouped into four categories:

    - **Structural** -- which macro roles are present, component count,
      connection density.
    - **Discretised signals** -- coverage / stability / cost buckets.
    - **Trajectory** -- last main_op, tree depth, defect profile.
    - **Budget / contract** -- budget pressure level.
    """

    # -- structural features --
    macro_roles_present: List[str] = field(default_factory=list)
    component_count: int = 0
    connection_density: str = "moderate"  # sparse | moderate | dense

    # -- discretised signals --
    coverage_bucket: str = "medium"     # low | medium | high
    stability_bucket: str = "moderate"  # unstable | moderate | stable
    cost_bucket: str = "medium"         # low | medium | high

    # -- trajectory features --
    last_main_op: str = ""              # most recent main_op type
    depth: int = 0                      # current tree depth
    defect_profile: List[str] = field(default_factory=list)

    # -- budget / contract context --
    budget_pressure: str = "none"       # none | moderate | tight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "macro_roles_present": self.macro_roles_present,
            "component_count": self.component_count,
            "connection_density": self.connection_density,
            "coverage_bucket": self.coverage_bucket,
            "stability_bucket": self.stability_bucket,
            "cost_bucket": self.cost_bucket,
            "last_main_op": self.last_main_op,
            "depth": self.depth,
            "defect_profile": self.defect_profile,
            "budget_pressure": self.budget_pressure,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextSignature":
        return cls(
            macro_roles_present=list(data.get("macro_roles_present") or []),
            component_count=int(data.get("component_count", 0)),
            connection_density=str(data.get("connection_density", "moderate")),
            coverage_bucket=str(data.get("coverage_bucket", "medium")),
            stability_bucket=str(data.get("stability_bucket", "moderate")),
            cost_bucket=str(data.get("cost_bucket", "medium")),
            last_main_op=str(data.get("last_main_op", "")),
            depth=int(data.get("depth", 0)),
            defect_profile=list(data.get("defect_profile") or []),
            budget_pressure=str(data.get("budget_pressure", "none")),
        )

    def signature_key(self) -> str:
        """Return a deterministic digest string for exact matching or hashing."""
        canonical = "|".join([
            ",".join(sorted(self.macro_roles_present)),
            str(self.component_count),
            self.connection_density,
            self.coverage_bucket,
            self.stability_bucket,
            self.cost_bucket,
            self.last_main_op,
            str(self.depth),
            ",".join(sorted(self.defect_profile)),
            self.budget_pressure,
        ])
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def structural_match_score(self, other: "ContextSignature") -> float:
        """Compute structural match between two ContextSignatures in [0, 1].

        Compares macro-role overlap, component-count proximity,
        connection density, discretised buckets, budget pressure,
        and defect-profile overlap, then returns the arithmetic mean.
        """
        scores: List[float] = []

        # macro_roles overlap (Jaccard)
        s1 = set(self.macro_roles_present)
        s2 = set(other.macro_roles_present)
        if s1 or s2:
            scores.append(len(s1 & s2) / max(len(s1 | s2), 1))
        else:
            scores.append(1.0)

        # component_count proximity (normalised diff)
        max_count = max(self.component_count, other.component_count, 1)
        scores.append(1.0 - abs(self.component_count - other.component_count) / max_count)

        # connection_density exact match
        scores.append(1.0 if self.connection_density == other.connection_density else 0.3)

        # bucket matches
        scores.append(1.0 if self.coverage_bucket == other.coverage_bucket else 0.3)
        scores.append(1.0 if self.stability_bucket == other.stability_bucket else 0.3)
        scores.append(1.0 if self.cost_bucket == other.cost_bucket else 0.3)

        # budget pressure match
        scores.append(1.0 if self.budget_pressure == other.budget_pressure else 0.4)

        # defect profile overlap (Jaccard)
        d1 = set(self.defect_profile)
        d2 = set(other.defect_profile)
        if d1 or d2:
            scores.append(len(d1 & d2) / max(len(d1 | d2), 1))
        else:
            scores.append(1.0)

        return sum(scores) / len(scores) if scores else 0.0


def extract_context_signature(
    components: Sequence[str],
    method_text: str = "",
    evaluation_scores: Optional[Dict[str, float]] = None,
    budget: Optional[Dict[str, Any]] = None,
    last_operator: str = "",
    depth: int = 0,
    defects: Optional[Sequence[str]] = None,
) -> ContextSignature:
    """Extract a ContextSignature (standard query feature format) from
    the parent-node state.

    Args:
        components: List of components in the current idea.
        method_text: Full method text (used to infer macro_roles).
        evaluation_scores: Evaluation dimension -> score mapping
            (novelty, feasibility, etc.).
        budget: Budget dictionary.
        last_operator: Name of the most recently executed skill / operator.
        depth: Tree depth.
        defects: List of current defect labels.
    """
    evaluation_scores = evaluation_scores or {}
    budget = budget or {}
    defects = list(defects or [])

    # structural features
    families = extract_component_families(components, method_text)
    roles_present = sorted(set(f["macro_role"] for f in families))
    comp_count = len(components)

    # connection_density heuristic: component count vs. role diversity
    role_ratio = len(roles_present) / max(comp_count, 1)
    if role_ratio > 0.8:
        connection_density = "sparse"
    elif role_ratio > 0.5:
        connection_density = "moderate"
    else:
        connection_density = "dense"

    # discretised signals
    coverage_raw = (
        evaluation_scores.get("novelty", 0.5) * 0.3
        + evaluation_scores.get("feasibility", 0.5) * 0.3
        + evaluation_scores.get("impact", 0.5) * 0.4
    )
    stability_raw = (
        1.0
        - evaluation_scores.get("risk", 0.3) * 0.5
        - evaluation_scores.get("complexity_penalty", 0.2) * 0.5
    )
    cost_raw = _budget_pressure_score(budget)

    coverage_bucket = _discretize(
        coverage_raw, _COVERAGE_THRESHOLDS, ("low", "medium", "high")
    )
    stability_bucket = _discretize(
        stability_raw, _STABILITY_THRESHOLDS, ("unstable", "moderate", "stable")
    )
    cost_bucket = _discretize(
        cost_raw, _COST_THRESHOLDS, ("low", "medium", "high")
    )

    # budget pressure
    bp_score = _budget_pressure_score(budget)
    if bp_score >= 0.7:
        budget_pressure = "tight"
    elif bp_score >= 0.4:
        budget_pressure = "moderate"
    else:
        budget_pressure = "none"

    return ContextSignature(
        macro_roles_present=roles_present,
        component_count=comp_count,
        connection_density=connection_density,
        coverage_bucket=coverage_bucket,
        stability_bucket=stability_bucket,
        cost_bucket=cost_bucket,
        last_main_op=last_operator,
        depth=depth,
        defect_profile=sorted(set(defects)),
        budget_pressure=budget_pressure,
    )


def _budget_pressure_score(budget: Dict[str, Any]) -> float:
    """Compute a [0, 1] pressure score from a budget dictionary."""
    if not budget:
        return 0.0
    pressure_keys = ["compute", "latency", "cost", "time", "memory"]
    scores: List[float] = []
    for key in pressure_keys:
        val = budget.get(key)
        if val is None:
            continue
        try:
            scores.append(max(0.0, min(1.0, float(val))))
        except (ValueError, TypeError):
            pass
    # also check remaining / total format
    remaining = budget.get("remaining")
    total = budget.get("total")
    if remaining is not None and total is not None:
        try:
            ratio = float(remaining) / max(float(total), 1e-9)
            scores.append(1.0 - max(0.0, min(1.0, ratio)))
        except (ValueError, TypeError):
            pass
    return sum(scores) / max(len(scores), 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main Op type definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# main_op is the component-level primary operation type, aligned with
# AtomicEditOp but abstracted as component-family-oriented action types.
MAIN_OP_TYPES: Dict[str, str] = {
    "add": "Introduce a new component-family instance into the architecture",
    "remove": "Remove a component-family instance from the architecture",
    "replace": "Replace the current component-family instance with another implementation",
    "rewire": "Change the connection pattern between two component families",
    "gate": "Add conditional activation gating to a component family",
    "tune": "Adjust hyper-parameters or configuration of a component family",
    "compose": "Combine multiple component families into a composite module",
    "split": "Decompose a component family into finer-grained sub-modules",
}

MAIN_OP_NAMES: List[str] = sorted(MAIN_OP_TYPES.keys())


def atomic_op_to_main_op(atomic_op: str) -> str:
    """Map an AtomicEditOp string to a main_op type."""
    mapping = {
        "ADD_COMPONENT": "add",
        "REMOVE_COMPONENT": "remove",
        "REPLACE_COMPONENT": "replace",
        "REWIRE": "rewire",
        "GATE_COMPONENT": "gate",
        "ADD_PROTOCOL": "tune",  # protocols change behaviour ~ tuning
    }
    return mapping.get(atomic_op.upper(), "tune")
