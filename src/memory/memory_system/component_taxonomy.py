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

import re
from typing import Dict, List, Sequence, Tuple


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


# Role-specific semantic subtype canonicalisation.
# Goal: keep subtype labels stable enough for symbolic-memory reuse while
# preserving the original component name elsewhere for traceability.
_SUBTYPE_EXACT_ALIASES_BY_ROLE: Dict[str, Dict[str, str]] = {
    "objective": {
        "uncertainty_loss": "uncertainty_aware_loss",
        "uncertainty_weighted_loss": "uncertainty_aware_loss",
        "variance_weighted_loss": "uncertainty_aware_loss",
        "heteroscedastic_nll": "uncertainty_aware_loss",
        "nll_loss": "likelihood_objective",
        "negative_log_likelihood": "likelihood_objective",
        "contrastive_loss": "contrastive_objective",
        "contrastive_objective": "contrastive_objective",
        "triplet_loss": "contrastive_objective",
        "distillation_loss": "distillation_objective",
        "kd_loss": "distillation_objective",
    },
    "controller": {
        "confidence_gate": "gating_module",
        "budget_gate": "gating_module",
        "routing_gate": "gating_module",
        "mixture_of_experts_gate": "gating_module",
        "router": "routing_module",
        "prediction_router": "routing_module",
        "scheduler": "scheduler_module",
    },
    "aggregation": {
        "fusion_module": "fusion_module",
        "cross_modal_fusion": "fusion_module",
        "feature_fusion": "fusion_module",
        "concat_fusion": "fusion_module",
        "multi_scale_fusion": "multi_scale_aggregator",
        "multi_scale_coordinator": "multi_scale_aggregator",
    },
    "evaluation": {
        "confidence_head": "confidence_estimator",
        "confidence_estimator": "confidence_estimator",
        "metric_head": "metric_evaluator",
        "validator": "validation_module",
        "contract_evaluator": "validation_module",
    },
    "adaptation": {
        "calibrator": "calibration_module",
        "temperature_scaler": "calibration_module",
        "drift_corrector": "drift_correction_module",
        "domain_adapter": "domain_adaptation_module",
    },
}


_SUBTYPE_PATTERN_RULES_BY_ROLE: Dict[str, List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]] = {
    # (canonical_name, any_tokens, all_tokens)
    "objective": [
        ("uncertainty_aware_loss", ("uncertainty", "variance", "heteroscedastic"), ("loss", "objective", "criterion", "nll", "penalty")),
        ("contrastive_objective", ("contrastive", "triplet", "pairwise"), ("loss", "objective", "criterion")),
        ("distillation_objective", ("distill", "teacher", "student", "kd"), ("loss", "objective", "criterion")),
        ("regularization_penalty", ("regular", "penalty", "weight_decay", "sparsity"), ("loss", "objective", "constraint", "penalty")),
        ("ranking_objective", ("rank", "margin"), ("loss", "objective", "criterion")),
        ("likelihood_objective", ("likelihood", "nll", "loglik"), ("loss", "objective", "criterion", "nll")),
    ],
    "controller": [
        ("gating_module", ("gate", "gating", "moe"), ()),
        ("routing_module", ("route", "router", "routing", "dispatch", "switch"), ()),
        ("scheduler_module", ("schedul", "curriculum"), ()),
        ("budget_controller", ("budget", "latency", "cost"), ("gate", "controller", "schedul", "router")),
    ],
    "retrieval": [
        ("attention_retriever", ("attention", "cross_attention", "self_attention"), ()),
        ("memory_retriever", ("memory", "lookup", "key_value"), ("retriev", "access", "select", "lookup", "memory")),
        ("search_retriever", ("search", "retrieve"), ()),
    ],
    "representation": [
        ("encoder", ("encoder", "encoding"), ()),
        ("embedding", ("embedding", "embed"), ()),
        ("tokenizer", ("tokenizer", "tokeniz"), ()),
        ("feature_extractor", ("feature", "extractor"), ("feature", "extract")),
        ("backbone", ("backbone",), ()),
    ],
    "reasoning": [
        ("graph_reasoner", ("graph", "gnn", "message_pass"), ("graph", "message", "reason", "infer", "pass")),
        ("inference_module", ("infer", "reason"), ()),
        ("diffusion_module", ("diffusion",), ()),
        ("transform_module", ("transform",), ()),
    ],
    "constraint": [
        ("regularizer", ("regular", "dropout", "norm"), ()),
        ("consistency_constraint", ("consistency", "invariant"), ()),
        ("physics_constraint", ("physics", "physical"), ("constraint", "law", "regular", "invariant")),
        ("sparsity_constraint", ("sparse", "sparsity"), ()),
    ],
    "evaluation": [
        ("confidence_estimator", ("confidence", "uncertainty"), ("estimate", "head", "score", "monitor", "check", "evaluator")),
        ("metric_evaluator", ("metric", "score"), ("eval", "valid", "check", "head", "evaluator")),
        ("validation_module", ("valid", "validator", "check"), ()),
        ("monitoring_module", ("monitor",), ()),
    ],
    "adaptation": [
        ("calibration_module", ("calibrat", "temperature"), ()),
        ("drift_correction_module", ("drift", "correct"), ()),
        ("domain_adaptation_module", ("domain", "adapt", "transfer"), ()),
        ("online_update_module", ("online", "continual"), ("learn", "update", "adapt")),
    ],
    "aggregation": [
        ("multi_scale_aggregator", ("multi_scale", "multiscale"), ()),
        ("fusion_module", ("fusion", "cross_modal"), ()),
        ("pooling_module", ("pool", "pooling"), ()),
        ("ensemble_aggregator", ("ensemble",), ()),
        ("merge_aggregator", ("merge", "concat"), ()),
    ],
    "generation": [
        ("decoder", ("decoder", "decode"), ()),
        ("prediction_head", ("head", "predict", "output"), ()),
        ("sampler", ("sampler", "sample"), ()),
        ("reconstruction_head", ("reconstruct",), ()),
        ("generator", ("generator", "generat"), ()),
    ],
}


def _normalize_sub_type_text(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_\- ]", " ", (text or "")).strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "generic"


def _tokenize_sub_type_text(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", (text or "").lower()) if tok]


def canonicalize_sub_type(
    macro_role: str,
    raw_sub_type: str,
    method_text: str = "",
) -> str:
    """Map a raw subtype string to a role-scoped canonical subtype label.

    This is a lightweight semantic merge layer built on top of alias tables
    and token-pattern rules. It intentionally preserves deterministic behavior
    and falls back to ``raw_sub_type`` when no canonical rule matches.
    """
    role = (macro_role or "representation").strip().lower()
    if role not in MACRO_ROLES:
        role = "representation"

    raw = _normalize_sub_type_text(raw_sub_type)
    alias_map = _SUBTYPE_EXACT_ALIASES_BY_ROLE.get(role, {})
    if raw in alias_map:
        return _normalize_sub_type_text(alias_map[raw])

    # Build a token set from subtype + method text to allow semantic grouping
    # even when the component name itself is abbreviated.
    context_blob = f"{raw} {(method_text or '').lower()}"
    tokens = set(_tokenize_sub_type_text(context_blob))
    compact_blob = context_blob.replace("-", "_")

    for canonical, any_tokens, all_tokens in _SUBTYPE_PATTERN_RULES_BY_ROLE.get(role, []):
        any_hit = False
        for token in any_tokens:
            t = _normalize_sub_type_text(token)
            # match by token or normalized substring for cases like "cross_attention"
            if t in compact_blob or t in tokens:
                any_hit = True
                break
        if not any_hit:
            continue

        if all_tokens:
            all_hit = True
            for token in all_tokens:
                t = _normalize_sub_type_text(token)
                if not (t in compact_blob or t in tokens):
                    all_hit = False
                    break
            if not all_hit:
                continue

        return _normalize_sub_type_text(canonical)

    # Conservative fallback: preserve the original normalized subtype.
    return raw


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
        # raw_sub_type keeps the literal normalized component name for traceability;
        # sub_type is a role-scoped canonical subtype used for better reuse.
        raw_sub = re.sub(r"[^a-zA-Z0-9_\- ]", "", comp).strip()
        raw_sub_type = re.sub(r"[\s\-]+", "_", raw_sub).lower() or "generic"
        sub_type = canonicalize_sub_type(macro, raw_sub_type, method_text)
        family = build_component_family(macro, sub_type)
        family_raw = build_component_family(macro, raw_sub_type)
        results.append({
            "component": comp,
            "macro_role": macro,
            "raw_sub_type": raw_sub_type,
            "sub_type": sub_type,
            "family_raw": family_raw,
            "family": family,
        })

    return results

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
