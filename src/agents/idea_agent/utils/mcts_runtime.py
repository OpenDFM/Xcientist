from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from src.agents.idea_agent.utils.mcts_helpers import clip_text


@dataclass
class EditOperator:
    name: str
    description: str
    defects: List[str]
    guardrails: List[str]


EDIT_OPERATORS: List[EditOperator] = [
    EditOperator(
        name="mechanism-commit-innovation",
        description="Introduce a concrete architectural or algorithmic intervention (new module, coupling, or training signal) and argue how it fixes the defect while outlining the validation harness.",
        defects=["stagnant_novelty", "unclear_mechanism", "validation_gap"],
        guardrails=[
            "must name the exact component being added/rewired and why it targets the defect",
            "must define the success metric and experiment that proves the mechanism works",
            "tie the intervention to at least one risk or failure surfaced earlier",
        ],
    ),
    EditOperator(
        name="counterfactual-contrast",
        description="Prototype counterfactual generators or rare-regime samplers that feed new signals into the learning pipeline, forcing the model to handle unseen physics or boundary cases.",
        defects=["missing_edge_cases", "weak_generalization", "dataset_bias"],
        guardrails=["limit to 1-2 new synthetic channels per iteration", "log how the new sampler plugs into training/eval"],
    ),
    EditOperator(
        name="adaptive-constraint-hybridization",
        description="Hybridize hard/soft constraints or controllers by adding a new auxiliary head, penalty, or differentiable solver coupling that directly enforces domain rules.",
        defects=["constraint_drift", "physical_invalidity", "weak_regularization"],
        guardrails=[
            "clearly state the additional constraint signal and how it is computed",
            "prove it does not explode training cost without justification",
        ],
    ),
    EditOperator(
        name="surgical-modularity",
        description="Split the method into orthogonal, swappable modules and re-solve the weakest block with a new mechanism (e.g., delegate geometry encoder, solver head, or monitor).",
        defects=["feature_dumping", "monolithic_design", "harder_to_ablate"],
        guardrails=["touch only one block", "outline interface contracts and how modules communicate"],
    ),
    EditOperator(
        name="data-contract-repair",
        description="Repair data or supervision contracts (coverage, labeling, alignment) before adding model tricks, potentially by inserting new labeling heads or contract tests.",
        defects=["data_quality", "label_noise", "missing_contracts"],
        guardrails=["state measurable contract tests", "forbid new model components unless the contract gap is proven"],
    ),
    EditOperator(
        name="multi-scale-coordinator",
        description="Introduce a coordinator/controller that fuses predictions from different scales or modalities, committing to a routing, aggregation, or scheduling mechanism.",
        defects=["scale_mismatch", "coordination_failure", "latency_bottleneck"],
        guardrails=["describe routing policy and how conflicts are resolved", "quantify added latency or compute budget"],
    ),
    EditOperator(
        name="self-supervised-corrector",
        description="Attach a corrective model (teacher, diffusion prior, energy head) that learns residuals or invariants without extra labels, producing explicit correction signals.",
        defects=["systematic_bias", "silent_failure", "drift"],
        guardrails=["specify the self-supervised loss and how corrections are injected", "explain how over-correction is prevented"],
    ),
    EditOperator(
        name="theory-transfer-injection",
        description="Port a principled mechanism or constraint from another discipline (control theory, info theory, neuro, geometry) and fuse it as a first-class module or objective.",
        defects=["stagnant_novelty", "theory_gap", "weak_generalization"],
        guardrails=[
            "identify the exact theorem/mechanism you are borrowing and how it plugs into the pipeline",
            "spell out the new capability it enables beyond gating or ensembling baselines",
        ],
    ),
    EditOperator(
        name="evaluation-contract-overhaul",
        description="Redesign the evaluation/training contract (stress datasets, protocol, reward shaping) so that new failure modes are surfaced and optimized.",
        defects=["evaluation_blindspot", "weak_accountability", "missing_contracts"],
        guardrails=[
            "describe concrete datasets/protocols introduced and the defect they expose",
            "clarify how the new contract integrates with training/inference cost ceilings",
        ],
    ),
]

EDIT_OPERATOR_INDEX: Dict[str, EditOperator] = {op.name: op for op in EDIT_OPERATORS}
CONSERVATIVE_OPERATOR_NAMES: Set[str] = {
    "mechanism-commit-innovation",
    "surgical-modularity",
    "data-contract-repair",
    "self-supervised-corrector",
}
MODERATE_OPERATOR_NAMES: Set[str] = {
    "counterfactual-contrast",
    "adaptive-constraint-hybridization",
    "multi-scale-coordinator",
}
AGGRESSIVE_OPERATOR_NAMES: Set[str] = {
    "theory-transfer-injection",
    "evaluation-contract-overhaul",
}

ANTI_PATTERN_CONSTRAINTS = [
    "No feature dumping: every add-on must map to a measured defect.",
    "Always declare baseline + ablation protocols for fairness.",
    "Describe at least one deliberate failure-mode surfacing plan.",
    "Constrain resource usage; note instrumentation for guardrails.",
    "Avoid trivial gating/ensembling tweaks; if an incremental safeguard is unavoidable, tag it 'incremental' and explain why it is temporary.",
    "Ensure at least one child is a moonshot-level mechanism or evaluation-contract overhaul suitable for ICML/NeurIPS novelty expectations.",
]


@dataclass
class MemorySnippet:
    identifier: str
    title: str
    detail: str
    tags: List[str] = field(default_factory=list)

    def to_prompt_line(self) -> str:
        tags_str = f" tags={','.join(self.tags)}" if self.tags else ""
        return f"[{self.identifier}] {self.title}{tags_str}: {self.detail}"


@dataclass
class MemoryBundle:
    field_knowledge: List[MemorySnippet] = field(default_factory=list)
    anti_patterns: List[MemorySnippet] = field(default_factory=list)
    fix_recipes: List[MemorySnippet] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        sections = []
        if self.field_knowledge:
            sections.append("== Field Knowledge ==")
            sections.extend(snippet.to_prompt_line() for snippet in self.field_knowledge)
        if self.anti_patterns:
            sections.append("== Anti-patterns ==")
            sections.extend(snippet.to_prompt_line() for snippet in self.anti_patterns)
        if self.fix_recipes:
            sections.append("== Fix Recipes ==")
            sections.extend(snippet.to_prompt_line() for snippet in self.fix_recipes)
        if not sections:
            return "No validated memory snippets matched. Rely on analysis context only."
        return "\n".join(sections)

    def referenced_ids(self) -> List[str]:
        ids = []
        for bank in (self.field_knowledge, self.anti_patterns, self.fix_recipes):
            ids.extend(snippet.identifier for snippet in bank)
        return ids


@dataclass(frozen=True)
class InvariantSpec:
    inv_id: str
    kind: str
    text: str


@dataclass(frozen=True)
class IdeaContract:
    scope_statement: str
    thesis: str
    core_claims: List[str]
    mechanism_invariants: List[str]
    evaluation_invariants: List[str]
    non_goals: List[str]
    budget_ceiling: Dict[str, Any]
    allowed_axes: List[str]
    contract_id: str = field(init=False)

    def __post_init__(self) -> None:
        canonical = json.dumps(
            {
                "scope_statement": self.scope_statement,
                "thesis": self.thesis,
                "core_claims": self.core_claims,
                "mechanism_invariants": self.mechanism_invariants,
                "evaluation_invariants": self.evaluation_invariants,
                "non_goals": self.non_goals,
                "budget_ceiling": self.budget_ceiling,
                "allowed_axes": self.allowed_axes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        object.__setattr__(
            self, "contract_id", hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "IdeaContract":
        def _list(key: str, limit: int) -> List[str]:
            raw = payload.get(key, [])
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, list):
                return []
            return [clip_text(item, 400) for item in raw if item][:limit]

        budget = payload.get("budget_ceiling") or {}
        if not isinstance(budget, dict):
            budget = {}

        return cls(
            scope_statement=clip_text(payload.get("scope_statement", ""), 800),
            thesis=clip_text(payload.get("thesis", ""), 800),
            core_claims=_list("core_claims", 4),
            mechanism_invariants=_list("mechanism_invariants", 6),
            evaluation_invariants=_list("evaluation_invariants", 6),
            non_goals=_list("non_goals", 8),
            budget_ceiling=budget,
            allowed_axes=_list("allowed_axes", 8),
        )

    def invariants(self) -> List[InvariantSpec]:
        items: List[InvariantSpec] = []
        for idx, text in enumerate(self.mechanism_invariants, start=1):
            items.append(InvariantSpec(inv_id=f"M{idx}", kind="mechanism", text=text))
        for idx, text in enumerate(self.evaluation_invariants, start=1):
            items.append(InvariantSpec(inv_id=f"E{idx}", kind="evaluation", text=text))
        return items

    def invariant_ids(self) -> List[str]:
        return [inv.inv_id for inv in self.invariants()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_statement": self.scope_statement,
            "thesis": self.thesis,
            "core_claims": self.core_claims,
            "mechanism_invariants": self.mechanism_invariants,
            "evaluation_invariants": self.evaluation_invariants,
            "non_goals": self.non_goals,
            "budget_ceiling": self.budget_ceiling,
            "allowed_axes": self.allowed_axes,
            "contract_id": self.contract_id,
        }

    def to_prompt_block(self) -> str:
        lines = [
            f"Scope: {self.scope_statement}",
            f"Thesis: {self.thesis}",
            "Core claims:",
        ]
        for claim in self.core_claims:
            lines.append(f"- {claim}")
        lines.append("Mechanism invariants:")
        for inv in self.invariants():
            if inv.kind != "mechanism":
                continue
            lines.append(f"- {inv.inv_id}: {inv.text}")
        lines.append("Evaluation invariants:")
        for inv in self.invariants():
            if inv.kind != "evaluation":
                continue
            lines.append(f"- {inv.inv_id}: {inv.text}")
        if self.non_goals:
            lines.append("Non-goals:")
            lines.extend(f"- {ng}" for ng in self.non_goals)
        if self.allowed_axes:
            lines.append("Allowed axes:")
            lines.extend(f"- {ax}" for ax in self.allowed_axes)
        if self.budget_ceiling:
            lines.append(f"Budget ceiling: {self.budget_ceiling}")
        return "\n".join(lines)


@dataclass
class SkillDelta:
    intervention: str
    mechanism: str
    implementation_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention": self.intervention,
            "mechanism": self.mechanism,
            "implementation_notes": self.implementation_notes,
        }


@dataclass
class ExperimentPatch:
    regression_tests: List[str]
    ablation_tests: List[str]
    stress_tests: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regression_tests": self.regression_tests,
            "ablation_tests": self.ablation_tests,
            "stress_tests": self.stress_tests,
        }


@dataclass
class InvariantImpact:
    status: str
    reason: str
    compensation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "compensation": self.compensation,
        }


@dataclass
class SkillOutput:
    delta: SkillDelta
    experiment_patch: ExperimentPatch
    risk_patch: List[str]
    introduced_concepts: List[str]
    anchor_mapping: Dict[str, str]
    invariant_impact: Dict[str, InvariantImpact]
    memory_refs: List[str]
    budget_impact: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        if self.raw:
            return self.raw
        return {
            "delta": self.delta.to_dict(),
            "experiment_patch": self.experiment_patch.to_dict(),
            "risk_patch": self.risk_patch,
            "introduced_concepts": self.introduced_concepts,
            "anchor_mapping": self.anchor_mapping,
            "invariant_impact": {
                key: val.to_dict() for key, val in self.invariant_impact.items()
            },
            "memory_refs": self.memory_refs,
            "budget_impact": self.budget_impact,
        }


@dataclass
class SkillValidation:
    ok: bool
    errors: List[str]
    warnings: List[str]
    alignment_score: float = 0.0
    complexity_penalty: float = 0.0
    anchor_coverage: float = 0.0


def log_message(
    logger: Any,
    log_sink: Optional[Any],
    level: str,
    message: str,
    *args: Any,
) -> None:
    log_fn = getattr(logger, level, logger.info)
    try:
        log_fn(message, *args)
    except Exception:
        logger.exception("MCTS logging failure for message: %s", message)
    if log_sink:
        try:
            formatted = message % args if args else message
        except Exception:
            formatted = f"{message} | args={args}"
        try:
            log_sink(level, formatted)
        except Exception as exc:
            logger.debug("MCTS log sink failed: %s", exc)


def reset_search_state(mcts: Any) -> None:
    """
    Drop cached nodes/evaluations between searches so long-running agents
    do not accumulate an ever-growing tree across topics.
    """
    mcts.signature_nodes = {}
    mcts.evaluation_cache = {}
    mcts.experience_cache.clear()
    mcts.trace = []
    mcts._id_iter = itertools.count()


def new_node(
    state: Any,
    depth: int,
    parent: Optional[Any],
    signature_nodes: Dict[str, Any],
    id_iter: Any,
    idea_node_cls: Any,
    operator_application_cls: Any,
) -> Any:
    existing = signature_nodes.get(state.signature)
    if existing:
        return existing
    node = idea_node_cls(
        node_id=next(id_iter),
        state=state,
        depth=depth,
        parent=parent,
        transformation=operator_application_cls(
            operator=state.operator,
            defects=state.target_defects,
            rationale=state.rationale,
            memory_refs=state.memory_refs,
        ),
    )
    signature_nodes[state.signature] = node
    if parent:
        parent.children.append(node)
    return node


def attach_child(
    parent: Any,
    state: Any,
    signature_nodes: Dict[str, Any],
    id_iter: Any,
    idea_node_cls: Any,
    operator_application_cls: Any,
    logger: Any,
    log_sink: Optional[Any] = None,
) -> Optional[Any]:
    child = signature_nodes.get(state.signature)
    if child is None:
        return new_node(
            state,
            depth=parent.depth + 1,
            parent=parent,
            signature_nodes=signature_nodes,
            id_iter=id_iter,
            idea_node_cls=idea_node_cls,
            operator_application_cls=operator_application_cls,
        )
    if child is parent or is_ancestor(parent, child):
        log_message(
            logger,
            log_sink,
            "debug",
            "[MCTS] Skip attaching signature=%s to avoid cycle (parent=%s child=%s).",
            state.signature,
            parent.node_id,
            child.node_id,
        )
        return None
    if child not in parent.children:
        parent.children.append(child)
    return child


def is_ancestor(node: Any, candidate: Any) -> bool:
    cursor: Optional[Any] = node
    while cursor is not None:
        if cursor is candidate:
            return True
        cursor = cursor.parent
    return False


def path_summary(path: Sequence[Any], limit: int = 1024) -> str:
    steps = []
    for hop in path:
        defects = hop.transformation.defects or ["unspecified"]
        steps.append(
            f"{hop.state.title} [{hop.transformation.operator}] -> defects {defects}"
        )
    return clip_text(" | ".join(steps), limit)


def path_cache_key(signature: str, path_summary_text: str) -> str:
    raw = f"{signature}|{path_summary_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_evaluation(
    signature: str,
    path_key: str,
    evaluation_cache: Dict[str, Dict[str, Any]],
) -> Optional[Any]:
    sig_cache = evaluation_cache.get(signature)
    if not sig_cache:
        return None
    return sig_cache.get(path_key)


def cache_evaluation(
    signature: str,
    path_key: str,
    evaluation: Any,
    evaluation_cache: Dict[str, Dict[str, Any]],
) -> None:
    evaluation_cache.setdefault(signature, {})[path_key] = evaluation


def get_best_cached_evaluation(
    signature: str,
    evaluation_cache: Dict[str, Dict[str, Any]],
) -> Optional[Any]:
    sig_cache = evaluation_cache.get(signature)
    if not sig_cache:
        return None
    return max(sig_cache.values(), key=lambda ev: ev.composite)


def maybe_record_experience(
    path_key: str,
    node: Any,
    evaluation: Any,
    path_summary_text: str,
    experiences: List[Dict[str, Any]],
    experience_cache: Set[str],
    memory_accessor: Any,
    min_confidence_for_memory: float,
) -> None:
    if path_key in experience_cache:
        return
    experience = harvest_experience(
        node,
        evaluation,
        path_summary_text,
        min_confidence_for_memory,
    )
    if not experience:
        return
    memory_accessor.persist_experience(experience)
    experiences.append(experience)
    experience_cache.add(path_key)


def build_root_state(
    topic: str,
    context: Dict[str, Any],
    idea_state_cls: Any,
) -> Any:
    mature_idea = (context.get("mature_idea") or "").strip()
    contract = context.get("idea_contract")
    if mature_idea:
        thesis = getattr(contract, "thesis", "") if contract else ""
        core_claims = getattr(contract, "core_claims", []) if contract else []
        mechanism_invariants = getattr(contract, "mechanism_invariants", []) if contract else []
        evaluation_invariants = getattr(contract, "evaluation_invariants", []) if contract else []
        title = thesis or _short_title_from_mature_idea(mature_idea, topic)
        core = thesis or (core_claims[0] if core_claims else "Mature idea core contribution.")
        method = (
            "Preserve mechanism invariants: "
            + "; ".join(mechanism_invariants)
            if mechanism_invariants
            else "Preserve the mature idea mechanism."
        )
        experiments = (
            "Preserve evaluation invariants: "
            + "; ".join(evaluation_invariants)
            if evaluation_invariants
            else "Preserve the mature idea evaluation protocol."
        )
        risks = "Risk of contract drift or mechanism chimera."
        tags = ["mature", "contract"]
        return idea_state_cls(
            title=title,
            abstract=str(mature_idea),
            core_contribution=str(core),
            method=str(method),
            experiments=str(experiments),
            risks=str(risks),
            tags=tags,
            operator="seed",
            target_defects=["contract_anchor"],
            rationale="Rooted in mature idea contract.",
            memory_refs=[],
        )

    idea_pool = context.get("idea_pool") or []
    background = context.get("background_knowledge") or []
    if idea_pool:
        latest = idea_pool[-1]
        if isinstance(latest, dict):
            title = latest.get("title", f"{topic} seed idea")
            abstract = latest.get("abstract", "")
            core = latest.get("core_contribution", "")
            method = latest.get("method", "")
            experiments = latest.get("experiments", "")
            risks = latest.get("risks", latest.get("evaluation", {}))
            tags = latest.get("tags")
        else:
            title = f"{topic} prior idea"
            abstract = str(latest)
            core = abstract
            method = ""
            experiments = ""
            risks = ""
            tags = ["seed"]
    else:
        title = f"{topic} baseline"
        abstract = background[-1] if background else "Kick-off seed idea from analysis."
        core = "Seed idea derived from current analysis and background knowledge."
        method = "Synthesize referenced methods, highlight open limitations."
        experiments = "Use current baselines and publicly reported setups."
        risks = "Need fairness checks and failure-mode surfacing."
        tags = ["seed"]

    return idea_state_cls(
        title=title,
        abstract=str(abstract),
        core_contribution=str(core),
        method=str(method),
        experiments=str(experiments),
        risks=str(risks),
        tags=[str(t) for t in tags] if isinstance(tags, list) else [str(tags)],
        operator="seed",
        target_defects=["unexplored_gap"],
        rationale="Starting point from existing idea pool or analysis.",
        memory_refs=[],
    )


def _short_title_from_mature_idea(mature_idea: str, topic: str) -> str:
    text = (mature_idea or "").strip()
    if not text:
        return f"{topic} mature idea"
    first = re.split(r"[.!?。！？]\s+", text, maxsplit=1)[0].strip()
    if not first:
        first = text
    lowered = first.lower()
    prefixes = [
        "a mature direction in",
        "a mature idea in",
        "a mature idea is",
        "a mature direction is",
        "this work",
        "this idea",
        "we propose",
        "we present",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            first = first[len(prefix) :].lstrip(" :,-")
            lowered = first.lower()
            break
    if " is to " in lowered:
        first = first.split(" is to ", 1)[1].strip()
    if " to " in lowered and len(first.split()) > 14:
        first = first.split(" to ", 1)[0].strip()
    words = first.split()
    if len(words) > 12:
        first = " ".join(words[:12]).strip()
    return first or f"{topic} mature idea"


def parse_child_state(data: Dict[str, Any], idea_state_cls: Any) -> Any:
    def _list(key: str) -> List[str]:
        raw = data.get(key, [])
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, str):
            return [raw]
        return []

    return idea_state_cls(
        title=str(data.get("title", "Untitled idea")).strip(),
        abstract=str(data.get("abstract", "")).strip(),
        core_contribution=str(
            data.get("core_contribution", data.get("core_contribute", ""))
        ).strip(),
        method=str(data.get("method", "")).strip(),
        experiments=str(data.get("experiments", data.get("experiment_design", ""))).strip(),
        risks=str(data.get("risks", "")).strip(),
        tags=_list("tags") or ["mcts-child"],
        operator=str(data.get("operator", "unknown")).strip(),
        target_defects=_list("target_defects"),
        rationale=str(data.get("rationale", "")).strip(),
        memory_refs=_list("memory_refs"),
    )


def fallback_child_payloads(
    node: Any,
    bundle: Any,
    edit_operators: Sequence[Any],
    branching_factor: int,
) -> List[Dict[str, Any]]:
    """
    Deterministically craft child payloads when the language model fails to expand a node.
    Ensures the tree keeps growing instead of aborting the search loop.
    """
    payloads: List[Dict[str, Any]] = []
    referenced_ids = bundle.referenced_ids()
    parent_title = node.state.title
    parent_gap = node.state.core_contribution or node.state.abstract
    base_tags = node.state.tags or []
    for idx, op in enumerate(edit_operators[:branching_factor]):
        payloads.append(
            {
                "title": f"{parent_title} | {op.name.replace('-', ' ').title()} #{idx+1}",
                "abstract": (
                    f"Apply {op.description} to stress {parent_title} against {', '.join(op.defects)}."
                ),
                "core_contribution": (
                    f"Operationalize {op.name} to fix {', '.join(op.defects)} highlighted in '{parent_title}'."
                ),
                "method": (
                    f"Modify the parent method focusing on {parent_gap} via {op.description}."
                ),
                "experiments": (
                    f"Design ablations that validate the {op.name} intervention on the parent idea."
                ),
                "risks": "Heuristic fallback idea; validate with full generation later.",
                "tags": list({*base_tags, op.name}),
                "operator": op.name,
                "target_defects": op.defects,
                "memory_refs": referenced_ids,
            }
        )
    return payloads


def best_candidate(root: Any, candidate_cls: Any) -> Optional[Any]:
    candidates: List[Any] = []
    stack = [root]
    visited: Set[int] = set()
    while stack:
        node = stack.pop()
        if node.node_id in visited:
            continue
        visited.add(node.node_id)
        if node.evaluation:
            candidates.append(candidate_cls(node=node, evaluation=node.evaluation))
        stack.extend(node.children)
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.evaluation.composite)


def pareto_candidates(root: Any, candidate_cls: Any) -> Dict[str, Optional[Any]]:
    by_metric = {
        "novel": lambda ev: ev.novelty,
        "feasible": lambda ev: ev.feasibility,
        "concise": lambda ev: ev.conciseness,
    }
    pareto: Dict[str, Optional[Any]] = {k: None for k in by_metric}
    stack = [root]
    visited_ids: Set[int] = set()
    visited: List[Any] = []
    while stack:
        node = stack.pop()
        if node.node_id in visited_ids:
            continue
        visited_ids.add(node.node_id)
        if node.evaluation:
            visited.append(candidate_cls(node=node, evaluation=node.evaluation))
        stack.extend(node.children)

    for label, scorer in by_metric.items():
        if visited:
            pareto[label] = max(visited, key=lambda c, s=scorer: s(c.evaluation))
    return pareto


def harvest_experience(
    node: Any,
    evaluation: Any,
    path_summary_text: str,
    min_confidence_for_memory: float,
) -> Optional[Dict[str, Any]]:
    if evaluation.confidence > min_confidence_for_memory:
        experience = {
            "defect": ", ".join(node.state.target_defects) or evaluation.defect_fix_summary,
            "action": node.state.operator,
            "lift": round(evaluation.lift_estimate, 2),
            "idea": node.state.title,
            "context": path_summary_text,
            "feedback": evaluation.feedback,
            "tags": node.state.tags + ["defect_fix"],
        }
        return experience
    return None
