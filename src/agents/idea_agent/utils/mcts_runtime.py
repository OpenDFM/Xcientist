from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from memory.api.component_taxonomy import ContextSignature, extract_component_families, extract_context_signature
from src.agents.idea_agent.utils.mcts_helpers import clip_text, parse_json_response


class AtomicEditOp(str, Enum):
    ADD_COMPONENT = "ADD_COMPONENT"
    REMOVE_COMPONENT = "REMOVE_COMPONENT"
    REPLACE_COMPONENT = "REPLACE_COMPONENT"
    REWIRE = "REWIRE"
    GATE_COMPONENT = "GATE_COMPONENT"
    ADD_PROTOCOL = "ADD_PROTOCOL"


ATOMIC_OP_VALUES: Set[str] = {op.value for op in AtomicEditOp}
# Descriptions for each atomic edit operation, used in prompt formatting and skill documentation.
ATOMIC_OP_DESCRIPTIONS: Dict[str, str] = {
    AtomicEditOp.ADD_COMPONENT: "Introduce a new module or sub-module into the architecture. ",
    AtomicEditOp.REMOVE_COMPONENT: "Delete an existing module from the architecture. ",
    AtomicEditOp.REPLACE_COMPONENT: "Swap an existing module with a new implementation. ",
    AtomicEditOp.REWIRE: "Change how two components are connected (data flow, gradient path, or API coupling). ",
    AtomicEditOp.GATE_COMPONENT: "Wrap a component with a conditional activation gate (e.g., budget, reliability, or confidence check). ",
    AtomicEditOp.ADD_PROTOCOL: "Attach a validation protocol (regression, ablation, or stress test) to the plan. "
}


def format_op_descriptions() -> str:
    """Return a human-readable reference block describing every atomic edit operation."""
    lines = ["Atomic edit operation reference:"]
    for op in AtomicEditOp:
        desc = ATOMIC_OP_DESCRIPTIONS.get(op, "No description.")
        lines.append(f"  - {op.value}: {desc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Defect Registry – canonical defect tags used across all edit-operator skills.
# The evaluator references this registry to output structured defect tags.
# ---------------------------------------------------------------------------
DEFECT_REGISTRY: Dict[str, str] = {
    # mechanism-commit-innovation / theory-transfer-injection
    "stagnant_novelty": (
        "The idea lacks a genuinely new mechanism; contributions feel incremental "
        "or re-package known techniques without a clear novel insight."
    ),
    "unclear_mechanism": (
        "The core algorithmic mechanism is vaguely described or under-specified, "
        "making it hard to reproduce or evaluate the contribution."
    ),
    "validation_gap": (
        "The experimental protocol is missing critical checks such as ablations, "
        "stress tests, or fair baselines needed to support the claims."
    ),
    # counterfactual-contrast
    "missing_edge_cases": (
        "The method is not tested on rare, adversarial, or boundary-condition inputs "
        "that are necessary to establish robustness."
    ),
    "weak_generalization": (
        "Evidence that the approach transfers across domains, distributions, or "
        "scales is insufficient or absent."
    ),
    "dataset_bias": (
        "Training or evaluation data contains systematic biases that may inflate "
        "reported performance or hide failure modes."
    ),
    # adaptive-constraint-hybridization
    "constraint_drift": (
        "Constraints or regularization terms shift or decay during training, "
        "undermining the intended structural guarantees."
    ),
    "physical_invalidity": (
        "Model outputs violate known physical laws, domain invariants, or "
        "hard constraints that the application requires."
    ),
    "weak_regularization": (
        "Regularization is too loose, causing overfitting, mode collapse, or "
        "uncontrolled capacity growth."
    ),
    # surgical-modularity
    "feature_dumping": (
        "Multiple components or features are added simultaneously without "
        "individual justification, making ablation impossible."
    ),
    "monolithic_design": (
        "The architecture is a single tightly-coupled block, resisting modular "
        "analysis, replacement, or incremental improvement."
    ),
    "harder_to_ablate": (
        "Design choices make it difficult to isolate the effect of any single "
        "component through ablation studies."
    ),
    # data-contract-repair
    "data_quality": (
        "Input data suffers from noise, missing values, mislabelling, or "
        "distribution issues that propagate into model errors."
    ),
    "label_noise": (
        "Ground-truth labels are unreliable, inconsistent, or systematically "
        "corrupted, weakening supervised learning signals."
    ),
    "missing_contracts": (
        "There are no explicit data or evaluation contracts specifying what "
        "inputs, outputs, and invariants must hold."
    ),
    # multi-scale-coordinator
    "scale_mismatch": (
        "The model operates at a single resolution or scale while the problem "
        "requires multi-scale reasoning or aggregation."
    ),
    "coordination_failure": (
        "Multiple sub-modules or branches fail to coordinate their predictions, "
        "causing conflicts, redundancy, or information loss."
    ),
    "latency_bottleneck": (
        "A specific component or data path introduces unacceptable latency, "
        "blocking real-time or large-scale deployment."
    ),
    # self-supervised-corrector
    "systematic_bias": (
        "The model consistently over- or under-predicts in a structured pattern "
        "that a targeted correction could mitigate."
    ),
    "silent_failure": (
        "The system produces confident but wrong outputs without raising any "
        "flag, making errors hard to detect downstream."
    ),
    "drift": (
        "Model performance degrades over time as the data distribution shifts "
        "away from the training regime."
    ),
    # theory-transfer-injection
    "theory_gap": (
        "The method lacks grounding in established theory that could provide "
        "convergence guarantees, error bounds, or interpretability."
    ),
    # evaluation-contract-overhaul
    "evaluation_blindspot": (
        "The evaluation protocol misses important dimensions such as fairness, "
        "calibration, out-of-distribution performance, or efficiency."
    ),
    "weak_accountability": (
        "There is no mechanism to attribute failures to specific components, "
        "data slices, or design decisions."
    ),
    # default fallback used when no context-specific defect is identified
    "unexplored_gap": (
        "No specific defect has been identified yet; the idea space is still "
        "being explored and requires targeted analysis."
    ),
}


def format_defect_registry() -> str:
    """Return a human-readable reference block listing every canonical defect tag and its description."""
    lines = ["Canonical defect tag registry (use ONLY these tags in detected_defects):"]
    for tag, desc in DEFECT_REGISTRY.items():
        lines.append(f"  - {tag}: {desc}")
    return "\n".join(lines)


@dataclass
class ComponentEdit:
    op: AtomicEditOp
    component: str
    target: str = ""
    condition: str = ""
    details: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op.value,
            "component": self.component,
            "target": self.target,
            "condition": self.condition,
            "details": self.details,
            "reason": self.reason,
        }


@dataclass
class ValidationProtocol:
    regression_tests: List[str] = field(default_factory=list)
    ablation_tests: List[str] = field(default_factory=list)
    stress_tests: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regression_tests": self.regression_tests,
            "ablation_tests": self.ablation_tests,
            "stress_tests": self.stress_tests,
        }


@dataclass
class EditPlan:
    skill_name: str
    objective: str
    target_defects: List[str]
    component_edits: List[ComponentEdit]
    validation: ValidationProtocol
    guardrails: List[str]
    memory_refs: List[str]
    estimated_budget_delta: Dict[str, float]
    compile_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "objective": self.objective,
            "target_defects": self.target_defects,
            "component_edits": [edit.to_dict() for edit in self.component_edits],
            "validation": self.validation.to_dict(),
            "guardrails": self.guardrails,
            "memory_refs": self.memory_refs,
            "estimated_budget_delta": self.estimated_budget_delta,
            "compile_notes": self.compile_notes,
        }


@dataclass
class EditOperatorSkill:
    name: str
    description: str
    defects: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    atomic_blueprint: List[str] = field(default_factory=list)
    required_protocols: List[str] = field(default_factory=list)
    avoid_combinations: List[str] = field(default_factory=list)
    execution_logic: List[str] = field(default_factory=list)
    source_path: str = ""

    def to_prompt_line(self) -> str:
        defects = ", ".join(self.defects) if self.defects else "unspecified"
        blueprint = ", ".join(self.atomic_blueprint) if self.atomic_blueprint else "none"
        guardrails = ", ".join(self.guardrails) if self.guardrails else "none"
        exec_logic = " | ".join(self.execution_logic) if self.execution_logic else "none"
        return (
            f"- {self.name}: {self.description} | defects={defects} "
            f"| blueprint={blueprint} | guardrails={guardrails} "
            f"| execution_logic={exec_logic}"
        )


@dataclass
class SkillUsagePrior:
    attempts: int = 0
    successes: int = 0
    reward_ema: float = 0.5
    prior: float = 0.5
    rule_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "reward_ema": self.reward_ema,
            "prior": self.prior,
            "rule_constraints": self.rule_constraints,
        }


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
        sections: List[str] = []
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
        ids: List[str] = []
        for bank in (self.field_knowledge, self.anti_patterns, self.fix_recipes):
            ids.extend(snippet.identifier for snippet in bank)
        return ids


DEFAULT_SKILL_TEMPLATES_PATH = (
    Path(__file__).resolve().parents[1]
    / "agent"
    / "skills"
    / "DEFAULT_SKILL_TEMPLATES.json"
)


def _load_default_skill_templates(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        name = str(key).strip()
        if not name or not isinstance(value, dict):
            continue
        normalized[name] = value
    return normalized


DEFAULT_SKILL_TEMPLATES: Dict[str, Dict[str, Any]] = _load_default_skill_templates(
    DEFAULT_SKILL_TEMPLATES_PATH
)

ANTI_PATTERN_CONSTRAINTS: List[str] = [
    "No feature dumping: every component edit must map to a measured defect.",
    "Always include fair regression + ablation checks; stress test high-risk paths.",
    "Constrain compute/latency budgets with explicit gating for expensive modules.",
    "Prefer mechanism clarity over loosely coupled add-ons.",
]


class SkillCatalog:
    def __init__(self, skill_root: Optional[Path] = None) -> None:
        if skill_root is None:
            skill_root = (
                Path(__file__).resolve().parents[1]
                / "agent"
                / "skills"
                / "edit_operator_skills"
            )
        self.skill_root = skill_root
        self.skills: Dict[str, EditOperatorSkill] = {}
        self.priors: Dict[str, SkillUsagePrior] = {}
        self._load()

    def _load(self) -> None:
        loaded: Dict[str, EditOperatorSkill] = {}
        if self.skill_root.exists():
            for skill_file in sorted(self.skill_root.glob("*/SKILL.md")):
                parsed = self._parse_skill_file(skill_file)
                if parsed:
                    loaded[parsed.name] = parsed
        for name, payload in DEFAULT_SKILL_TEMPLATES.items():
            if name in loaded:
                continue
            loaded[name] = EditOperatorSkill(
                name=name,
                description=payload["description"],
                defects=list(payload.get("defects", [])),
                guardrails=list(payload.get("guardrails", [])),
                atomic_blueprint=list(payload.get("atomic_blueprint", [])),
                required_protocols=list(payload.get("required_protocols", [])),
                avoid_combinations=list(payload.get("avoid_combinations", [])),
                execution_logic=list(payload.get("execution_logic", [])),
                source_path="builtin-default",
            )
        self.skills = loaded
        for name in self.skills:
            self.priors.setdefault(name, SkillUsagePrior())

    def _parse_skill_file(self, path: Path) -> Optional[EditOperatorSkill]:
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _split_frontmatter(text)
        name = str(frontmatter.get("name", "")).strip() or path.parent.name
        description = str(frontmatter.get("description", "")).strip()
        sections = _parse_markdown_sections(body)
        defects = sections.get("defect_tags", []) or sections.get("defects", [])
        guardrails = sections.get("guardrails", [])
        atomic_blueprint = sections.get("atomic_blueprint", [])
        required_protocols = sections.get("required_protocols", [])
        avoid_combinations = sections.get("avoid_combinations", [])
        execution_logic = sections.get("execution_logic", [])

        template = DEFAULT_SKILL_TEMPLATES.get(name, {})
        if not description:
            description = str(template.get("description", ""))
        if not defects:
            defects = list(template.get("defects", []))
        if not guardrails:
            guardrails = list(template.get("guardrails", []))
        if not atomic_blueprint:
            atomic_blueprint = list(template.get("atomic_blueprint", []))
        if not required_protocols:
            required_protocols = list(template.get("required_protocols", []))
        if not execution_logic:
            execution_logic = list(template.get("execution_logic", []))
        if not description:
            return None

        return EditOperatorSkill(
            name=name,
            description=description,
            defects=defects,
            guardrails=guardrails,
            atomic_blueprint=atomic_blueprint,
            required_protocols=required_protocols,
            avoid_combinations=avoid_combinations,
            execution_logic=execution_logic,
            source_path=str(path),
        )

    def list_skills(self) -> List[EditOperatorSkill]:
        return [self.skills[key] for key in sorted(self.skills.keys())]

    def format_for_prompt(self, skills: Optional[Sequence[EditOperatorSkill]] = None) -> str:
        chosen = list(skills) if skills is not None else self.list_skills()
        if not chosen:
            return "No edit-operator skills available."
        op_ref = format_op_descriptions()
        skill_lines = "\n".join(skill.to_prompt_line() for skill in chosen)
        return f"{op_ref}\n\nAvailable edit-operator skills:\n{skill_lines}"

    def select_skills(
        self,
        defect_tags: Sequence[str],
        budget: Dict[str, Any],
        max_children: int,
    ) -> List[EditOperatorSkill]:
        defects = {str(tag).strip().lower() for tag in defect_tags if str(tag).strip()}
        if not defects:
            defects = {"unexplored_gap"}

        scored: List[Tuple[float, EditOperatorSkill]] = []
        budget_tight = _is_budget_tight(budget)
        for skill in self.skills.values():
            skill_defects = {d.lower() for d in skill.defects}
            overlap = len(defects & skill_defects)
            defect_score = overlap / max(1, len(defects))
            prior = self.priors.get(skill.name, SkillUsagePrior()).prior
            gate_score = 0.0
            uses_gate = any(step.startswith("GATE_COMPONENT") for step in skill.atomic_blueprint)
            if budget_tight and uses_gate:
                gate_score = 1.0
            elif not budget_tight:
                gate_score = 0.5
            total = 0.45 * defect_score + 0.40 * prior + 0.15 * gate_score
            scored.append((total, skill))

        scored.sort(key=lambda item: item[0], reverse=True)
        picked = [skill for _, skill in scored[: max(1, max_children)]]
        if not picked:
            return self.list_skills()[: max(1, max_children)]
        return picked

    def compile_plan(
        self,
        skill: EditOperatorSkill,
        parent_title: str,
        parent_components: Sequence[str],
        target_defects: Sequence[str],
        budget: Dict[str, Any],
        memory_refs: Sequence[str],
    ) -> EditPlan:
        parsed_steps = [_parse_blueprint_step(step) for step in skill.atomic_blueprint]
        parsed_steps = [step for step in parsed_steps if step is not None]

        component_edits: List[ComponentEdit] = []
        validation = ValidationProtocol()

        required_protocols = {
            token.lower().strip()
            for token in skill.required_protocols
            if token and token.strip()
        }

        for step in parsed_steps:
            op = step["op"]
            if op == AtomicEditOp.ADD_PROTOCOL.value:
                protocols = step.get("protocols", [])
                for protocol in protocols:
                    required_protocols.add(protocol.lower().strip())
                continue
            component_edits.append(
                ComponentEdit(
                    op=AtomicEditOp(op),
                    component=step.get("component", ""),
                    target=step.get("target", ""),
                    condition=step.get("condition", ""),
                    details=step.get("details", ""),
                )
            )

        if _is_budget_tight(budget):
            has_gate = any(edit.op == AtomicEditOp.GATE_COMPONENT for edit in component_edits)
            if not has_gate:
                gate_target = next(
                    (
                        edit.component
                        for edit in component_edits
                        if edit.op in {AtomicEditOp.ADD_COMPONENT, AtomicEditOp.REPLACE_COMPONENT}
                    ),
                    parent_components[0] if parent_components else "primary_module",
                )
                component_edits.append(
                    ComponentEdit(
                        op=AtomicEditOp.GATE_COMPONENT,
                        component=gate_target,
                        condition="if_compute_or_latency_budget_tight",
                        details="Auto-added budget gate.",
                    )
                )

        if not required_protocols:
            required_protocols = {"regression", "ablation", "stress"}

        for protocol in sorted(required_protocols):
            test_text = _default_protocol_text(protocol, skill.name, parent_title, target_defects)
            if protocol == "regression":
                validation.regression_tests.append(test_text)
            elif protocol == "ablation":
                validation.ablation_tests.append(test_text)
            else:
                validation.stress_tests.append(test_text)
            component_edits.append(
                ComponentEdit(
                    op=AtomicEditOp.ADD_PROTOCOL,
                    component=protocol,
                    details=test_text,
                )
            )

        objective_defect = next(iter(target_defects), "unspecified_defect")
        estimated_budget_delta = _estimate_budget_delta(component_edits)
        plan = EditPlan(
            skill_name=skill.name,
            objective=f"Use {skill.name} to address {objective_defect}",
            target_defects=[str(tag) for tag in target_defects] or skill.defects[:1] or ["unspecified_defect"],
            component_edits=component_edits,
            validation=validation,
            guardrails=list(skill.guardrails),
            memory_refs=[str(ref) for ref in memory_refs][:6],
            estimated_budget_delta=estimated_budget_delta,
            compile_notes=(
                f"Compiled from skill '{skill.name}' with blueprint ops={len(skill.atomic_blueprint)}; "
                f"source={skill.source_path or 'builtin'}"
            ),
        )
        return plan

    def update_prior(
        self,
        skill_name: str,
        reward: float,
        feedback: str,
        failure_modes: Sequence[str],
        success_threshold: float = 0.6,
    ) -> SkillUsagePrior:
        prior = self.priors.setdefault(skill_name, SkillUsagePrior())
        prior.attempts += 1
        clipped_reward = max(0.0, min(1.0, reward))
        if clipped_reward >= max(0.0, min(1.0, success_threshold)):
            prior.successes += 1
        prior.reward_ema = 0.8 * prior.reward_ema + 0.2 * clipped_reward
        beta_mean = (prior.successes + 1.0) / (prior.attempts + 2.0)
        prior.prior = 0.55 * beta_mean + 0.45 * prior.reward_ema

        feedback_l = (feedback or "").lower()
        if "budget" in feedback_l and "gate" not in " ".join(prior.rule_constraints).lower():
            prior.rule_constraints.append("Prefer adding explicit GATE_COMPONENT when budget risk appears.")
        for failure in failure_modes:
            text = str(failure).strip()
            if not text:
                continue
            rule = f"Avoid failure mode: {text}"
            if rule not in prior.rule_constraints:
                prior.rule_constraints.append(rule)
        prior.rule_constraints = prior.rule_constraints[:8]
        return prior


def _split_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text

    lines = stripped.splitlines()
    frontmatter: Dict[str, str] = {}
    if len(lines) < 3:
        return frontmatter, text

    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
        line = lines[idx]
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")

    if end_idx is None:
        return frontmatter, text
    body = "\n".join(lines[end_idx + 1 :])
    return frontmatter, body


def _parse_markdown_sections(body: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            title = line[3:].strip().lower().replace(" ", "_")
            current = title
            sections.setdefault(current, [])
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            sections[current].append(stripped[2:].strip())
        elif re.match(r"^\d+\.\s", stripped):
            # Support numbered lists (e.g. "1. Step description")
            sections[current].append(re.sub(r"^\d+\.\s", "", stripped).strip())
    return sections


def _parse_blueprint_step(step: str) -> Optional[Dict[str, Any]]:
    raw = (step or "").strip()
    if not raw:
        return None
    match = re.match(r"^(?P<op>[A-Z_]+)\((?P<body>.*)\)$", raw)
    if not match:
        return None

    op = match.group("op").strip()
    body = match.group("body").strip()
    if op not in ATOMIC_OP_VALUES:
        return None

    if op == AtomicEditOp.ADD_PROTOCOL.value:
        protocols = [token.strip().lower() for token in body.split(",") if token.strip()]
        return {"op": op, "protocols": protocols or ["regression", "ablation", "stress"]}

    if op == AtomicEditOp.REWIRE.value:
        if "->" in body:
            source, target = [segment.strip() for segment in body.split("->", 1)]
        else:
            source, target = body, "downstream"
        return {
            "op": op,
            "component": source,
            "target": target,
            "details": f"Rewire {source} -> {target}",
        }

    if op == AtomicEditOp.REPLACE_COMPONENT.value:
        if "->" in body:
            old_component, new_component = [segment.strip() for segment in body.split("->", 1)]
        else:
            old_component, new_component = body, f"{body}_replacement"
        return {
            "op": op,
            "component": new_component,
            "target": old_component,
            "details": f"Replace {old_component} with {new_component}",
        }

    if op == AtomicEditOp.GATE_COMPONENT.value:
        parts = [segment.strip() for segment in body.split(",", 1)]
        component = parts[0] if parts else "component"
        condition = parts[1] if len(parts) > 1 else "if_budget_or_risk_requires"
        return {
            "op": op,
            "component": component,
            "condition": condition,
            "details": f"Gate {component} under condition '{condition}'",
        }

    component = body.strip() or "component"
    return {"op": op, "component": component, "details": f"{op} on {component}"}


def _default_protocol_text(
    protocol: str,
    skill_name: str,
    parent_title: str,
    target_defects: Sequence[str],
) -> str:
    defect = next((str(tag) for tag in target_defects if str(tag).strip()), "target defect")
    if protocol == "regression":
        return (
            f"Run regression against parent '{parent_title}' and verify no degradation on core metrics while fixing {defect}."
        )
    if protocol == "ablation":
        return (
            f"Ablate the {skill_name} delta to isolate contribution and confirm defect-level lift on {defect}."
        )
    return (
        f"Stress test {skill_name} under worst-case conditions tied to {defect} and record failure boundaries."
    )


def _is_budget_tight(budget: Dict[str, Any]) -> bool:
    if not isinstance(budget, dict):
        return False
    numeric_values: List[float] = []
    for key in ("compute", "latency", "memory", "token", "wall_clock", "gpu_hours"):
        value = budget.get(key)
        if value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric_values:
        return False
    return any(val <= 1.0 for val in numeric_values)


def _estimate_budget_delta(component_edits: Sequence[ComponentEdit]) -> Dict[str, float]:
    compute = 0.0
    latency = 0.0
    memory = 0.0
    for edit in component_edits:
        if edit.op == AtomicEditOp.ADD_COMPONENT:
            compute += 0.15
            latency += 0.10
            memory += 0.12
        elif edit.op == AtomicEditOp.REMOVE_COMPONENT:
            compute -= 0.12
            latency -= 0.08
            memory -= 0.10
        elif edit.op == AtomicEditOp.REPLACE_COMPONENT:
            compute += 0.05
            latency += 0.04
            memory += 0.05
        elif edit.op == AtomicEditOp.REWIRE:
            compute += 0.02
            latency += 0.02
        elif edit.op == AtomicEditOp.GATE_COMPONENT:
            compute -= 0.05
            latency -= 0.03
        elif edit.op == AtomicEditOp.ADD_PROTOCOL:
            compute += 0.03
            latency += 0.01
    return {
        "compute": round(compute, 4),
        "latency": round(latency, 4),
        "memory": round(memory, 4),
    }


def apply_edit_plan_to_components(
    components: Sequence[str],
    edit_plan: EditPlan,
) -> List[str]:
    ordered = [str(comp) for comp in components if str(comp).strip()]
    existing = list(ordered)

    def _contains(name: str) -> bool:
        return any(name == item for item in existing)

    for edit in edit_plan.component_edits:
        component = edit.component.strip()
        target = edit.target.strip()
        if edit.op == AtomicEditOp.ADD_COMPONENT:
            if component and not _contains(component):
                existing.append(component)
        elif edit.op == AtomicEditOp.REMOVE_COMPONENT:
            if component:
                existing = [item for item in existing if item != component]
        elif edit.op == AtomicEditOp.REPLACE_COMPONENT:
            if target:
                replaced = False
                for idx, item in enumerate(existing):
                    if item == target:
                        existing[idx] = component or target
                        replaced = True
                        break
                if not replaced and component and not _contains(component):
                    existing.append(component)
            elif component and not _contains(component):
                existing.append(component)
        elif edit.op == AtomicEditOp.REWIRE:
            # REWIRE affects topology, not component inventory.
            continue
        elif edit.op == AtomicEditOp.GATE_COMPONENT:
            if component and not _contains(component):
                existing.append(component)
        elif edit.op == AtomicEditOp.ADD_PROTOCOL:
            # Protocols are first-class edits but not structural components.
            continue

    return existing


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


def _dedupe_keep_order_strings(items: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        text = str(item).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _safe_pretty_json(value: Any, pretty_json: Optional[Any] = None) -> str:
    if callable(pretty_json):
        try:
            return str(pretty_json(value))
        except Exception:
            pass
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


def _safe_float_default(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalize_budget_dict(budget: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(budget, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in budget.items():
        if isinstance(value, (int, float)):
            cleaned[str(key)] = float(value)
        else:
            try:
                cleaned[str(key)] = float(value)
            except (TypeError, ValueError):
                cleaned[str(key)] = value
    return cleaned


def apply_budget_delta_to_parent(
    parent_budget: Dict[str, Any],
    delta: Dict[str, Any],
) -> Dict[str, Any]:
    next_budget = normalize_budget_dict(parent_budget)
    for key, val in (delta or {}).items():
        if key not in next_budget:
            next_budget[key] = _safe_float_default(val, 0.0)
            continue
        base = next_budget.get(key)
        if isinstance(base, (int, float)):
            next_budget[key] = round(float(base) + _safe_float_default(val, 0.0), 4)
    return next_budget


def plan_to_method_text(plan: Any) -> str:
    lines: List[str] = []
    for idx, edit in enumerate(getattr(plan, "component_edits", []) or [], start=1):
        op = getattr(getattr(edit, "op", None), "value", getattr(edit, "op", ""))
        target = f" -> {getattr(edit, 'target', '')}" if getattr(edit, "target", "") else ""
        condition = (
            f" [condition: {getattr(edit, 'condition', '')}]"
            if getattr(edit, "condition", "")
            else ""
        )
        details = f"; {getattr(edit, 'details', '')}" if getattr(edit, "details", "") else ""
        lines.append(f"{idx}. {op}({getattr(edit, 'component', '')}{target}){condition}{details}")
    return "\n".join(lines)


def plan_to_experiment_text(plan: Any) -> str:
    blocks: List[str] = []
    validation = getattr(plan, "validation", None)
    regression_tests = getattr(validation, "regression_tests", []) if validation else []
    ablation_tests = getattr(validation, "ablation_tests", []) if validation else []
    stress_tests = getattr(validation, "stress_tests", []) if validation else []
    if regression_tests:
        blocks.append("Regression:\n- " + "\n- ".join(regression_tests))
    if ablation_tests:
        blocks.append("Ablation:\n- " + "\n- ".join(ablation_tests))
    if stress_tests:
        blocks.append("Stress:\n- " + "\n- ".join(stress_tests))
    return "\n\n".join(blocks)


def compute_protocol_score_from_plan(plan: Optional[Dict[str, Any]]) -> float:
    if not plan:
        return 0.0
    validation = plan.get("validation") if isinstance(plan.get("validation"), dict) else {}
    score = 0.0
    if validation.get("regression_tests"):
        score += 1.7
    if validation.get("ablation_tests"):
        score += 1.7
    if validation.get("stress_tests"):
        score += 1.6
    return min(5.0, score)


def memory_bundle_log_payload(bundle: Any) -> Dict[str, Any]:
    def _snippet_payload(snippet: Any) -> Dict[str, Any]:
        return {
            "id": getattr(snippet, "identifier", ""),
            "title": getattr(snippet, "title", ""),
            "detail": getattr(snippet, "detail", ""),
            "tags": list(getattr(snippet, "tags", []) or []),
        }

    return {
        "field_knowledge": [_snippet_payload(s) for s in getattr(bundle, "field_knowledge", []) or []],
        "anti_patterns": [_snippet_payload(s) for s in getattr(bundle, "anti_patterns", []) or []],
        "fix_recipes": [_snippet_payload(s) for s in getattr(bundle, "fix_recipes", []) or []],
    }


def expand_symbolic_memory_log_payload(
    mcts: Any,
    parent_ctx_sig: ContextSignature,
    component_families: List[Dict[str, Any]],
    action_priors: Dict[str, float],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "component_families": component_families,
        "action_priors": {op: round(val, 4) for op, val in action_priors.items()},
        "records_by_family": [],
    }
    seen_record_ids: Set[str] = set()

    for cf in component_families:
        family = str(cf.get("family", "")).strip()
        if not family:
            continue
        try:
            records = mcts.symbolic_memory.retrieve_hierarchical(
                target_family=family,
                context_sig=parent_ctx_sig,
                limit=5,
                threshold=0.05,
                agent_id="idea_agent",
            )
        except Exception as exc:
            payload["records_by_family"].append({"family": family, "error": str(exc), "records": []})
            continue

        family_records: List[Dict[str, Any]] = []
        for score, rec in records or []:
            rec_id = getattr(rec, "id", None)
            if rec_id and rec_id in seen_record_ids:
                continue
            if rec_id:
                seen_record_ids.add(rec_id)
            family_records.append(
                {
                    "id": rec_id,
                    "main_op": getattr(rec, "main_op", ""),
                    "summary": getattr(rec, "summary", ""),
                    "delta_score": round(float(getattr(rec, "delta_score", 0.0)), 4),
                    "confidence": round(float(getattr(rec, "confidence", 0.0)), 4),
                    "match_score": round(float(score), 4),
                    "anti_patterns": list(getattr(rec, "anti_patterns", []) or [])[:3],
                }
            )
        if family_records:
            payload["records_by_family"].append({"family": family, "records": family_records})
    return payload


def simulate_log_payload(evaluation: Any) -> Dict[str, Any]:
    payload = {**evaluation.to_dict(), "composite": evaluation.composite}
    return payload


def materialize_child_state(
    mcts: Any,
    parent_state: Any,
    plan: Any,
    instantiated: Optional[Dict[str, Any]] = None,
    *,
    idea_state_cls: Any,
) -> Any:
    new_components = apply_edit_plan_to_components(parent_state.components, plan)
    next_budget = apply_budget_delta_to_parent(parent_state.budget, plan.estimated_budget_delta)

    inst = instantiated or {}
    title = inst.get("title") or f"{parent_state.title} | {plan.skill_name.replace('-', ' ').title()}"
    abstract = inst.get("abstract") or (
        f"Component-level macro action '{plan.skill_name}' targets defects "
        f"{', '.join(plan.target_defects)} via {len(plan.component_edits)} atomic edits."
    )
    core = inst.get("core_contribution") or plan.objective
    method = inst.get("method") or plan_to_method_text(plan)
    experiments = inst.get("experiments") or plan_to_experiment_text(plan)
    risks = inst.get("risks") or (
        f"Guardrails: {'; '.join(plan.guardrails)} | Budget delta: {plan.estimated_budget_delta}"
    )
    rationale = inst.get("rationale") or plan.compile_notes
    tags = _dedupe_keep_order_strings(list(parent_state.tags) + [plan.skill_name] + list(plan.target_defects))

    return idea_state_cls(
        title=title,
        abstract=abstract,
        core_contribution=core,
        method=method,
        experiments=experiments,
        risks=risks,
        tags=tags,
        operator=plan.skill_name,
        target_defects=plan.target_defects,
        rationale=rationale,
        memory_refs=plan.memory_refs,
        budget=next_budget,
        components=new_components,
        paper_graph_context=parent_state.paper_graph_context,
        edit_plan=plan.to_dict(),
        skill_metrics={
            "skill_prior_before": mcts._skill_prior_for_prompt(plan.skill_name),
            "guardrails": plan.guardrails,
            "constraints": ANTI_PATTERN_CONSTRAINTS,
            "llm_instantiated": bool(inst),
        },
    )


def extract_context_sig_from_node(node: Any) -> ContextSignature:
    eval_scores: Dict[str, float] = {}
    if getattr(node, "evaluation", None) is not None:
        eval_scores = {
            "novelty": node.evaluation.novelty,
            "feasibility": node.evaluation.feasibility,
            "impact": node.evaluation.impact,
            "risk": node.evaluation.risk,
            "complexity_penalty": node.evaluation.complexity_penalty,
        }

    return extract_context_signature(
        components=node.state.components,
        method_text=node.state.method,
        evaluation_scores=eval_scores,
        budget=node.state.budget,
        last_operator=node.state.operator,
        depth=node.depth,
        defects=node.state.target_defects,
    )


def instantiate_skill_plan_for_node(
    mcts: Any,
    plan: Any,
    parent_state: Any,
    bundle: Any,
    *,
    prompt_template: str,
) -> Optional[Dict[str, Any]]:
    component_edits_text = plan_to_method_text(plan)
    validation_text = plan_to_experiment_text(plan)

    prompt = prompt_template.format(
        topic=mcts.topic,
        mature_idea=mcts.mature_idea or "None",
        parent_summary=parent_state.describe(),
        parent_components=", ".join(parent_state.components) if parent_state.components else "None",
        paper_context=mcts.paper_context,
        memory_bundle=bundle.to_prompt_block(),
        skill_name=plan.skill_name,
        plan_objective=plan.objective,
        target_defects=", ".join(plan.target_defects),
        component_edits=component_edits_text,
        validation_protocols=validation_text,
        guardrails="; ".join(plan.guardrails) if plan.guardrails else "None",
    )
    try:
        response = mcts.chat_fn(
            prompt,
            model=mcts.config.generation_model,
            temperature=mcts.config.generation_temperature,
            max_output_tokens=mcts.config.generation_max_tokens,
        )
        payload = parse_json_response(response)
        if isinstance(payload, list):
            payload = payload[0]
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception as exc:
        log_message(
            mcts.logger,
            mcts.log_sink,
            "warning",
            "⚠️  Skill instantiation failed for %s: %s",
            plan.skill_name,
            exc,
        )
        return None


def build_symbolic_eval_hints(mcts: Any, node: Any) -> str:
    component_families = extract_component_families(node.state.components, node.state.method)
    if not component_families:
        return "No symbolic memory hints available."

    ctx_sig = extract_context_sig_from_node(node)
    hints_parts: List[str] = []
    seen_record_ids: Set[str] = set()

    for cf in component_families:
        family = cf.get("family", "")
        if not family:
            continue

        records = mcts.symbolic_memory.retrieve_hierarchical(
            target_family=family,
            context_sig=ctx_sig,
            limit=5,
            threshold=0.05,
            agent_id="idea_agent",
        )
        if not records:
            continue

        family_lines: List[str] = [f"  Component family: {family}"]
        for score, rec in records:
            rec_id = getattr(rec, "id", None)
            if rec_id is not None and rec_id in seen_record_ids:
                continue
            if rec_id is not None:
                seen_record_ids.add(rec_id)

            delta_score = float(getattr(rec, "delta_score", 0.0))
            delta_tag = f"+{delta_score:.2f}" if delta_score >= 0 else f"{delta_score:.2f}"
            line = (
                f"    - [{getattr(rec, 'main_op', '')}] {getattr(rec, 'summary', '')}  "
                f"(delta={delta_tag}, conf={float(getattr(rec, 'confidence', 0.0)):.2f}, score={float(score):.3f})"
            )
            anti_patterns = list(getattr(rec, "anti_patterns", []) or [])
            if anti_patterns:
                line += f"  anti-patterns: {'; '.join(anti_patterns[:3])}"
            family_lines.append(line)
        if len(family_lines) > 1:
            hints_parts.append("\n".join(family_lines))

    if not hints_parts:
        return "No symbolic memory hints available."

    header = (
        "Historical symbolic memory records for the component families "
        "in this idea (positive delta = previously beneficial, "
        "negative delta = previously harmful):"
    )
    return header + "\n" + "\n".join(hints_parts)


def update_skill_prior_from_evaluation(mcts: Any, node: Any, evaluation: Any) -> None:
    skill_name = node.state.operator
    if not skill_name or skill_name == "seed":
        return

    normalized_reward = max(0.0, min(1.0, evaluation.composite / 5.0))
    prior = mcts.skill_catalog.update_prior(
        skill_name=skill_name,
        reward=normalized_reward,
        feedback=evaluation.feedback,
        failure_modes=evaluation.failure_modes,
        success_threshold=mcts.config.skill_prior_success_threshold,
    )
    node.state.skill_metrics["skill_prior_after"] = prior.to_dict()


def extract_mature_idea_components_via_llm(
    mcts: Any,
    mature_idea: str,
    topic: str,
    *,
    prompt_template: str,
    max_components: int,
) -> List[str]:
    prompt = prompt_template.format(mature_idea=mature_idea, topic=topic)
    try:
        response = mcts.chat_fn(
            prompt,
            model=mcts.config.generation_model,
            temperature=0.3,
            max_output_tokens=512,
        )
        payload = parse_json_response(response)
        if isinstance(payload, list):
            payload = payload[0]
        if isinstance(payload, dict):
            raw = payload.get("components", [])
            if isinstance(raw, list):
                components = [str(c).strip() for c in raw if str(c).strip()]
                components = components[:max_components]
                if components:
                    return components
    except Exception as exc:
        log_message(
            mcts.logger,
            mcts.log_sink,
            "warning",
            "⚠️  Component extraction from mature idea failed: %s",
            exc,
        )
    return []


def select_leaf_for_rollout(mcts: Any, node: Any) -> Tuple[Any, List[Any]]:
    current = node
    path = [node]
    while current.children and current.expanded:
        parent = current
        current = max(
            parent.children,
            key=lambda child: child.uct_value(
                parent_visits=parent.visits or 1,
                exploration_constant=mcts.config.exploration_constant,
            ),
        )
        path.append(current)
    return current, path


def expand_node_with_skills(
    mcts: Any,
    node: Any,
    path: List[Any],
    *,
    min_components: int,
    max_components: int,
    pretty_json: Optional[Any] = None,
) -> Tuple[Optional[Any], List[Any]]:
    bundle = mcts.memory_accessor.retrieve_bundle(
        query=(
            f"{mcts.topic}\n{node.state.title}\n"
            f"{node.state.core_contribution}\n"
            f"defects={','.join(node.state.target_defects)}"
        )
    )
    log_message(
        mcts.logger,
        mcts.log_sink,
        "info",
        "[MCTS] Expand: vector_memory\n%s",
        _safe_pretty_json(mcts._memory_bundle_log_payload(bundle), pretty_json),
    )
    # --- Compute symbolic-memory action priors for PUCT boosting ---
    parent_ctx_sig = mcts._extract_context_sig(node)
    component_families = extract_component_families(node.state.components, node.state.method)
    action_priors: Dict[str, float] = {}
    for cf in component_families:
        family = cf.get("family", "")
        if not family:
            continue
        priors = mcts.symbolic_memory.compute_action_priors(
            target_family=family,
            context_sig=parent_ctx_sig,
            limit=20,
            agent_id="idea_agent",
        )
        for op, delta in priors.items():
            if op not in action_priors or delta > action_priors[op]:
                action_priors[op] = delta
    log_message(
        mcts.logger,
        mcts.log_sink,
        "info",
        "[MCTS] Expand: symbolic_memory\n%s",
        _safe_pretty_json(
            mcts._expand_symbolic_memory_log_payload(
                parent_ctx_sig=parent_ctx_sig,
                component_families=component_families,
                action_priors=action_priors,
            ),
            pretty_json,
        ),
    )

    skill_candidates = mcts.skill_catalog.select_skills(
        defect_tags=node.state.target_defects,
        budget=node.state.budget,
        max_children=mcts.config.branching_factor,
    )
    log_message(
        mcts.logger,
        mcts.log_sink,
        "info",
        "[MCTS] Expand: skill_prior\n%s",
        _safe_pretty_json(
            {skill.name: mcts._skill_prior_for_prompt(skill.name) for skill in skill_candidates},
            pretty_json,
        ),
    )

    if action_priors:

        def _skill_prior_boost(sk: Any) -> float:
            ops = {step.split("(")[0].strip().lower() for step in (sk.atomic_blueprint or [])}
            return max((action_priors.get(op, 0.0) for op in ops), default=0.0)

        skill_candidates = sorted(skill_candidates, key=_skill_prior_boost, reverse=True)

    payload_count = len(skill_candidates)
    pre_children = len(node.children)
    new_child: Optional[Any] = None

    idea_node_cls = type(node)
    operator_application_cls = type(node.transformation)
    for skill in skill_candidates:
        plan = mcts.skill_catalog.compile_plan(
            skill=skill,
            parent_title=node.state.title,
            parent_components=node.state.components,
            target_defects=node.state.target_defects,
            budget=node.state.budget,
            memory_refs=bundle.referenced_ids(),
        )
        prior_constraints = mcts.skill_catalog.priors.get(skill.name, SkillUsagePrior()).rule_constraints
        if prior_constraints:
            plan.guardrails = _dedupe_keep_order_strings(plan.guardrails + list(prior_constraints))

        if action_priors:
            plan_ops = set()
            for edit in plan.component_edits:
                op_val = edit.op.value if hasattr(edit.op, "value") else str(edit.op)
                plan_ops.add(op_val.lower().strip())

        current_count = len(node.state.components)
        filtered_edits: List[ComponentEdit] = []
        for edit in plan.component_edits:
            if current_count <= min_components and edit.op == AtomicEditOp.REMOVE_COMPONENT:
                continue
            if current_count >= max_components and edit.op in (
                AtomicEditOp.ADD_COMPONENT,
                AtomicEditOp.GATE_COMPONENT,
            ):
                continue
            if edit.op == AtomicEditOp.ADD_COMPONENT:
                current_count += 1
            elif edit.op == AtomicEditOp.REMOVE_COMPONENT:
                current_count -= 1
            elif edit.op == AtomicEditOp.GATE_COMPONENT:
                current_count += 1
            filtered_edits.append(edit)
        plan.component_edits = filtered_edits

        instantiated = mcts._instantiate_skill_plan(plan, node.state, bundle)
        log_message(
            mcts.logger,
            mcts.log_sink,
            "info",
            "[MCTS] Expand: skill=%s instantiation_result\n%s",
            skill.name,
            _safe_pretty_json(
                instantiated
                if instantiated is not None
                else {"status": "empty", "message": "instantiation returned no output"},
                pretty_json,
            ),
        )

        if instantiated and isinstance(instantiated.get("component_mapping"), dict):
            mapping = instantiated["component_mapping"]
            edit_reasons = instantiated.get("edit_reasons")
            if isinstance(edit_reasons, list):
                for reason_idx, edit in enumerate(plan.component_edits):
                    if reason_idx < len(edit_reasons) and isinstance(edit_reasons[reason_idx], str):
                        edit.reason = edit_reasons[reason_idx]

            for edit in plan.component_edits:
                if edit.component in mapping:
                    edit.component = mapping[edit.component]
                if edit.target and edit.target in mapping:
                    edit.target = mapping[edit.target]
                if edit.op == AtomicEditOp.REWIRE:
                    edit.details = f"Rewire {edit.component} -> {edit.target}"
                elif edit.op == AtomicEditOp.REPLACE_COMPONENT:
                    edit.details = f"Replace {edit.target} with {edit.component}"
                elif edit.op == AtomicEditOp.GATE_COMPONENT:
                    cond = f" under condition '{edit.condition}'" if edit.condition else ""
                    edit.details = f"Gate {edit.component}{cond}"
                elif edit.op == AtomicEditOp.ADD_COMPONENT:
                    edit.details = f"ADD_COMPONENT on {edit.component}"

        protocol_names: List[str] = []
        for edit_idx, edit in enumerate(plan.component_edits):
            if edit.op == AtomicEditOp.ADD_PROTOCOL:
                protocol_names.append(edit.component)
                continue
            op_dict = {
                "op": edit.op.value if hasattr(edit.op, "value") else edit.op,
                "component": edit.component,
                "target": edit.target,
                "condition": edit.condition,
                "details": edit.details or "",
                "reason": edit.reason or "",
            }

        child_state = mcts._materialize_child_state(node.state, plan, instantiated)
        child_node = attach_child(
            node,
            child_state,
            signature_nodes=mcts.signature_nodes,
            id_iter=mcts._id_iter,
            idea_node_cls=idea_node_cls,
            operator_application_cls=operator_application_cls,
            logger=mcts.logger,
            log_sink=mcts.log_sink,
        )
        if child_node is None:
            continue
        cached_eval = get_best_cached_evaluation(child_state.signature, mcts.evaluation_cache)
        if cached_eval:
            child_node.evaluation = cached_eval
        if new_child is None and child_node.visits == 0:
            new_child = child_node

    node.expanded = True

    if new_child is None and node.children:
        new_child = min(node.children, key=lambda c: c.visits)
    if new_child:
        return new_child, path + [new_child]
    return node, path


def simulate_node_value(
    mcts: Any,
    node: Any,
    path: List[Any],
    experiences: List[Dict[str, Any]],
    *,
    idea_evaluation_cls: Any,
    pretty_json: Optional[Any] = None,
) -> Optional[Any]:
    path_summary_text = path_summary(path)
    path_key = path_cache_key(node.state.signature, path_summary_text)

    cached_evaluation = get_cached_evaluation(node.state.signature, path_key, mcts.evaluation_cache)
    if cached_evaluation:
        node.evaluation = cached_evaluation
        node.latest_path_summary = path_summary_text
        log_message(
            mcts.logger,
            mcts.log_sink,
            "info",
            "[MCTS] Simulate (cache hit): node=%s score=%.4f\n%s",
            node.state.title[:80],
            cached_evaluation.composite,
            _safe_pretty_json(mcts._simulate_log_payload(cached_evaluation), pretty_json),
        )
        prev_len = len(experiences)
        maybe_record_experience(
            path_key,
            node,
            cached_evaluation,
            path_summary_text,
            experiences,
            mcts.experience_cache,
            mcts.memory_accessor,
            mcts.config.min_confidence_for_memory,
        )
        if len(experiences) > prev_len:
            experience = experiences[-1]
            
        return cached_evaluation

    symbolic_hints = mcts._build_symbolic_eval_hints(node)
    log_message(
        mcts.logger,
        mcts.log_sink,
        "info",
        "[MCTS] Simulate: symbolic_memory\n%s",
        symbolic_hints,
    )

    eval_idea_payload = node.state.to_payload()
    skill_metrics = eval_idea_payload.get("skill_metrics")
    if isinstance(skill_metrics, dict):
        filtered_metrics = dict(skill_metrics)
        filtered_metrics.pop("skill_prior_before", None)
        filtered_metrics.pop("skill_prior_after", None)
        if filtered_metrics:
            eval_idea_payload["skill_metrics"] = filtered_metrics
        else:
            eval_idea_payload.pop("skill_metrics", None)

    prompt = mcts.evaluation_prompt.format(
        topic=mcts.topic,
        mature_idea=mcts.mature_idea or "None",
        analysis=mcts.analysis_blob,
        paper_context=mcts.paper_context,
        skill_output=json.dumps(node.state.edit_plan, ensure_ascii=False, indent=2)
        if node.state.edit_plan
        else "null",
        edit_plan=json.dumps(node.state.edit_plan, ensure_ascii=False, indent=2)
        if node.state.edit_plan
        else "null",
        idea=json.dumps(eval_idea_payload, ensure_ascii=False, indent=2),
        path_summary=path_summary_text,
        defect_registry=format_defect_registry(),
        symbolic_memory_hints=symbolic_hints,
    )

    try:
        response = mcts.chat_fn(
            prompt,
            model=mcts.config.evaluation_model,
            temperature=mcts.config.evaluation_temperature,
            max_output_tokens=mcts.config.evaluation_max_tokens,
        )
        payload = parse_json_response(response)
        if isinstance(payload, list):
            payload = payload[0]
        evaluation = idea_evaluation_cls.from_payload(
            payload,
            weights={
                "novelty_weight": mcts.config.novelty_weight,
                "impact_weight": mcts.config.impact_weight,
                "feasibility_weight": mcts.config.feasibility_weight,
                "clarity_weight": mcts.config.clarity_weight,
                "conciseness_weight": mcts.config.conciseness_weight,
                "risk_weight": mcts.config.risk_weight,
                "alignment_weight": mcts.config.alignment_weight,
                "complexity_weight": mcts.config.complexity_weight,
                "protocol_weight": mcts.config.protocol_weight,
            },
        )
    except Exception as exc:
        log_message(mcts.logger, mcts.log_sink, "warning", "⚠️  Simulation failed: %s", exc)
        return None

    if evaluation.protocol_score <= 0.0:
        evaluation.protocol_score = mcts._compute_protocol_score(node.state.edit_plan)

    log_message(
        mcts.logger,
        mcts.log_sink,
        "info",
        "[MCTS] Simulate: node=%s score=%.4f\n%s",
        node.state.title[:80],
        evaluation.composite,
        _safe_pretty_json(mcts._simulate_log_payload(evaluation), pretty_json),
    )

    cache_evaluation(node.state.signature, path_key, evaluation, mcts.evaluation_cache)
    node.evaluation = evaluation
    node.latest_path_summary = path_summary_text

    prev_len = len(experiences)
    maybe_record_experience(
        path_key,
        node,
        evaluation,
        path_summary_text,
        experiences,
        mcts.experience_cache,
        mcts.memory_accessor,
        mcts.config.min_confidence_for_memory,
    )

    return evaluation


def backpropagate_rollout(path: List[Any], evaluation: Any) -> None:
    score = evaluation.composite
    for hop in reversed(path):
        hop.visits += 1
        hop.value_sum += score


def reset_search_state(mcts: Any) -> None:
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


def path_summary(path: Sequence[Any], limit: int = 2048) -> str:
    steps: List[str] = []
    for hop in path:
        defects = hop.transformation.defects or ["unspecified"]
        steps.append(f"{hop.state.title} [{hop.transformation.operator}] -> defects {defects}")
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
    idea_pool = context.get("idea_pool") or []
    background = context.get("background_knowledge") or []
    defect_tags = context.get("defect_tags") or ["unexplored_gap"]
    budget = context.get("budget") if isinstance(context.get("budget"), dict) else {}
    if not budget:
        budget = {"compute": 1.0, "latency": 1.0, "memory": 1.0}
    components = context.get("components") if isinstance(context.get("components"), list) else []
    if not components:
        components = [
            "backbone_model",
            "objective",
            "evaluation_harness",
        ]

    if idea_pool:
        latest = idea_pool[-1]
        if isinstance(latest, dict):
            title = latest.get("title", f"{topic} seed idea")
            abstract = latest.get("abstract", "")
            core = latest.get("core_contribution", latest.get("core_contribute", ""))
            method = latest.get("method", latest.get("methodology", ""))
            experiments = latest.get("experiments", latest.get("experiment_design", ""))
            risks = latest.get("risks", latest.get("evaluation", ""))
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
        method = "Synthesize referenced methods and expose unresolved bottlenecks."
        experiments = "Use reported baselines and add defect-oriented checks."
        risks = "Need fairness checks and failure-mode surfacing."
        tags = ["seed"]

    return idea_state_cls(
        title=str(title),
        abstract=str(abstract),
        core_contribution=str(core),
        method=str(method),
        experiments=str(experiments),
        risks=str(risks),
        tags=[str(t) for t in tags] if isinstance(tags, list) else [str(tags)],
        operator="seed",
        target_defects=[str(tag) for tag in defect_tags],
        rationale="Starting point from existing idea pool or analysis.",
        memory_refs=[],
        budget=budget,
        components=components,
        paper_graph_context=str(context.get("paper_context") or ""),
    )


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
            "edit_plan": node.state.edit_plan,
        }
        return experience
    return None
