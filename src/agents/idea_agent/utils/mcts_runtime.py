from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.agents.idea_agent.utils.mcts_helpers import clip_text


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op.value,
            "component": self.component,
            "target": self.target,
            "condition": self.condition,
            "details": self.details,
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
            "retriever",
            "objective",
            "data_pipeline",
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
