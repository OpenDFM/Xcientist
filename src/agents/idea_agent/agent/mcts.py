from __future__ import annotations

import json
import math
import hashlib
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from tqdm import tqdm

from memory.api.faiss_memory_system_api import FAISSMemorySystem
from memory.api.slot_process_api import SlotProcess
from memory.memory_system.models import SemanticRecord, EpisodicRecord, ProceduralRecord
from memory.memory_system.working_slot import WorkingSlot
from memory.memory_system.utils import (
    _safe_dump_str,
    _multi_thread_run,
)
from agent import get_logger
from src.agents.idea_agent.utils.mcts_helpers import (
    parse_json_response,
    format_analysis_blob,
    format_edit_operators,
)

MAX_IDEA_TEXT = 800
MAX_RATIONALE_TEXT = 600
MAX_TITLE_TEXT = 256
MAX_LIST_ENTRIES = 12
MAX_REF_TEXT = 96
ELLIPSIS = "..."


def _clip_text(value: Any, limit: int = MAX_IDEA_TEXT) -> str:
    text = "" if value is None else str(value).strip()
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)] + ELLIPSIS

module_logger = get_logger()


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


class LongTermMemoryAccessor:
    def __init__(
        self,
        semantic_cfg: Optional[Dict[str, Any]] = None,
        episodic_cfg: Optional[Dict[str, Any]] = None,
        procedural_cfg: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.semantic_cfg = semantic_cfg or {}
        self.episodic_cfg = episodic_cfg or {}
        self.procedural_cfg = procedural_cfg or {}
        self.logger = logger or module_logger
        self._stores: Dict[str, Optional[FAISSMemorySystem]] = {
            "semantic": None,
            "episodic": None,
            "procedural": None,
        }

    def _get_store(self, memory_type: str) -> Optional[FAISSMemorySystem]:
        if memory_type not in self._stores:
            return None
        if self._stores[memory_type] is None:
            cfg = {
                "semantic": self.semantic_cfg,
                "episodic": self.episodic_cfg,
                "procedural": self.procedural_cfg,
            }.get(memory_type, {})
            try:
                # Create the memory store instance
                self._stores[memory_type] = FAISSMemorySystem(
                    memory_type=memory_type,
                    llm_name="mimo-v2-flash",
                    backend="openai",
                    **cfg,
                )
            except Exception as exc:
                self.logger.warning(
                    "⚠️  Unable to initialize %s memory store: %s",
                    memory_type,
                    exc,
                )
                self._stores[memory_type] = None
        return self._stores[memory_type]

    def _query_store(
        self,
        store: FAISSMemorySystem,
        query: str,
        limit: int,
        prefix: str,
    ) -> List[MemorySnippet]:
        snippets: List[MemorySnippet] = []
        try:
            records_with_scores = store.query(
                query_text=query,
                method="embedding",
                limit=limit,
                agent_id="idea_agent",
                threshold=0.4,
            )
        except Exception as exc:
            self.logger.warning("⚠️  Memory query failed (%s): %s", prefix, exc)
            return snippets

        for idx, (_, record) in enumerate(records_with_scores, start=1):
            if not record:
                continue
            identifier = f"{prefix}#{idx}"

            if isinstance(record, SemanticRecord):
                title = record.summary
                detail = record.detail
            elif isinstance(record, EpisodicRecord):
                title = record.summary
                detail = _safe_dump_str(record.detail)
            else:
                title = record.name
                detail = record.description
                
            snippets.append(
                MemorySnippet(
                    identifier=identifier,
                    title=title[:80],
                    detail=str(detail)[:400],
                    tags=list(getattr(record, "tags", []) or []),
                )
            )
        return snippets

    def retrieve_bundle(self, query: str, limit: int = 3) -> MemoryBundle:
        bundle = MemoryBundle()
        semantic = self._get_store("semantic")
        episodic = self._get_store("episodic")
        procedural = self._get_store("procedural")

        if semantic:
            bundle.field_knowledge = self._query_store(
                semantic, query, limit, prefix="Field"
            )
        if episodic:
            bundle.anti_patterns = self._query_store(
                episodic, query, limit, prefix="Pattern"
            )
        if procedural:
            bundle.fix_recipes = self._query_store(
                procedural, query, limit, prefix="Recipe"
            )

        return bundle

    def persist_experience(self, experience: Dict[str, Any], max_workers: int = 20) -> None:
        if not experience:
            return

        semantic = self._get_store("semantic")
        episodic = self._get_store("episodic")
        procedural = self._get_store("procedural")
        if not semantic and not episodic and not procedural:
            self.logger.info("ℹ️ Skipping persistence because semantic store is unavailable.")
            return
        
        slot_process = SlotProcess(llm_name="mimo-v2-flash", llm_backend="openai") # lazy loading
        try:
            # 1. Multi-threaded run for contexts transformation
            working_slots = slot_process.transfer_idea_agent_context_to_working_slots(experience)
            self.logger.info(
                "[MCTS] Transferred experience to working slots (count=%d)",
                len(working_slots),
            )
            # 2. Multi-threaded run for slots filter and route
            _multi_thread_run(slot_process._multi_thread_filter_and_route_slot, working_slots, max_workers)
            # 3. Multi-threaded run for experience persistence
            _multi_thread_run(slot_process._multi_thread_transfer_slot_to_memory, slot_process.routed_slot_container, max_workers)
        except Exception as exc:
            self.logger.warning("⚠️  Failed to persist experience: %s", exc)
            return

        semantic_records: List[SemanticRecord] = []
        episodic_records: List[EpisodicRecord] = []
        procedural_records: List[ProceduralRecord] = []
        for memory in slot_process.memory_dict:
            if memory["memory_type"] == "semantic" and semantic:
                semantic_records.append(semantic.instantiate_sem_record(**memory["input"]))
            elif memory["memory_type"] == "episodic" and episodic:
                episodic_records.append(episodic.instantiate_epi_record(**memory["input"]))
            elif memory["memory_type"] == "procedural" and procedural:
                procedural_records.append(procedural.instantiate_proc_record(**memory["input"]))
        
        if semantic and len(semantic_records) > 0:
            try:
                semantic.add(semantic_records, agent_id="idea_agent")
            except Exception as exc:
                self.logger.warning("⚠️  Failed to persist semantic records: %s", exc)
        if episodic and len(episodic_records) > 0:
            try:
                episodic.add(episodic_records, agent_id="idea_agent")
            except Exception as exc:
                self.logger.warning("⚠️  Failed to persist episodic records: %s", exc)
        if procedural and len(procedural_records) > 0:
            try:
                procedural.add(procedural_records, agent_id="idea_agent")
            except Exception as exc:
                self.logger.warning("⚠️  Failed to persist procedural records: %s", exc)

        self.logger.debug(
            "[MCTS] Persisted records -> semantic=%d | episodic=%d | procedural=%d",
            len(semantic_records),
            len(episodic_records),
            len(procedural_records),
        )


@dataclass
class IdeaState:
    title: str
    abstract: str
    core_contribution: str
    method: str
    experiments: str
    risks: str
    tags: List[str]
    operator: str
    target_defects: List[str]
    rationale: str
    memory_refs: List[str] = field(default_factory=list)
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        self.title = _clip_text(self.title, MAX_TITLE_TEXT)
        self.abstract = _clip_text(self.abstract, MAX_IDEA_TEXT)
        self.core_contribution = _clip_text(self.core_contribution, MAX_IDEA_TEXT)
        self.method = _clip_text(self.method, MAX_IDEA_TEXT)
        self.experiments = _clip_text(self.experiments, MAX_IDEA_TEXT)
        self.risks = _clip_text(self.risks, MAX_IDEA_TEXT)
        self.rationale = _clip_text(self.rationale, MAX_RATIONALE_TEXT)
        self.tags = [_clip_text(tag, 48) for tag in self.tags[:MAX_LIST_ENTRIES]]
        self.target_defects = [
            _clip_text(defect, 48) for defect in self.target_defects[:MAX_LIST_ENTRIES]
        ]
        self.memory_refs = [
            _clip_text(ref, MAX_REF_TEXT) for ref in self.memory_refs[:MAX_LIST_ENTRIES]
        ]
        canonical = "|".join(
            [
                self.title.lower(),
                self.core_contribution.lower(),
                self.method.lower(),
                ",".join(sorted(self.tags)),
            ]
        )
        self.signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def describe(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Core Contribution: {self.core_contribution}\n"
            f"Method: {self.method}\n"
            f"Experiments: {self.experiments}\n"
            f"Risks: {self.risks}\n"
            f"Tags: {', '.join(self.tags)}\n"
            f"Operator: {self.operator} targeting {self.target_defects or ['unspecified']}"
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "core_contribute": self.core_contribution,
            "methodology": self.method,
            "experiment_design": self.experiments,
            "risks": self.risks,
            "tags": self.tags,
            "operator": self.operator,
            "target_defects": self.target_defects,
            "memory_refs": self.memory_refs,
            "rationale": self.rationale,
        }


@dataclass
class IdeaEvaluation:
    novelty: float
    feasibility: float
    clarity: float
    impact: float
    risk: float
    conciseness: float
    confidence: float
    failure_modes: List[str]
    fairness_protocol: str
    feedback: str
    defect_fix_summary: str
    lift_estimate: float

    def __post_init__(self) -> None:
        self.failure_modes = [
            _clip_text(mode, 160) for mode in (self.failure_modes or [])[:MAX_LIST_ENTRIES]
        ]
        self.fairness_protocol = _clip_text(self.fairness_protocol, MAX_IDEA_TEXT)
        self.feedback = _clip_text(self.feedback, MAX_IDEA_TEXT)
        self.defect_fix_summary = _clip_text(self.defect_fix_summary, MAX_IDEA_TEXT)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "IdeaEvaluation":
        def _num(key: str, default: float = 0.0) -> float:
            val = payload.get(key, default)
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def _list(key: str) -> List[str]:
            raw = payload.get(key, [])
            if isinstance(raw, list):
                return [str(x) for x in raw]
            if isinstance(raw, str):
                return [raw]
            return []

        return cls(
            novelty=_num("novelty"),
            feasibility=_num("feasibility"),
            clarity=_num("clarity"),
            impact=_num("impact"),
            risk=_num("risk"),
            conciseness=_num("conciseness"),
            confidence=max(0.0, min(1.0, _num("confidence"))),
            failure_modes=_list("failure_modes"),
            fairness_protocol=str(payload.get("fairness_protocol", "")),
            feedback=str(payload.get("feedback", "")),
            defect_fix_summary=str(payload.get("defect_fix_summary", "")),
            lift_estimate=max(0.0, _num("lift_estimate", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "novelty": self.novelty,
            "feasibility": self.feasibility,
            "clarity": self.clarity,
            "impact": self.impact,
            "risk": self.risk,
            "conciseness": self.conciseness,
            "confidence": self.confidence,
            "failure_modes": self.failure_modes,
            "fairness_protocol": self.fairness_protocol,
            "feedback": self.feedback,
            "defect_fix_summary": self.defect_fix_summary,
            "lift_estimate": self.lift_estimate,
        }

    @property
    def composite(self) -> float:
        positive = (
            0.30 * self.novelty
            + 0.25 * self.impact
            + 0.20 * self.feasibility
            + 0.15 * self.clarity
            + 0.10 * self.conciseness
        )
        penalty = 0.2 * self.risk
        return positive - penalty


@dataclass
class OperatorApplication:
    operator: str
    defects: List[str]
    rationale: str
    memory_refs: List[str]

    def __post_init__(self) -> None:
        self.operator = _clip_text(self.operator, 80)
        self.defects = [_clip_text(defect, 48) for defect in self.defects[:MAX_LIST_ENTRIES]]
        self.rationale = _clip_text(self.rationale, MAX_RATIONALE_TEXT)
        self.memory_refs = [
            _clip_text(ref, MAX_REF_TEXT) for ref in self.memory_refs[:MAX_LIST_ENTRIES]
        ]


@dataclass
class IdeaNode:
    node_id: int
    state: IdeaState
    depth: int
    parent: Optional["IdeaNode"]
    transformation: OperatorApplication
    children: List["IdeaNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    evaluation: Optional[IdeaEvaluation] = None
    latest_path_summary: str = ""

    def uct_value(self, parent_visits: int, exploration_constant: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.value_sum / self.visits
        explore = exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / self.visits
        )
        return exploit + explore

    def path(self) -> List["IdeaNode"]:
        chain = []
        node: Optional[IdeaNode] = self
        while node:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    def path_summary(self) -> str:
        if self.latest_path_summary:
            return self.latest_path_summary
        steps = []
        for hop in self.path():
            steps.append(
                f"{hop.state.title} [{hop.transformation.operator}] -> defects {hop.transformation.defects}"
            )
        return " | ".join(steps)


@dataclass
class MCTSConfig:
    max_iterations = 2
    max_depth = 4
    branching_factor: int = 3
    exploration_constant: float = 1.15
    generation_model: str = "mimo-v2-flash"
    evaluation_model: str = "mimo-v2-flash"
    generation_temperature: float = 0.65
    evaluation_temperature: float = 0.0
    min_confidence_for_memory: float = 0.6
    pareto_top_k: int = 5


@dataclass
class SearchCandidate:
    node: IdeaNode
    evaluation: IdeaEvaluation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idea": self.node.state.to_payload(),
            "evaluation": self.evaluation.to_dict(),
            "score": self.evaluation.composite,
            "path": self.node.path_summary(),
        }


@dataclass
class SearchResult:
    best: Optional[SearchCandidate]
    pareto: Dict[str, Optional[SearchCandidate]]
    trace: List[Dict[str, Any]]
    cache_size: int
    experiences: List[Dict[str, Any]]


class MemoryGuidedMCTS:
    def __init__(
        self,
        chat_fn: Callable[..., str],
        generation_prompt: str,
        evaluation_prompt: str,
        config: Optional[MCTSConfig] = None,
        memory_accessor: Optional[LongTermMemoryAccessor] = None,
        logger: Optional[logging.Logger] = None,
        log_sink: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.chat_fn = chat_fn
        self.generation_prompt = generation_prompt
        self.evaluation_prompt = evaluation_prompt
        self.config = config or MCTSConfig()
        self.logger = logger or module_logger
        self.log_sink = log_sink
        self.memory_accessor = memory_accessor or LongTermMemoryAccessor(logger=self.logger)
        self._id_iter = itertools.count()
        self.signature_nodes: Dict[str, IdeaNode] = {}
        self.evaluation_cache: Dict[str, Dict[str, IdeaEvaluation]] = {}
        self.experience_cache: Set[str] = set()
        self.trace: List[Dict[str, Any]] = []
        self.topic: str = ""
        self.analysis_blob: str = ""
        self.paper_context: str = ""

    def _log(self, level: str, message: str, *args: Any) -> None:
        log_fn = getattr(self.logger, level, self.logger.info)
        try:
            log_fn(message, *args)
        except Exception:
            self.logger.exception("MCTS logging failure for message: %s", message)
        if self.log_sink:
            try:
                formatted = message % args if args else message
            except Exception:
                formatted = f"{message} | args={args}"
            try:
                self.log_sink(level, formatted)
            except Exception as exc:
                self.logger.debug("⚠️  MCTS log sink failed: %s", exc)

    def _reset_search_state(self) -> None:
        """
        Drop cached nodes/evaluations between searches so long-running agents
        do not accumulate an ever-growing tree across topics.
        """
        self.signature_nodes = {}
        self.evaluation_cache = {}
        self.experience_cache.clear()
        self.trace = []
        self._id_iter = itertools.count()

    def search(self, topic: str, context: Dict[str, Any]) -> SearchResult:
        self._reset_search_state()
        self.topic = topic
        self.analysis_blob = format_analysis_blob(context.get("analysis", []))
        self.paper_context = context.get("paper_context") or "No curated papers available yet."
        root_state = self._build_root_state(topic, context)
        root = self._new_node(root_state, depth=0, parent=None)
        experiences = []

        for iteration in tqdm(range(self.config.max_iterations)):
            leaf, path = self._select(root)
            current_depth = len(path) - 1
            if current_depth >= self.config.max_depth:
                target = leaf
                rollout_path = path
            else:
                target, rollout_path = self._expand(leaf, path)
            if target is None:
                continue
            evaluation = self._simulate(target, rollout_path, experiences)
            if evaluation is None:
                continue
            self._backpropagate(rollout_path, evaluation)
            path_summary = self._path_summary(rollout_path)
            transformation = target.transformation
            defects = list(transformation.defects) if transformation and transformation.defects else []
            operator = transformation.operator if transformation else "unknown"
            action_summary = f"{operator} -> {', '.join(defects) if defects else 'unspecified_defect'}"
            self.trace.append(
                {
                    "iteration": iteration,
                    "node_id": target.node_id,
                    "depth": len(rollout_path) - 1,
                    "title": target.state.title,
                    "operator": operator,
                    "defects": defects,
                    "memory_refs": list(transformation.memory_refs) if transformation else [],
                    "rationale": transformation.rationale if transformation else "",
                    "score": evaluation.composite,
                    "visits": target.visits,
                    "path": path_summary,
                    "action_summary": action_summary,
                    "evaluation": {**evaluation.to_dict(), "composite": evaluation.composite},
                    "signature": target.state.signature,
                }
            )

        best = self._best_candidate(root)
        pareto = self._pareto_candidates(root)
    
        cache_entries = sum(len(entries) for entries in self.evaluation_cache.values())
        return SearchResult(
            best=best,
            pareto=pareto,
            trace=self.trace,
            cache_size=cache_entries,
            experiences=experiences,
        )

    def _new_node(
        self,
        state: IdeaState,
        depth: int,
        parent: Optional[IdeaNode],
    ) -> IdeaNode:
        existing = self.signature_nodes.get(state.signature)
        if existing:
            return existing
        node = IdeaNode(
            node_id=next(self._id_iter),
            state=state,
            depth=depth,
            parent=parent,
            transformation=OperatorApplication(
                operator=state.operator,
                defects=state.target_defects,
                rationale=state.rationale,
                memory_refs=state.memory_refs,
            ),
        )
        self.signature_nodes[state.signature] = node
        if parent:
            parent.children.append(node)
        return node

    def _attach_child(self, parent: IdeaNode, state: IdeaState) -> Optional[IdeaNode]:
        child = self.signature_nodes.get(state.signature)
        if child is None:
            return self._new_node(state, depth=parent.depth + 1, parent=parent)
        if child is parent or self._is_ancestor(parent, child):
            self._log(
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

    def _is_ancestor(self, node: IdeaNode, candidate: IdeaNode) -> bool:
        cursor: Optional[IdeaNode] = node
        while cursor is not None:
            if cursor is candidate:
                return True
            cursor = cursor.parent
        return False

    def _path_summary(self, path: List[IdeaNode]) -> str:
        steps = []
        for hop in path:
            defects = hop.transformation.defects or ["unspecified"]
            steps.append(
                f"{hop.state.title} [{hop.transformation.operator}] -> defects {defects}"
            )
        return _clip_text(" | ".join(steps), 1024)

    def _path_cache_key(self, signature: str, path_summary: str) -> str:
        raw = f"{signature}|{path_summary}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_cached_evaluation(self, signature: str, path_key: str) -> Optional[IdeaEvaluation]:
        sig_cache = self.evaluation_cache.get(signature)
        if not sig_cache:
            return None
        return sig_cache.get(path_key)

    def _cache_evaluation(self, signature: str, path_key: str, evaluation: IdeaEvaluation) -> None:
        self.evaluation_cache.setdefault(signature, {})[path_key] = evaluation

    def _get_best_cached_evaluation(self, signature: str) -> Optional[IdeaEvaluation]:
        sig_cache = self.evaluation_cache.get(signature)
        if not sig_cache:
            return None
        return max(sig_cache.values(), key=lambda ev: ev.composite)

    def _maybe_record_experience(
        self,
        path_key: str,
        node: IdeaNode,
        evaluation: IdeaEvaluation,
        path_summary: str,
        experiences: List[Dict[str, Any]],
    ) -> None:
        if path_key in self.experience_cache:
            return
        experience = self._harvest_experience(node, evaluation, path_summary)
        if not experience:
            return
        self.memory_accessor.persist_experience(experience)
        experiences.append(experience)
        self.experience_cache.add(path_key)

    def _build_root_state(self, topic: str, context: Dict[str, Any]) -> IdeaState:
        idea_pool = context.get("idea_pool") or []
        background = context.get("background_knowledge") or []
        if idea_pool:
            latest = idea_pool[-1]
            if isinstance(latest, dict):
                title = latest.get("title", f"{topic} seed idea")
                abstract = latest.get("abstract") or json.dumps(latest, ensure_ascii=False)[:200]
                core = latest.get("core_contribute") or latest.get("core_contribution", "")
                method = latest.get("methodology") or latest.get("method", "")
                experiments = latest.get("experiment_design") or latest.get("experiments", "")
                risks = latest.get("risks", latest.get("evaluation", {}))
                tags = latest.get("tags") or ["seed"]
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

        return IdeaState(
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

    def _select(self, node: IdeaNode) -> Tuple[IdeaNode, List[IdeaNode]]:
        current = node
        path = [node]
        while current.children and current.expanded:
            parent = current
            current = max(
                parent.children,
                key=lambda child: child.uct_value(
                    parent_visits=parent.visits or 1,
                    exploration_constant=self.config.exploration_constant,
                ),
            )
            path.append(current)
        return current, path

    def _expand(self, node: IdeaNode, path: List[IdeaNode]) -> Tuple[Optional[IdeaNode], List[IdeaNode]]:
        bundle = self.memory_accessor.retrieve_bundle(
            query=f"{self.topic}\n{node.state.title}\n{node.state.core_contribution}"
        )
        prompt = self.generation_prompt.format(
            topic=self.topic,
            current_summary=node.state.describe(),
            paper_context=self.paper_context,
            memory_bundle=bundle.to_prompt_block(),
            edit_operators=format_edit_operators(EDIT_OPERATORS),
            max_children=self.config.branching_factor,
            constraints="\n".join(ANTI_PATTERN_CONSTRAINTS),
        )
        prompt += "\n Directly output JSON. DO NOT include any commentary outside the JSON."
        children_payload: List[Dict[str, Any]]
        try:
            response = self.chat_fn(
                prompt,
                model=self.config.generation_model,
                temperature=self.config.generation_temperature,
                max_tokens=8192,
            )
            self._log("debug", "[MCTS] Generation response: %s", response)
            if not response or not response.strip():
                raise ValueError("Empty response from generation model")
            payload = parse_json_response(response)
            if isinstance(payload, list):
                payload = payload[0] 
            children_payload = payload.get("children", [])[: self.config.branching_factor]
        except Exception as exc:
            self._log(
                "warning",
                "⚠️  Expansion failed: %s. Falling back to heuristic children.",
                exc,
            )
            children_payload = self._fallback_child_payloads(node, bundle)

        new_child: Optional[IdeaNode] = None
        for child_data in children_payload:
            state = self._parse_child_state(child_data)
            child_node = self._attach_child(node, state)
            if child_node is None:
                continue
            cached_eval = self._get_best_cached_evaluation(state.signature)
            if cached_eval:
                child_node.evaluation = cached_eval
            if new_child is None and child_node.visits == 0:
                new_child = child_node
        node.expanded = True
        if new_child is None and node.children:
            # All children seen before; pick the least explored one.
            new_child = min(node.children, key=lambda c: c.visits)
        if new_child:
            return new_child, path + [new_child]
        return node, path

    def _parse_child_state(self, data: Dict[str, Any]) -> IdeaState:
        def _list(key: str) -> List[str]:
            raw = data.get(key, [])
            if isinstance(raw, list):
                return [str(x) for x in raw]
            if isinstance(raw, str):
                return [raw]
            return []

        return IdeaState(
            title=str(data.get("title", "Untitled idea")).strip(),
            abstract=str(data.get("abstract", "")).strip(),
            core_contribution=str(data.get("core_contribution", data.get("core_contribute", ""))).strip(),
            method=str(data.get("method", "")).strip(),
            experiments=str(data.get("experiments", data.get("experiment_design", ""))).strip(),
            risks=str(data.get("risks", "")).strip(),
            tags=_list("tags") or ["mcts-child"],
            operator=str(data.get("operator", "unknown")).strip(),
            target_defects=_list("target_defects"),
            rationale=str(data.get("rationale", "")).strip(),
            memory_refs=_list("memory_refs"),
        )

    def _fallback_child_payloads(self, node: IdeaNode, bundle: MemoryBundle) -> List[Dict[str, Any]]:
        """
        Deterministically craft child payloads when the language model fails to expand a node.
        Ensures the tree keeps growing instead of aborting the search loop.
        """
        payloads: List[Dict[str, Any]] = []
        referenced_ids = bundle.referenced_ids()
        parent_title = node.state.title
        parent_gap = node.state.core_contribution or node.state.abstract
        base_tags = node.state.tags or []
        for idx, op in enumerate(EDIT_OPERATORS[: self.config.branching_factor]):
            payloads.append(
                {
                    "title": f"{parent_title} | {op.name.replace('-', ' ').title()} #{idx+1}",
                    "abstract": f"Apply {op.description} to stress {parent_title} against {', '.join(op.defects)}.",
                    "core_contribution": f"Operationalize {op.name} to fix {', '.join(op.defects)} highlighted in '{parent_title}'.",
                    "method": f"Modify the parent method focusing on {parent_gap} via {op.description}.",
                    "experiments": f"Design ablations that validate the {op.name} intervention on the parent idea.",
                    "risks": "Heuristic fallback idea; validate with full generation later.",
                    "tags": list({*base_tags, op.name}),
                    "operator": op.name,
                    "target_defects": op.defects,
                    "memory_refs": referenced_ids,
                }
            )
        return payloads

    def _simulate(self, node: IdeaNode, path: List[IdeaNode], experiences: List[Dict[str, Any]]) -> Optional[IdeaEvaluation]:
        path_summary_text = self._path_summary(path)
        path_key = self._path_cache_key(node.state.signature, path_summary_text)
        cached_evaluation = self._get_cached_evaluation(node.state.signature, path_key)
        if cached_evaluation:
            node.evaluation = cached_evaluation
            node.latest_path_summary = path_summary_text
            self._maybe_record_experience(path_key, node, cached_evaluation, path_summary_text, experiences)
            return cached_evaluation

        prompt = self.evaluation_prompt.format(
            topic=self.topic,
            analysis=self.analysis_blob,
            paper_context=self.paper_context,
            idea=json.dumps(node.state.to_payload(), ensure_ascii=False, indent=2),
            path_summary=path_summary_text,
        )
        try:
            response = self.chat_fn(
                prompt,
                model=self.config.evaluation_model,
                temperature=self.config.evaluation_temperature,
                max_tokens=8192,
            )
            payload = parse_json_response(response)
            if isinstance(payload, list):
                payload = payload[0] # Sometimes returns a list of evaluations, we take the first one.
            evaluation = IdeaEvaluation.from_payload(payload)
        except Exception as exc:
            self._log("warning", "⚠️  Simulation failed: %s", exc)
            return None

        self._cache_evaluation(node.state.signature, path_key, evaluation)
        node.evaluation = evaluation
        node.latest_path_summary = path_summary_text

        self._maybe_record_experience(path_key, node, evaluation, path_summary_text, experiences)

        return evaluation

    def _backpropagate(self, path: List[IdeaNode], evaluation: IdeaEvaluation) -> None:
        score = evaluation.composite
        for hop in reversed(path):
            hop.visits += 1
            hop.value_sum += score

    def _best_candidate(self, root: IdeaNode) -> Optional[SearchCandidate]:
        candidates: List[SearchCandidate] = []
        stack = [root]
        visited: Set[int] = set()
        while stack:
            node = stack.pop()
            if node.node_id in visited:
                continue
            visited.add(node.node_id)
            if node.evaluation:
                candidates.append(SearchCandidate(node=node, evaluation=node.evaluation))
            stack.extend(node.children)
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.evaluation.composite)

    def _pareto_candidates(self, root: IdeaNode) -> Dict[str, Optional[SearchCandidate]]:
        by_metric = {
            "novel": lambda ev: ev.novelty,
            "feasible": lambda ev: ev.feasibility,
            "concise": lambda ev: ev.conciseness,
        }
        pareto: Dict[str, Optional[SearchCandidate]] = {k: None for k in by_metric}
        stack = [root]
        visited_ids: Set[int] = set()
        visited: List[SearchCandidate] = []
        while stack:
            node = stack.pop()
            if node.node_id in visited_ids:
                continue
            visited_ids.add(node.node_id)
            if node.evaluation:
                visited.append(SearchCandidate(node=node, evaluation=node.evaluation))
            stack.extend(node.children)

        for label, scorer in by_metric.items():
            if visited:
                pareto[label] = max(visited, key=lambda c, s=scorer: s(c.evaluation))
        return pareto

    def _harvest_experience(self, node: IdeaNode, evaluation: IdeaEvaluation, path_summary: str) -> Optional[Dict[str, Any]]:
        if evaluation.confidence > self.config.min_confidence_for_memory:
            experience = {
                    "defect": ", ".join(node.state.target_defects) or evaluation.defect_fix_summary,
                    "action": node.state.operator,
                    "lift": round(evaluation.lift_estimate, 2),
                    "idea": node.state.title,
                    "context": path_summary,
                    "feedback": evaluation.feedback,
                    "tags": node.state.tags + ["defect_fix"],
                }

            return experience
        else:
            return None
