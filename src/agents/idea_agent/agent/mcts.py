from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional runtime fallback
    OmegaConf = None
from tqdm import tqdm

from memory.api.faiss_memory_system_api import FAISSMemorySystem
from memory.api.slot_process_api import SlotProcess
from memory.api.symbolic_memory_system_api import SymbolicMemorySystem
from memory.api.component_taxonomy import (
    ContextSignature,
)
from memory.memory_system.models import EpisodicRecord, ProceduralRecord, SemanticRecord
from memory.memory_system.utils import _multi_thread_run, _safe_dump_str
from agent import get_logger
from src.agents.idea_agent.utils.component_novelty import ComponentNoveltyScorer
from src.agents.idea_agent.utils.mcts_helpers import clip_text, format_analysis_blob
from src.agents.idea_agent.agent.prompts.skill_instantiation import SKILL_INSTANTIATION_PROMPT
from src.agents.idea_agent.agent.prompts.component_extraction import COMPONENT_EXTRACTION_PROMPT
from src.agents.idea_agent.utils.mcts_runtime import (
    EditPlan,
    MemoryBundle,
    MemorySnippet,
    SkillCatalog,
    SkillUsagePrior,
    apply_budget_delta_to_parent,
    best_candidate,
    component_inventory_payload,
    build_symbolic_eval_hints,
    build_root_state,
    compute_protocol_score_from_plan,
    expand_symbolic_memory_log_payload,
    extract_context_sig_from_node,
    extract_mature_idea_components_via_llm,
    instantiate_skill_plan_for_node,
    log_message,
    materialize_child_state,
    memory_bundle_log_payload,
    new_node,
    normalize_component_explanations,
    normalize_budget_dict,
    pareto_candidates,
    path_summary,
    plan_to_experiment_text,
    plan_to_method_text,
    backpropagate_rollout,
    expand_node_with_skills,
    reset_search_state,
    select_leaf_for_rollout,
    simulate_log_payload,
    simulate_node_value,
    update_skill_prior_from_evaluation,
)


MAX_IDEA_TEXT = 900
MAX_RATIONALE_TEXT = 700
MAX_TITLE_TEXT = 256
MAX_LIST_ENTRIES = 16
MAX_REF_TEXT = 128
MIN_COMPONENTS = 1
MAX_COMPONENTS = 5

module_logger = get_logger()


def _load_mcts_defaults() -> Dict[str, Any]:
    if OmegaConf is None:
        return {}
    config_path = Path(__file__).resolve().parents[1] / "config" / "mcts" / "default.yaml"
    config = OmegaConf.load(config_path)
    mcts_config = config.get("mcts") if hasattr(config, "get") else None
    if mcts_config is None:
        return {}
    try:
        return OmegaConf.to_container(mcts_config, resolve=True) or {}
    except Exception:
        return dict(mcts_config) if isinstance(mcts_config, dict) else {}


_MCTS_DEFAULTS = _load_mcts_defaults()


def _mcts_default(key: str, fallback: Any) -> Any:
    value = _MCTS_DEFAULTS.get(key, fallback)
    return fallback if value is None else value


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return fallback


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered


def _pretty_json(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


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
    budget: Dict[str, Any] = field(default_factory=dict)
    components: List[str] = field(default_factory=list)
    component_explanations: Dict[str, str] = field(default_factory=dict)
    paper_graph_context: str = ""
    edit_plan: Optional[Dict[str, Any]] = None
    skill_metrics: Dict[str, Any] = field(default_factory=dict)
    signature: str = field(init=False)

    def __post_init__(self) -> None:
        self.title = clip_text(self.title, MAX_TITLE_TEXT)
        self.abstract = clip_text(self.abstract, MAX_IDEA_TEXT)
        self.core_contribution = clip_text(self.core_contribution, MAX_IDEA_TEXT)
        self.method = clip_text(self.method, MAX_IDEA_TEXT)
        self.experiments = clip_text(self.experiments, MAX_IDEA_TEXT)
        self.risks = clip_text(self.risks, MAX_IDEA_TEXT)
        self.rationale = clip_text(self.rationale, MAX_RATIONALE_TEXT)
        self.paper_graph_context = clip_text(self.paper_graph_context, MAX_IDEA_TEXT)

        self.tags = _dedupe_keep_order(
            [clip_text(tag, MAX_IDEA_TEXT) for tag in self.tags[:MAX_LIST_ENTRIES]]
        )
        self.target_defects = _dedupe_keep_order(
            [clip_text(tag, MAX_IDEA_TEXT) for tag in self.target_defects[:MAX_LIST_ENTRIES]]
        )
        self.memory_refs = _dedupe_keep_order(
            [clip_text(ref, MAX_REF_TEXT) for ref in self.memory_refs[:MAX_LIST_ENTRIES]]
        )
        self.components = _dedupe_keep_order(
            [clip_text(comp, MAX_REF_TEXT) for comp in self.components[: 2 * MAX_LIST_ENTRIES]]
        )
        self.component_explanations = normalize_component_explanations(
            self.components,
            self.component_explanations,
        )

        if not isinstance(self.budget, dict):
            self.budget = {}

        canonical = "|".join(
            [
                self.title.lower(),
                self.core_contribution.lower(),
                self.method.lower(),
                ",".join(sorted(self.tags)),
                ",".join(sorted(self.components)),
                "|".join(
                    f"{component}:{self.component_explanations.get(component, '').lower()}"
                    for component in self.components
                ),
            ]
        )
        self.signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def describe(self) -> str:
        base = (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Core Contribution: {self.core_contribution}\n"
            f"Method: {self.method}\n"
            f"Experiments: {self.experiments}\n"
            f"Risks: {self.risks}\n"
            f"Components: {', '.join(self.components)}\n"
            f"Component Roles: {self.component_explanations}\n"
            f"Defects: {', '.join(self.target_defects)}\n"
            f"Budget: {self.budget}\n"
            f"Action Skill: {self.operator}"
        )
        if self.edit_plan:
            objective = self.edit_plan.get("objective", "")
            edits = self.edit_plan.get("component_edits", [])
            return (
                f"{base}\n"
                f"Plan Objective: {objective}\n"
                f"Atomic Edits: {len(edits)}"
            )
        return base

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "title": self.title,
            "abstract": self.abstract,
            "core_contribution": self.core_contribution,
            "method": self.method,
            "experiments": self.experiments,
            "risks": self.risks,
            "tags": self.tags,
            "operator": self.operator,
            "target_defects": self.target_defects,
            "memory_refs": self.memory_refs,
            "rationale": self.rationale,
            "budget": self.budget,
            "components": self.components,
            "component_explanations": self.component_explanations,
            "components_with_explanations": self.component_inventory(),
            "paper_graph_context": self.paper_graph_context,
        }
        if self.edit_plan:
            payload["edit_plan"] = self.edit_plan
        if self.skill_metrics:
            payload["skill_metrics"] = self.skill_metrics
        return payload

    def component_inventory(self) -> List[Dict[str, str]]:
        return component_inventory_payload(self.components, self.component_explanations)


@dataclass
class IdeaEvaluation:
    novelty: float
    feasibility: float
    clarity: float
    impact: float
    risk: float
    conciseness: float
    alignment_score: float
    complexity_penalty: float
    protocol_score: float
    confidence: float
    failure_modes: List[str]
    fairness_protocol: str
    feedback: str
    defect_fix_summary: str
    detected_defects: List[str]
    lift_estimate: float
    novelty_weight: float = 0.30
    impact_weight: float = 0.25
    feasibility_weight: float = 0.20
    clarity_weight: float = 0.15
    conciseness_weight: float = 0.10
    risk_weight: float = 0.20
    alignment_weight: float = 0.25
    complexity_weight: float = 0.20
    protocol_weight: float = 0.15

    def __post_init__(self) -> None:
        self.failure_modes = [
            clip_text(mode, MAX_IDEA_TEXT)
            for mode in (self.failure_modes or [])[:MAX_LIST_ENTRIES]
        ]
        self.fairness_protocol = clip_text(self.fairness_protocol, MAX_IDEA_TEXT)
        self.feedback = clip_text(self.feedback, MAX_IDEA_TEXT)
        self.defect_fix_summary = clip_text(self.defect_fix_summary, MAX_IDEA_TEXT)
        self.detected_defects = [
            clip_text(tag, MAX_IDEA_TEXT)
            for tag in (self.detected_defects or [])[:MAX_LIST_ENTRIES]
        ]
        self.confidence = max(0.0, min(1.0, _safe_float(self.confidence, 0.0)))

    @classmethod
    def from_payload(
        cls,
        payload: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
    ) -> "IdeaEvaluation":
        def _num(key: str, default: float = 0.0) -> float:
            return _safe_float(payload.get(key, default), default)

        def _list(key: str) -> List[str]:
            raw = payload.get(key, [])
            if isinstance(raw, list):
                return [str(x) for x in raw]
            if isinstance(raw, str):
                return [raw]
            return []

        w = weights or {}
        return cls(
            novelty=_num("novelty"),
            feasibility=_num("feasibility"),
            clarity=_num("clarity"),
            impact=_num("impact"),
            risk=_num("risk"),
            conciseness=_num("conciseness"),
            alignment_score=_num("alignment_score"),
            complexity_penalty=_num("complexity_penalty"),
            protocol_score=_num("protocol_score"),
            confidence=_num("confidence", 0.0),
            failure_modes=_list("failure_modes"),
            fairness_protocol=str(payload.get("fairness_protocol", "")),
            feedback=str(payload.get("feedback", "")),
            defect_fix_summary=str(payload.get("defect_fix_summary", "")),
            detected_defects=_list("detected_defects"),
            lift_estimate=max(0.0, _num("lift_estimate", 0.0)),
            novelty_weight=w.get("novelty_weight", 0.30),
            impact_weight=w.get("impact_weight", 0.25),
            feasibility_weight=w.get("feasibility_weight", 0.20),
            clarity_weight=w.get("clarity_weight", 0.15),
            conciseness_weight=w.get("conciseness_weight", 0.10),
            risk_weight=w.get("risk_weight", 0.20),
            alignment_weight=w.get("alignment_weight", 0.25),
            complexity_weight=w.get("complexity_weight", 0.20),
            protocol_weight=w.get("protocol_weight", 0.15),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "novelty": self.novelty,
            "feasibility": self.feasibility,
            "clarity": self.clarity,
            "impact": self.impact,
            "risk": self.risk,
            "conciseness": self.conciseness,
            "alignment_score": self.alignment_score,
            "complexity_penalty": self.complexity_penalty,
            "protocol_score": self.protocol_score,
            "confidence": self.confidence,
            "failure_modes": self.failure_modes,
            "fairness_protocol": self.fairness_protocol,
            "feedback": self.feedback,
            "defect_fix_summary": self.defect_fix_summary,
            "detected_defects": self.detected_defects,
            "lift_estimate": self.lift_estimate,
        }

    @property
    def composite(self) -> float:
        pos_total = max(
            self.novelty_weight + self.impact_weight + self.feasibility_weight
            + self.clarity_weight + self.conciseness_weight + self.protocol_weight,
            1e-9,
        )
        positive = (
            self.novelty_weight * self.novelty
            + self.impact_weight * self.impact
            + self.feasibility_weight * self.feasibility
            + self.clarity_weight * self.clarity
            + self.conciseness_weight * self.conciseness
            + self.protocol_weight * self.protocol_score
        ) / pos_total
        adj_total = max(
            self.risk_weight + self.complexity_weight + self.alignment_weight, 1e-9
        )
        adjustment = (
            self.alignment_weight * self.alignment_score
            - self.risk_weight * self.risk
            - self.complexity_weight * self.complexity_penalty
        ) / adj_total
        return positive + adjustment


@dataclass
class OperatorApplication:
    operator: str
    defects: List[str]
    rationale: str
    memory_refs: List[str]

    def __post_init__(self) -> None:
        self.operator = clip_text(self.operator, MAX_IDEA_TEXT)
        self.defects = [clip_text(defect, MAX_IDEA_TEXT) for defect in self.defects[:MAX_LIST_ENTRIES]]
        self.rationale = clip_text(self.rationale, MAX_RATIONALE_TEXT)
        self.memory_refs = [clip_text(ref, MAX_REF_TEXT) for ref in self.memory_refs[:MAX_LIST_ENTRIES]]


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
        explore = exploration_constant * math.sqrt(math.log(max(1, parent_visits)) / self.visits)
        return exploit + explore

    def path(self) -> List["IdeaNode"]:
        chain: List[IdeaNode] = []
        node: Optional[IdeaNode] = self
        while node:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    def path_summary(self) -> str:
        if self.latest_path_summary:
            return self.latest_path_summary
        steps: List[str] = []
        for hop in self.path():
            steps.append(
                f"{hop.state.title} [{hop.transformation.operator}] -> defects {hop.transformation.defects}"
            )
        return " | ".join(steps)


@dataclass
class MCTSConfig:
    max_iterations: int = _mcts_default("max_iterations", 128)
    max_depth: int = _mcts_default("max_depth", 4)
    branching_factor: int = _mcts_default("branching_factor", 3)
    exploration_constant: float = _mcts_default("exploration_constant", 1.15)

    generation_model: str = _mcts_default("generation_model", "gpt-5-mini")
    evaluation_model: str = _mcts_default("evaluation_model", "gpt-5.2")
    generation_temperature: float = _mcts_default("generation_temperature", 0.2)
    evaluation_temperature: float = _mcts_default("evaluation_temperature", 0.01)
    generation_max_tokens: int = _mcts_default("generation_max_tokens", 8192)
    evaluation_max_tokens: int = _mcts_default("evaluation_max_tokens", 8192)
    component_novelty_model: str = _mcts_default(
        "component_novelty_model", "all-MiniLM-L6-v2"
    )
    component_novelty_index_dir: Optional[str] = _mcts_default(
        "component_novelty_index_dir", None
    )
    component_novelty_retrieval_top_k: int = _mcts_default(
        "component_novelty_retrieval_top_k", 50
    )
    component_novelty_evidence_top_k: int = _mcts_default(
        "component_novelty_evidence_top_k", 5
    )
    component_novelty_eval_model: Optional[str] = _mcts_default(
        "component_novelty_eval_model", None
    )
    component_novelty_eval_temperature: float = _mcts_default(
        "component_novelty_eval_temperature",
        _mcts_default("evaluation_temperature", 0.01),
    )
    component_novelty_eval_max_tokens: int = _mcts_default(
        "component_novelty_eval_max_tokens", 4096
    )

    min_confidence_for_memory: float = _mcts_default("min_confidence_for_memory", 0.6)
    pareto_top_k: int = _mcts_default("pareto_top_k", 5)

    alignment_weight: float = _mcts_default("alignment_weight", 0.25)
    complexity_weight: float = _mcts_default("complexity_weight", 0.2)
    novelty_weight: float = _mcts_default("novelty_weight", 0.30)
    impact_weight: float = _mcts_default("impact_weight", 0.25)
    feasibility_weight: float = _mcts_default("feasibility_weight", 0.20)
    clarity_weight: float = _mcts_default("clarity_weight", 0.15)
    conciseness_weight: float = _mcts_default("conciseness_weight", 0.10)
    risk_weight: float = _mcts_default("risk_weight", 0.20)
    protocol_weight: float = _mcts_default("protocol_weight", 0.15)

    skill_prior_memory_path: str = _mcts_default(
        "skill_prior_memory_path", "output/idea_skill_priors"
    )
    skill_prior_success_threshold: float = _mcts_default(
        "skill_prior_success_threshold", 0.6
    )


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


class VectorMemoryAccessor:
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
                self._stores[memory_type] = FAISSMemorySystem(
                    memory_type=memory_type,
                    llm_name="gpt-4.1-mini",
                    backend="openai",
                    **cfg,
                )
            except Exception as exc:
                log_message(
                    self.logger,
                    None,
                    "warning",
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
            log_message(
                self.logger,
                None,
                "warning",
                "⚠️  Memory query failed (%s): %s",
                prefix,
                exc,
            )
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
                    title=(title or "")[:80],
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
            bundle.field_knowledge = self._query_store(semantic, query, limit, prefix="Field")
        if episodic:
            bundle.anti_patterns = self._query_store(episodic, query, limit, prefix="Pattern")
        if procedural:
            bundle.fix_recipes = self._query_store(procedural, query, limit, prefix="Recipe")

        return bundle

    def persist_experience(self, experience: Dict[str, Any], max_workers: int = 20) -> None:
        if not experience:
            return

        semantic = self._get_store("semantic")
        episodic = self._get_store("episodic")
        procedural = self._get_store("procedural")
        if not semantic and not episodic and not procedural:
            log_message(
                self.logger,
                None,
                "info",
                "ℹ️ Skipping persistence because semantic store is unavailable.",
            )
            return

        slot_process = SlotProcess(llm_name="gpt-4.1", llm_backend="openai")
        try:
            working_slots = slot_process.transfer_idea_agent_context_to_working_slots(experience)
            log_message(
                self.logger,
                None,
                "debug",
                "[MCTS] Transferred experience to working slots (count=%d)",
                len(working_slots),
            )
            _multi_thread_run(slot_process._multi_thread_filter_and_route_slot, working_slots, max_workers)
            _multi_thread_run(
                slot_process._multi_thread_transfer_slot_to_memory,
                slot_process.routed_slot_container,
                max_workers,
            )
        except Exception as exc:
            log_message(
                self.logger,
                None,
                "warning",
                "⚠️  Failed to persist experience: %s",
                exc,
            )
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

        if semantic and semantic_records:
            try:
                semantic.add(semantic_records, agent_id="idea_agent")
            except Exception as exc:
                log_message(self.logger, None, "warning", "⚠️  Failed to persist semantic records: %s", exc)
        if episodic and episodic_records:
            try:
                episodic.add(episodic_records, agent_id="idea_agent")
            except Exception as exc:
                log_message(self.logger, None, "warning", "⚠️  Failed to persist episodic records: %s", exc)
        if procedural and procedural_records:
            try:
                procedural.add(procedural_records, agent_id="idea_agent")
            except Exception as exc:
                log_message(self.logger, None, "warning", "⚠️  Failed to persist procedural records: %s", exc)

        log_message(
            self.logger,
            None,
            "debug",
            "[MCTS] Persisted records -> semantic=%d | episodic=%d | procedural=%d",
            len(semantic_records),
            len(episodic_records),
            len(procedural_records),
        )


class MemoryGuidedMCTS:
    def __init__(
        self,
        chat_fn: Callable[..., str],
        evaluation_prompt: str,
        config: Optional[MCTSConfig] = None,
        memory_accessor: Optional[VectorMemoryAccessor] = None,
        logger: Optional[logging.Logger] = None,
        log_sink: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.chat_fn = chat_fn
        self.evaluation_prompt = evaluation_prompt

        self.config = config or MCTSConfig()
        self.logger = logger or module_logger
        self.log_sink = log_sink
        self.memory_accessor = memory_accessor or VectorMemoryAccessor(logger=self.logger)
        self.component_novelty_scorer = ComponentNoveltyScorer(
            model_name_or_path=self.config.component_novelty_model,
            index_dir=self.config.component_novelty_index_dir or None,
            retrieval_top_k=self.config.component_novelty_retrieval_top_k,
            evidence_node_top_k=self.config.component_novelty_evidence_top_k,
            evaluation_model=(
                self.config.component_novelty_eval_model or self.config.evaluation_model
            ),
            evaluation_temperature=self.config.component_novelty_eval_temperature,
            evaluation_max_tokens=self.config.component_novelty_eval_max_tokens,
            chat_fn=self.chat_fn,
            logger=self.logger,
            log_sink=self.log_sink,
        )

        self.skill_catalog = SkillCatalog()
        self.symbolic_memory = SymbolicMemorySystem()
        self.symbolic_memory_path = Path(self.config.skill_prior_memory_path)

        self._id_iter = itertools.count()
        self.signature_nodes: Dict[str, IdeaNode] = {}
        self.evaluation_cache: Dict[str, Dict[str, IdeaEvaluation]] = {}
        self.experience_cache: Set[str] = set()
        self.trace: List[Dict[str, Any]] = []

        self.topic: str = ""
        self.analysis_blob: str = ""
        self.paper_context: str = ""
        self.mature_idea: str = ""
        self._mature_idea_components: List[str] = []
        self._mature_idea_component_explanations: Dict[str, str] = {}
        self._component_novelty_fallback_logged = False

        self._load_skill_prior_memory()

    def _load_skill_prior_memory(self) -> None:
        """Load the symbolic memory store so that compute_action_priors is available
        during expand.  The store is populated externally (e.g. from experiment
        ablation results or paper-graph conclusions) — not from within MCTS.
        """
        try:
            if not self.symbolic_memory_path.exists():
                return
            self.symbolic_memory.load(str(self.symbolic_memory_path))
        except Exception as exc:
            log_message(
                self.logger,
                self.log_sink,
                "warning",
                "⚠️  Unable to load symbolic memory store: %s",
                exc,
            )

    def _skill_prior_for_prompt(self, skill_name: str) -> Dict[str, Any]:
        prior = self.skill_catalog.priors.get(skill_name, SkillUsagePrior())
        return prior.to_dict()

    def _normalize_budget(self, budget: Dict[str, Any]) -> Dict[str, Any]:
        return normalize_budget_dict(budget)

    def _apply_budget_delta(
        self,
        parent_budget: Dict[str, Any],
        delta: Dict[str, Any],
    ) -> Dict[str, Any]:
        return apply_budget_delta_to_parent(parent_budget, delta)

    def _plan_to_method_text(self, plan: EditPlan) -> str:
        return plan_to_method_text(plan)

    def _plan_to_experiment_text(self, plan: EditPlan) -> str:
        return plan_to_experiment_text(plan)

    def _compute_protocol_score(self, plan: Optional[Dict[str, Any]]) -> float:
        return compute_protocol_score_from_plan(plan)

    def _memory_bundle_log_payload(self, bundle: MemoryBundle) -> Dict[str, Any]:
        return memory_bundle_log_payload(bundle)

    def _expand_symbolic_memory_log_payload(
        self,
        parent_ctx_sig: ContextSignature,
        component_families: List[Dict[str, Any]],
        action_priors: Dict[str, float],
    ) -> Dict[str, Any]:
        return expand_symbolic_memory_log_payload(
            self,
            parent_ctx_sig,
            component_families,
            action_priors,
        )

    def _simulate_log_payload(self, evaluation: IdeaEvaluation) -> Dict[str, Any]:
        return simulate_log_payload(evaluation)

    def _score_component_novelty(self, state: IdeaState) -> Optional[float]:
        try:
            return self.component_novelty_scorer.score(state=state, topic=self.topic)
        except Exception as exc:
            if not self._component_novelty_fallback_logged:
                log_message(
                    self.logger,
                    self.log_sink,
                    "warning",
                    "⚠️  Component novelty scorer unavailable (%s); falling back to LLM novelty.",
                    exc,
                )
                self._component_novelty_fallback_logged = True
            return None

    def _materialize_child_state(
        self,
        parent_state: IdeaState,
        plan: EditPlan,
        instantiated: Optional[Dict[str, Any]] = None,
    ) -> IdeaState:
        return materialize_child_state(
            self,
            parent_state,
            plan,
            instantiated,
            idea_state_cls=IdeaState,
        )

    def _extract_context_sig(self, node: IdeaNode) -> ContextSignature:
        return extract_context_sig_from_node(node)

    def _select(self, node: IdeaNode) -> Tuple[IdeaNode, List[IdeaNode]]:
        return select_leaf_for_rollout(self, node)

    def _instantiate_skill_plan(
        self,
        plan: EditPlan,
        parent_state: IdeaState,
        bundle: MemoryBundle,
    ) -> Optional[Dict[str, Any]]:
        return instantiate_skill_plan_for_node(
            self,
            plan,
            parent_state,
            bundle,
            prompt_template=SKILL_INSTANTIATION_PROMPT,
        )

    def _expand(self, node: IdeaNode, path: List[IdeaNode]) -> Tuple[Optional[IdeaNode], List[IdeaNode]]:
        return expand_node_with_skills(
            self,
            node,
            path,
            min_components=MIN_COMPONENTS,
            max_components=MAX_COMPONENTS,
            pretty_json=_pretty_json,
        )


    def _build_symbolic_eval_hints(self, node: IdeaNode) -> str:
        return build_symbolic_eval_hints(self, node)

    def _simulate(
        self,
        node: IdeaNode,
        path: List[IdeaNode],
        experiences: List[Dict[str, Any]],
    ) -> Optional[IdeaEvaluation]:
        return simulate_node_value(
            self,
            node,
            path,
            experiences,
            idea_evaluation_cls=IdeaEvaluation,
            pretty_json=_pretty_json,
        )

    def _update_skill_prior(self, node: IdeaNode, evaluation: IdeaEvaluation) -> None:
        update_skill_prior_from_evaluation(self, node, evaluation)

    def _backpropagate(self, path: List[IdeaNode], evaluation: IdeaEvaluation) -> None:
        backpropagate_rollout(path, evaluation)

    def _extract_mature_idea_components(
        self,
        mature_idea: str,
        topic: str,
    ) -> Tuple[List[str], Dict[str, str]]:
        return extract_mature_idea_components_via_llm(
            self,
            mature_idea,
            topic,
            prompt_template=COMPONENT_EXTRACTION_PROMPT,
            max_components=MAX_COMPONENTS,
        )

    def search(self, topic: str, context: Dict[str, Any]) -> SearchResult:
        reset_search_state(self)
        self.topic = topic
        self.analysis_blob = format_analysis_blob(context.get("analysis", []))
        self.paper_context = context.get("paper_context") or "No curated papers available yet."
        self.mature_idea = (context.get("mature_idea") or "").strip()

        # Extract components from mature idea if provided
        self._mature_idea_components = []
        self._mature_idea_component_explanations = {}
        if self.mature_idea:
            (
                self._mature_idea_components,
                self._mature_idea_component_explanations,
            ) = self._extract_mature_idea_components(
                self.mature_idea, topic
            )
            if self._mature_idea_components:
                context["components"] = self._mature_idea_components
                context["component_explanations"] = self._mature_idea_component_explanations
                log_message(
                    self.logger,
                    self.log_sink,
                    "debug",
                    "[MCTS] Extracted %d component(s) from mature idea: %s",
                    len(self._mature_idea_components),
                    self._mature_idea_components,
                )

        root_state = build_root_state(topic, context, IdeaState)
        log_message(
            self.logger,
            self.log_sink,
            "debug",
            "[MCTS] Root state component_count=%d components=%s",
            len(root_state.components),
            root_state.components,
        )
        root = new_node(
            root_state,
            depth=0,
            parent=None,
            signature_nodes=self.signature_nodes,
            id_iter=self._id_iter,
            idea_node_cls=IdeaNode,
            operator_application_cls=OperatorApplication,
        )
        experiences: List[Dict[str, Any]] = []

        # Prime root defects via one evaluator pass before the first expand so
        # initial skill selection is not forced to rely on the placeholder
        # "unexplored_gap" defect tag.
        root_eval = self._simulate(root, [root], experiences)
        if root_eval and root_eval.detected_defects:
            inferred_defects = _dedupe_keep_order(
                [str(tag) for tag in root_eval.detected_defects if str(tag).strip()]
            )
            if inferred_defects:
                previous_defects = list(root.state.target_defects)
                root.state.target_defects = inferred_defects
                root.transformation.defects = inferred_defects
                root.latest_path_summary = ""
                log_message(
                    self.logger,
                    self.log_sink,
                    "debug",
                    "[MCTS] Root priming inferred defects=%s (previous=%s)",
                    inferred_defects,
                    previous_defects,
                )

        for iteration in tqdm(range(self.config.max_iterations)):
            leaf, path = self._select(root)
            depth = len(path) - 1

            if depth >= self.config.max_depth:
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
            self._update_skill_prior(target, evaluation)

            rollout_summary = path_summary(rollout_path)
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
                    "path": rollout_summary,
                    "action_summary": action_summary,
                    "evaluation": {**evaluation.to_dict(), "composite": evaluation.composite},
                    "signature": target.state.signature,
                    "edit_plan": target.state.edit_plan,
                    "skill_metrics": target.state.skill_metrics,
                }
            )

        best = best_candidate(root, SearchCandidate)
        pareto = pareto_candidates(root, SearchCandidate)
        cache_entries = sum(len(entries) for entries in self.evaluation_cache.values())

        # Persist injected symbolic priors to disk so they survive across runs
        try:
            self.symbolic_memory_path.parent.mkdir(parents=True, exist_ok=True)
            self.symbolic_memory.save(str(self.symbolic_memory_path))
            log_message(
                self.logger,
                self.log_sink,
                "debug",
                "[MCTS] Symbolic memory saved to %s",
                self.symbolic_memory_path,
            )
        except Exception as exc:
            log_message(
                self.logger,
                self.log_sink,
                "warning",
                "⚠️  Failed to save symbolic memory: %s",
                exc,
            )

        return SearchResult(
            best=best,
            pareto=pareto,
            trace=self.trace,
            cache_size=cache_entries,
            experiences=experiences,
        )
