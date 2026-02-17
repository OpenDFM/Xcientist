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
from memory.memory_system.models import EpisodicRecord, ProceduralRecord, SemanticRecord
from memory.memory_system.utils import _multi_thread_run, _safe_dump_str
from agent import get_logger
from src.agents.idea_agent.utils.mcts_helpers import clip_text, format_analysis_blob, parse_json_response
from src.agents.idea_agent.agent.prompts.skill_instantiation import SKILL_INSTANTIATION_PROMPT
from src.agents.idea_agent.agent.prompts.component_extraction import COMPONENT_EXTRACTION_PROMPT
from src.agents.idea_agent.utils.mcts_runtime import (
    ANTI_PATTERN_CONSTRAINTS,
    AtomicEditOp,
    ComponentEdit,
    EditPlan,
    MemoryBundle,
    MemorySnippet,
    SkillCatalog,
    SkillUsagePrior,
    apply_edit_plan_to_components,
    attach_child,
    best_candidate,
    build_root_state,
    cache_evaluation,
    format_defect_registry,
    get_best_cached_evaluation,
    get_cached_evaluation,
    log_message,
    maybe_record_experience,
    new_node,
    pareto_candidates,
    path_cache_key,
    path_summary,
    reset_search_state,
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

        if not isinstance(self.budget, dict):
            self.budget = {}

        canonical = "|".join(
            [
                self.title.lower(),
                self.core_contribution.lower(),
                self.method.lower(),
                ",".join(sorted(self.tags)),
                ",".join(sorted(self.components)),
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
            "core_contribute": self.core_contribution,
            "core_contribution": self.core_contribution,
            "methodology": self.method,
            "method": self.method,
            "experiment_design": self.experiments,
            "experiments": self.experiments,
            "risks": self.risks,
            "tags": self.tags,
            "operator": self.operator,
            "target_defects": self.target_defects,
            "memory_refs": self.memory_refs,
            "rationale": self.rationale,
            "budget": self.budget,
            "components": self.components,
            "paper_graph_context": self.paper_graph_context,
        }
        if self.edit_plan:
            payload["edit_plan"] = self.edit_plan
        if self.skill_metrics:
            payload["skill_metrics"] = self.skill_metrics
        return payload


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
    def from_payload(cls, payload: Dict[str, Any]) -> "IdeaEvaluation":
        def _num(key: str, default: float = 0.0) -> float:
            return _safe_float(payload.get(key, default), default)

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
        positive = (
            self.novelty_weight * self.novelty
            + self.impact_weight * self.impact
            + self.feasibility_weight * self.feasibility
            + self.clarity_weight * self.clarity
            + self.conciseness_weight * self.conciseness
            + self.protocol_weight * self.protocol_score
        )
        penalty = self.risk_weight * self.risk + self.complexity_weight * self.complexity_penalty
        bonus = self.alignment_weight * self.alignment_score
        return positive + bonus - penalty


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
                "info",
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
            "info",
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

        self.skill_catalog = SkillCatalog()
        self.symbolic_memory = SymbolicMemorySystem()
        self.symbolic_memory_path = Path(self.config.skill_prior_memory_path)
        self._symbolic_dirty = False

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

        self._load_skill_prior_memory()

    def _load_skill_prior_memory(self) -> None:
        try:
            if not self.symbolic_memory_path.exists():
                return
            loaded = self.symbolic_memory.load(str(self.symbolic_memory_path))
            if not loaded:
                return
            records, _ = self.symbolic_memory.get_last_k_records(300)
            for payload in records:
                if not isinstance(payload, dict):
                    continue
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                skill_name = str(metadata.get("skill_name", "")).strip()
                if not skill_name or skill_name not in self.skill_catalog.priors:
                    continue
                prior = self.skill_catalog.priors[skill_name]
                prior.attempts = max(prior.attempts, int(metadata.get("attempts", prior.attempts) or prior.attempts))
                prior.successes = max(prior.successes, int(metadata.get("successes", prior.successes) or prior.successes))
                prior.reward_ema = max(prior.reward_ema, _safe_float(metadata.get("reward_ema"), prior.reward_ema))
                prior.prior = max(prior.prior, _safe_float(metadata.get("prior"), prior.prior))
                anti_patterns = payload.get("anti_patterns") if isinstance(payload.get("anti_patterns"), list) else []
                for item in anti_patterns:
                    rule = f"Avoid failure mode: {str(item).strip()}"
                    if rule not in prior.rule_constraints:
                        prior.rule_constraints.append(rule)
                prior.rule_constraints = prior.rule_constraints[:8]
        except Exception as exc:
            log_message(
                self.logger,
                self.log_sink,
                "warning",
                "⚠️  Unable to load skill prior memory: %s",
                exc,
            )

    def _persist_skill_prior_memory(self) -> None:
        if not self._symbolic_dirty:
            return
        try:
            self.symbolic_memory_path.mkdir(parents=True, exist_ok=True)
            self.symbolic_memory.save(str(self.symbolic_memory_path))
            self._symbolic_dirty = False
        except Exception as exc:
            log_message(
                self.logger,
                self.log_sink,
                "warning",
                "⚠️  Unable to save skill prior memory: %s",
                exc,
            )

    def _skill_prior_for_prompt(self, skill_name: str) -> Dict[str, Any]:
        prior = self.skill_catalog.priors.get(skill_name, SkillUsagePrior())
        return prior.to_dict()

    def _normalize_budget(self, budget: Dict[str, Any]) -> Dict[str, Any]:
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

    def _apply_budget_delta(
        self,
        parent_budget: Dict[str, Any],
        delta: Dict[str, Any],
    ) -> Dict[str, Any]:
        next_budget = self._normalize_budget(parent_budget)
        for key, val in delta.items():
            if key not in next_budget:
                next_budget[key] = _safe_float(val, 0.0)
                continue
            base = next_budget.get(key)
            if isinstance(base, (int, float)):
                next_budget[key] = round(float(base) + _safe_float(val, 0.0), 4)
        return next_budget

    def _plan_to_method_text(self, plan: EditPlan) -> str:
        lines: List[str] = []
        for idx, edit in enumerate(plan.component_edits, start=1):
            target = f" -> {edit.target}" if edit.target else ""
            condition = f" [condition: {edit.condition}]" if edit.condition else ""
            details = f"; {edit.details}" if edit.details else ""
            lines.append(
                f"{idx}. {edit.op.value}({edit.component}{target}){condition}{details}"
            )
        return "\n".join(lines)

    def _plan_to_experiment_text(self, plan: EditPlan) -> str:
        blocks: List[str] = []
        if plan.validation.regression_tests:
            blocks.append("Regression:\n- " + "\n- ".join(plan.validation.regression_tests))
        if plan.validation.ablation_tests:
            blocks.append("Ablation:\n- " + "\n- ".join(plan.validation.ablation_tests))
        if plan.validation.stress_tests:
            blocks.append("Stress:\n- " + "\n- ".join(plan.validation.stress_tests))
        return "\n\n".join(blocks)

    def _compute_protocol_score(self, plan: Optional[Dict[str, Any]]) -> float:
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

    def _materialize_child_state(
        self,
        parent_state: IdeaState,
        plan: EditPlan,
        instantiated: Optional[Dict[str, Any]] = None,
    ) -> IdeaState:
        new_components = apply_edit_plan_to_components(parent_state.components, plan)
        next_budget = self._apply_budget_delta(parent_state.budget, plan.estimated_budget_delta)

        inst = instantiated or {}
        # Use LLM-generated content if available, otherwise fall back to plan-template text
        title = inst.get("title") or f"{parent_state.title} | {plan.skill_name.replace('-', ' ').title()}"
        abstract = inst.get("abstract") or (
            f"Component-level macro action '{plan.skill_name}' targets defects "
            f"{', '.join(plan.target_defects)} via {len(plan.component_edits)} atomic edits."
        )
        core = inst.get("core_contribution") or plan.objective
        method = inst.get("method") or self._plan_to_method_text(plan)
        experiments = inst.get("experiments") or self._plan_to_experiment_text(plan)
        risks = inst.get("risks") or (
            f"Guardrails: {'; '.join(plan.guardrails)} | "
            f"Budget delta: {plan.estimated_budget_delta}"
        )
        rationale = inst.get("rationale") or plan.compile_notes
        tags = _dedupe_keep_order(parent_state.tags + [plan.skill_name] + plan.target_defects)

        return IdeaState(
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
                "skill_prior_before": self._skill_prior_for_prompt(plan.skill_name),
                "guardrails": plan.guardrails,
                "constraints": ANTI_PATTERN_CONSTRAINTS,
                "llm_instantiated": bool(inst),
            },
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

    def _instantiate_skill_plan(
        self,
        plan: EditPlan,
        parent_state: IdeaState,
        bundle: MemoryBundle,
    ) -> Optional[Dict[str, Any]]:
        """Call LLM to instantiate a compiled skill plan into concrete, topic-specific content."""
        component_edits_text = self._plan_to_method_text(plan)
        validation_text = self._plan_to_experiment_text(plan)

        prompt = SKILL_INSTANTIATION_PROMPT.format(
            topic=self.topic,
            mature_idea=self.mature_idea or "None",
            parent_summary=parent_state.describe(),
            parent_components=", ".join(parent_state.components) if parent_state.components else "None",
            paper_context=self.paper_context,
            memory_bundle=bundle.to_prompt_block(),
            skill_name=plan.skill_name,
            plan_objective=plan.objective,
            target_defects=", ".join(plan.target_defects),
            component_edits=component_edits_text,
            validation_protocols=validation_text,
            guardrails="; ".join(plan.guardrails) if plan.guardrails else "None",
        )
        try:
            response = self.chat_fn(
                prompt,
                model=self.config.generation_model,
                temperature=self.config.generation_temperature,
                max_output_tokens=self.config.generation_max_tokens,
            )
            payload = parse_json_response(response)
            if isinstance(payload, list):
                payload = payload[0]
            if not isinstance(payload, dict):
                return None
            return payload
        except Exception as exc:
            log_message(
                self.logger, self.log_sink, "warning",
                "\u26a0\ufe0f  Skill instantiation failed for %s: %s", plan.skill_name, exc,
            )
            return None

    def _expand(self, node: IdeaNode, path: List[IdeaNode]) -> Tuple[Optional[IdeaNode], List[IdeaNode]]:
        bundle = self.memory_accessor.retrieve_bundle(
            query=(
                f"{self.topic}\n{node.state.title}\n"
                f"{node.state.core_contribution}\n"
                f"defects={','.join(node.state.target_defects)}"
            )
        )
        skill_candidates = self.skill_catalog.select_skills(
            defect_tags=node.state.target_defects,
            budget=node.state.budget,
            max_children=self.config.branching_factor,
        )

        payload_count = len(skill_candidates)
        pre_children = len(node.children)
        new_child: Optional[IdeaNode] = None

        log_message(
            self.logger,
            self.log_sink,
            "info",
            "[MCTS] Expand: selected %d skill candidate(s) for defects=%s",
            len(skill_candidates),
            ", ".join(node.state.target_defects),
        )
        for idx, sk in enumerate(skill_candidates):
            blueprint_str = ", ".join(sk.atomic_blueprint) if sk.atomic_blueprint else "none"
            log_message(
                self.logger,
                self.log_sink,
                "info",
                "[MCTS] Expand: skill[%d] name=%s | description=%s | atomic_blueprint=[%s]",
                idx,
                sk.name,
                sk.description,
                blueprint_str,
            )

        for skill in skill_candidates:
            plan = self.skill_catalog.compile_plan(
                skill=skill,
                parent_title=node.state.title,
                parent_components=node.state.components,
                target_defects=node.state.target_defects,
                budget=node.state.budget,
                memory_refs=bundle.referenced_ids(),
            )
            prior_constraints = self.skill_catalog.priors.get(skill.name, SkillUsagePrior()).rule_constraints
            if prior_constraints:
                plan.guardrails = _dedupe_keep_order(plan.guardrails + list(prior_constraints))

            # Filter forbidden atomic ops based on current component count
            current_count = len(node.state.components)
            filtered_edits: List[ComponentEdit] = []
            for edit in plan.component_edits:
                if current_count <= MIN_COMPONENTS and edit.op == AtomicEditOp.REMOVE_COMPONENT:
                    log_message(
                        self.logger,
                        self.log_sink,
                        "info",
                        "[MCTS] Expand: skill=%s BLOCKED %s on '%s' (component_count=%d <= MIN=%d)",
                        skill.name,
                        edit.op.value,
                        edit.component,
                        current_count,
                        MIN_COMPONENTS,
                    )
                    continue
                if current_count >= MAX_COMPONENTS and edit.op in (
                    AtomicEditOp.ADD_COMPONENT,
                    AtomicEditOp.GATE_COMPONENT,
                ):
                    # ADD_COMPONENT and GATE_COMPONENT both grow the component list
                    log_message(
                        self.logger,
                        self.log_sink,
                        "info",
                        "[MCTS] Expand: skill=%s BLOCKED %s on '%s' (component_count=%d >= MAX=%d)",
                        skill.name,
                        edit.op.value,
                        edit.component,
                        current_count,
                        MAX_COMPONENTS,
                    )
                    continue
                # Track how the count will change for subsequent edits
                if edit.op == AtomicEditOp.ADD_COMPONENT:
                    current_count += 1
                elif edit.op == AtomicEditOp.REMOVE_COMPONENT:
                    current_count -= 1
                elif edit.op == AtomicEditOp.GATE_COMPONENT:
                    current_count += 1  # gate may add a wrapper component
                filtered_edits.append(edit)
            plan.component_edits = filtered_edits

            # LLM instantiation: fill concrete content into the compiled plan
            instantiated = self._instantiate_skill_plan(plan, node.state, bundle)

            # Apply component_mapping from LLM instantiation to plan.component_edits
            if instantiated and isinstance(instantiated.get("component_mapping"), dict):
                mapping = instantiated["component_mapping"]
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Expand: skill=%s component_mapping=%s",
                    skill.name,
                    json.dumps(mapping, ensure_ascii=False),
                )
                # Apply edit_reasons from LLM instantiation to plan.component_edits
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
                    # Rebuild details with mapped names
                    if edit.op == AtomicEditOp.REWIRE:
                        edit.details = f"Rewire {edit.component} -> {edit.target}"
                    elif edit.op == AtomicEditOp.REPLACE_COMPONENT:
                        edit.details = f"Replace {edit.target} with {edit.component}"
                    elif edit.op == AtomicEditOp.GATE_COMPONENT:
                        cond = f" under condition '{edit.condition}'" if edit.condition else ""
                        edit.details = f"Gate {edit.component}{cond}"
                    elif edit.op == AtomicEditOp.ADD_COMPONENT:
                        edit.details = f"ADD_COMPONENT on {edit.component}"

            # Log the atomic operations in the compiled plan
            protocol_names: List[str] = []
            for edit_idx, edit in enumerate(plan.component_edits):
                # Collect ADD_PROTOCOL ops into a single summary line
                if edit.op == AtomicEditOp.ADD_PROTOCOL:
                    protocol_names.append(edit.component)
                    continue
                op_dict = {
                    "op": edit.op.value if hasattr(edit.op, 'value') else edit.op,
                    "component": edit.component,
                    "target": edit.target,
                    "condition": edit.condition,
                    "details": edit.details or "",
                    "reason": edit.reason or "",
                }
                op_str = json.dumps(op_dict, ensure_ascii=False, indent=2)
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Expand: skill=%s atomic_op[%d]\n%s",
                    skill.name,
                    edit_idx,
                    op_str,
                )
            if protocol_names:
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Expand: skill=%s protocols=[%s]",
                    skill.name,
                    ", ".join(protocol_names),
                )

            # Log the skill instantiation output
            if instantiated:
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Expand: skill=%s instantiation output keys=%s",
                    skill.name,
                    list(instantiated.keys()),
                )
                for k, v in instantiated.items():
                    log_message(
                        self.logger,
                        self.log_sink,
                        "info",
                        "[MCTS] Expand: skill=%s output[%s]=%s",
                        skill.name,
                        k,
                        str(v) if v else "",
                    )
            else:
                log_message(
                    self.logger,
                    self.log_sink,
                    "warning",
                    "[MCTS] Expand: skill=%s instantiation returned no output",
                    skill.name,
                )

            child_state = self._materialize_child_state(node.state, plan, instantiated)

            child_node = attach_child(
                node,
                child_state,
                signature_nodes=self.signature_nodes,
                id_iter=self._id_iter,
                idea_node_cls=IdeaNode,
                operator_application_cls=OperatorApplication,
                logger=self.logger,
                log_sink=self.log_sink,
            )
            if child_node is None:
                continue
            cached_eval = get_best_cached_evaluation(child_state.signature, self.evaluation_cache)
            if cached_eval:
                child_node.evaluation = cached_eval
            if new_child is None and child_node.visits == 0:
                new_child = child_node

        node.expanded = True
        log_message(
            self.logger,
            self.log_sink,
            "info",
            "[MCTS] Expansion summary (skill-plan) payloads=%d children_before=%d children_after=%d new_children=%d",
            payload_count,
            pre_children,
            len(node.children),
            len(node.children) - pre_children,
        )
        # Log component count for each child idea
        for child in node.children[pre_children:]:
            log_message(
                self.logger,
                self.log_sink,
                "info",
                "[MCTS] Expand: child idea=%s component_count=%d components=%s",
                child.state.title[:80],
                len(child.state.components),
                child.state.components,
            )

        if new_child is None and node.children:
            new_child = min(node.children, key=lambda c: c.visits)
        if new_child:
            return new_child, path + [new_child]
        return node, path

    def _simulate(
        self,
        node: IdeaNode,
        path: List[IdeaNode],
        experiences: List[Dict[str, Any]],
    ) -> Optional[IdeaEvaluation]:
        path_summary_text = path_summary(path)
        path_key = path_cache_key(node.state.signature, path_summary_text)

        cached_evaluation = get_cached_evaluation(
            node.state.signature,
            path_key,
            self.evaluation_cache,
        )
        if cached_evaluation:
            cached_evaluation.alignment_weight = self.config.alignment_weight
            cached_evaluation.complexity_weight = self.config.complexity_weight
            cached_evaluation.novelty_weight = self.config.novelty_weight
            cached_evaluation.impact_weight = self.config.impact_weight
            cached_evaluation.feasibility_weight = self.config.feasibility_weight
            cached_evaluation.clarity_weight = self.config.clarity_weight
            cached_evaluation.conciseness_weight = self.config.conciseness_weight
            cached_evaluation.risk_weight = self.config.risk_weight
            cached_evaluation.protocol_weight = self.config.protocol_weight

            node.evaluation = cached_evaluation
            node.latest_path_summary = path_summary_text
            prev_len = len(experiences)
            maybe_record_experience(
                path_key,
                node,
                cached_evaluation,
                path_summary_text,
                experiences,
                self.experience_cache,
                self.memory_accessor,
                self.config.min_confidence_for_memory,
            )
            if len(experiences) > prev_len:
                experience = experiences[-1]
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Memory recorded (cache) defect=%s action=%s lift=%s idea=%s",
                    experience.get("defect"),
                    experience.get("action"),
                    experience.get("lift"),
                    experience.get("idea"),
                )
            return cached_evaluation

        prompt = self.evaluation_prompt.format(
            topic=self.topic,
            mature_idea=self.mature_idea or "None",
            analysis=self.analysis_blob,
            paper_context=self.paper_context,
            skill_output=json.dumps(node.state.edit_plan, ensure_ascii=False, indent=2)
            if node.state.edit_plan
            else "null",
            edit_plan=json.dumps(node.state.edit_plan, ensure_ascii=False, indent=2)
            if node.state.edit_plan
            else "null",
            skill_prior=json.dumps(self._skill_prior_for_prompt(node.state.operator), ensure_ascii=False, indent=2),
            idea=json.dumps(node.state.to_payload(), ensure_ascii=False, indent=2),
            path_summary=path_summary_text,
            defect_registry=format_defect_registry(),
        )

        try:
            response = self.chat_fn(
                prompt,
                model=self.config.evaluation_model,
                temperature=self.config.evaluation_temperature,
                max_output_tokens=self.config.evaluation_max_tokens,
            )
            payload = parse_json_response(response)
            if isinstance(payload, list):
                payload = payload[0]
            evaluation = IdeaEvaluation.from_payload(payload)
        except Exception as exc:
            log_message(self.logger, self.log_sink, "warning", "⚠️  Simulation failed: %s", exc)
            return None

        if evaluation.protocol_score <= 0.0:
            evaluation.protocol_score = self._compute_protocol_score(node.state.edit_plan)

        evaluation.alignment_weight = self.config.alignment_weight
        evaluation.complexity_weight = self.config.complexity_weight
        evaluation.novelty_weight = self.config.novelty_weight
        evaluation.impact_weight = self.config.impact_weight
        evaluation.feasibility_weight = self.config.feasibility_weight
        evaluation.clarity_weight = self.config.clarity_weight
        evaluation.conciseness_weight = self.config.conciseness_weight
        evaluation.risk_weight = self.config.risk_weight
        evaluation.protocol_weight = self.config.protocol_weight

        cache_evaluation(node.state.signature, path_key, evaluation, self.evaluation_cache)
        node.evaluation = evaluation
        node.latest_path_summary = path_summary_text

        prev_len = len(experiences)
        maybe_record_experience(
            path_key,
            node,
            evaluation,
            path_summary_text,
            experiences,
            self.experience_cache,
            self.memory_accessor,
            self.config.min_confidence_for_memory,
        )
        if len(experiences) > prev_len:
            experience = experiences[-1]
            log_message(
                self.logger,
                self.log_sink,
                "info",
                "[MCTS] Memory recorded defect=%s action=%s lift=%s idea=%s",
                experience.get("defect"),
                experience.get("action"),
                experience.get("lift"),
                experience.get("idea"),
            )

        return evaluation

    def _record_skill_prior_memory(
        self,
        node: IdeaNode,
        evaluation: IdeaEvaluation,
        prior: SkillUsagePrior,
    ) -> None:
        if node.state.operator == "seed":
            return

        plan = node.state.edit_plan or {}
        edits = plan.get("component_edits") if isinstance(plan.get("component_edits"), list) else []
        action_tokens = [node.state.operator]
        for edit in edits:
            if not isinstance(edit, dict):
                continue
            op = str(edit.get("op", "")).strip()
            component = str(edit.get("component", "")).strip()
            if op:
                action_tokens.append(f"{op}:{component}" if component else op)

        try:
            record = self.symbolic_memory.instantiate_symbolic_record(
                summary=f"Skill prior update for {node.state.operator}",
                pattern=f"topic={self.topic}; defects={','.join(node.state.target_defects)}",
                conditions=node.state.target_defects,
                actions=action_tokens,
                rationale=evaluation.feedback,
                expected_outcomes=[evaluation.defect_fix_summary],
                anti_patterns=evaluation.failure_modes,
                tags=["edit_operator_skill", node.state.operator],
                priority=max(0.0, min(1.0, evaluation.composite / 5.0)),
                confidence=max(0.0, min(1.0, prior.prior)),
                source="mcts_backprop",
                support_count=max(1, prior.attempts),
                metadata={
                    "skill_name": node.state.operator,
                    "prior": prior.prior,
                    "reward_ema": prior.reward_ema,
                    "attempts": prior.attempts,
                    "successes": prior.successes,
                },
            )
            self.symbolic_memory.upsert_normal_records([record], agent_id="idea_agent")
            self._symbolic_dirty = True
        except Exception as exc:
            log_message(
                self.logger,
                self.log_sink,
                "warning",
                "⚠️  Failed to upsert skill prior symbolic memory: %s",
                exc,
            )

    def _update_skill_prior(self, node: IdeaNode, evaluation: IdeaEvaluation) -> None:
        skill_name = node.state.operator
        if not skill_name or skill_name == "seed":
            return

        normalized_reward = max(0.0, min(1.0, evaluation.composite / 5.0))
        prior = self.skill_catalog.update_prior(
            skill_name=skill_name,
            reward=normalized_reward,
            feedback=evaluation.feedback,
            failure_modes=evaluation.failure_modes,
            success_threshold=self.config.skill_prior_success_threshold,
        )
        node.state.skill_metrics["skill_prior_after"] = prior.to_dict()
        self._record_skill_prior_memory(node, evaluation, prior)

    def _backpropagate(self, path: List[IdeaNode], evaluation: IdeaEvaluation) -> None:
        score = evaluation.composite
        for hop in reversed(path):
            hop.visits += 1
            hop.value_sum += score

    def _extract_mature_idea_components(self, mature_idea: str, topic: str) -> List[str]:
        """Use LLM to extract 1-5 key components from the mature idea."""
        prompt = COMPONENT_EXTRACTION_PROMPT.format(
            mature_idea=mature_idea,
            topic=topic,
        )
        try:
            response = self.chat_fn(
                prompt,
                model=self.config.generation_model,
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
                    # Enforce 1-5 range
                    components = components[:MAX_COMPONENTS]
                    if components:
                        return components
        except Exception as exc:
            log_message(
                self.logger, self.log_sink, "warning",
                "⚠️  Component extraction from mature idea failed: %s", exc,
            )
        return []

    def search(self, topic: str, context: Dict[str, Any]) -> SearchResult:
        reset_search_state(self)
        self.topic = topic
        self.analysis_blob = format_analysis_blob(context.get("analysis", []))
        self.paper_context = context.get("paper_context") or "No curated papers available yet."
        self.mature_idea = (context.get("mature_idea") or "").strip()

        # Extract components from mature idea if provided
        self._mature_idea_components = []
        if self.mature_idea:
            self._mature_idea_components = self._extract_mature_idea_components(
                self.mature_idea, topic
            )
            if self._mature_idea_components:
                context["components"] = self._mature_idea_components
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Extracted %d component(s) from mature idea: %s",
                    len(self._mature_idea_components),
                    self._mature_idea_components,
                )

        root_state = build_root_state(topic, context, IdeaState)
        log_message(
            self.logger,
            self.log_sink,
            "info",
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

        self._persist_skill_prior_memory()

        best = best_candidate(root, SearchCandidate)
        pareto = pareto_candidates(root, SearchCandidate)
        cache_entries = sum(len(entries) for entries in self.evaluation_cache.values())

        return SearchResult(
            best=best,
            pareto=pareto,
            trace=self.trace,
            cache_size=cache_entries,
            experiences=experiences,
        )
