from __future__ import annotations

import json
import math
import hashlib
import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set

from omegaconf import OmegaConf
from tqdm import tqdm

from memory.api.faiss_memory_system_api import FAISSMemorySystem
from memory.api.slot_process_api import SlotProcess
from memory.memory_system.models import SemanticRecord, EpisodicRecord, ProceduralRecord
from memory.memory_system.utils import (
    _safe_dump_str,
    _multi_thread_run,
)
from agent import get_logger
from src.agents.idea_agent.utils.mcts_helpers import (
    parse_json_response,
    format_analysis_blob,
    format_edit_operators,
    clip_text,
)
from src.agents.idea_agent.utils.mcts_runtime import (
    AGGRESSIVE_OPERATOR_NAMES,
    ANTI_PATTERN_CONSTRAINTS,
    CONSERVATIVE_OPERATOR_NAMES,
    MODERATE_OPERATOR_NAMES,
    EDIT_OPERATORS,
    EditOperator,
    ExperimentPatch,
    IdeaContract,
    InvariantImpact,
    MemoryBundle,
    MemorySnippet,
    SkillDelta,
    SkillOutput,
    SkillValidation,
    attach_child,
    best_candidate,
    build_root_state,
    cache_evaluation,
    fallback_child_payloads,
    get_best_cached_evaluation,
    get_cached_evaluation,
    log_message,
    maybe_record_experience,
    new_node,
    pareto_candidates,
    parse_child_state,
    path_cache_key,
    path_summary,
    reset_search_state,
)

MAX_IDEA_TEXT = 800
MAX_RATIONALE_TEXT = 600
MAX_TITLE_TEXT = 256
MAX_LIST_ENTRIES = 10
MAX_REF_TEXT = 96

module_logger = get_logger()

def _load_mcts_defaults() -> Dict[str, Any]:
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
    skill_output: Optional[Dict[str, Any]] = None
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
        self.tags = [clip_text(tag, 48) for tag in self.tags[:MAX_LIST_ENTRIES]]
        self.target_defects = [
            clip_text(defect, 48) for defect in self.target_defects[:MAX_LIST_ENTRIES]
        ]
        self.memory_refs = [
            clip_text(ref, MAX_REF_TEXT) for ref in self.memory_refs[:MAX_LIST_ENTRIES]
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
        base = (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Core Contribution: {self.core_contribution}\n"
            f"Method: {self.method}\n"
            f"Experiments: {self.experiments}\n"
            f"Risks: {self.risks}\n"
            f"Tags: {', '.join(self.tags)}\n"
            f"Operator: {self.operator} targeting {self.target_defects or ['unspecified']}"
        )
        if self.skill_output:
            delta = self.skill_output.get("delta", {})
            intervention = delta.get("intervention", "")
            mechanism = delta.get("mechanism", "")
            return f"{base}\nDelta: {intervention}\nMechanism: {mechanism}"
        return base

    def to_payload(self) -> Dict[str, Any]:
        payload = {
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
        if self.skill_output:
            payload["skill_output"] = self.skill_output
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
    confidence: float
    failure_modes: List[str]
    fairness_protocol: str
    feedback: str
    defect_fix_summary: str
    lift_estimate: float
    alignment_weight: float = 0.2
    complexity_weight: float = 0.2

    def __post_init__(self) -> None:
        self.failure_modes = [
            clip_text(mode, 160) for mode in (self.failure_modes or [])[:MAX_LIST_ENTRIES]
        ]
        self.fairness_protocol = clip_text(self.fairness_protocol, MAX_IDEA_TEXT)
        self.feedback = clip_text(self.feedback, MAX_IDEA_TEXT)
        self.defect_fix_summary = clip_text(self.defect_fix_summary, MAX_IDEA_TEXT)

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
            alignment_score=_num("alignment_score"),
            complexity_penalty=_num("complexity_penalty"),
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
            "alignment_score": self.alignment_score,
            "complexity_penalty": self.complexity_penalty,
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
        penalty = 0.2 * self.risk + self.complexity_weight * self.complexity_penalty
        bonus = self.alignment_weight * self.alignment_score
        return positive + bonus - penalty


@dataclass
class OperatorApplication:
    operator: str
    defects: List[str]
    rationale: str
    memory_refs: List[str]

    def __post_init__(self) -> None:
        self.operator = clip_text(self.operator, 80)
        self.defects = [clip_text(defect, 48) for defect in self.defects[:MAX_LIST_ENTRIES]]
        self.rationale = clip_text(self.rationale, MAX_RATIONALE_TEXT)
        self.memory_refs = [
            clip_text(ref, MAX_REF_TEXT) for ref in self.memory_refs[:MAX_LIST_ENTRIES]
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
    max_iterations: int = _mcts_default("max_iterations", 128)
    max_depth: int = _mcts_default("max_depth", 4)
    branching_factor: int = _mcts_default("branching_factor", 3)
    exploration_constant: float = _mcts_default("exploration_constant", 1.15)
    generation_model: str = _mcts_default("generation_model", "gpt-4.1")
    evaluation_model: str = _mcts_default("evaluation_model", "gpt-4.1")
    generation_temperature: float = _mcts_default("generation_temperature", 0.65)
    evaluation_temperature: float = _mcts_default("evaluation_temperature", 0.01)
    generation_max_tokens: int = _mcts_default("generation_max_tokens", 8192)
    evaluation_max_tokens: int = _mcts_default("evaluation_max_tokens", 8192)
    min_confidence_for_memory: float = _mcts_default("min_confidence_for_memory", 0.6)
    pareto_top_k: int = _mcts_default("pareto_top_k", 5)
    alignment_weight: float = _mcts_default("alignment_weight", 0.2)
    complexity_weight: float = _mcts_default("complexity_weight", 0.2)
    min_anchor_coverage: float = _mcts_default("min_anchor_coverage", 0.7)
    conservative_depth: int = _mcts_default("conservative_depth", 1)
    aggressive_depth: int = _mcts_default("aggressive_depth", 2)
    enable_skill_repair: bool = _mcts_default("enable_skill_repair", False)


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
    idea_contract: Optional[Dict[str, Any]] = None


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
            log_message(
                self.logger,
                None,
                "info",
                "ℹ️ Skipping persistence because semantic store is unavailable.",
            )
            return
        
        slot_process = SlotProcess(llm_name="gpt-4.1", llm_backend="openai") # lazy loading
        try:
            # 1. Multi-threaded run for contexts transformation
            working_slots = slot_process.transfer_idea_agent_context_to_working_slots(experience)
            log_message(
                self.logger,
                None,
                "info",
                "[MCTS] Transferred experience to working slots (count=%d)",
                len(working_slots),
            )
            # 2. Multi-threaded run for slots filter and route
            _multi_thread_run(slot_process._multi_thread_filter_and_route_slot, working_slots, max_workers)
            # 3. Multi-threaded run for experience persistence
            _multi_thread_run(slot_process._multi_thread_transfer_slot_to_memory, slot_process.routed_slot_container, max_workers)
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
        
        if semantic and len(semantic_records) > 0:
            try:
                semantic.add(semantic_records, agent_id="idea_agent")
            except Exception as exc:
                log_message(
                    self.logger,
                    None,
                    "warning",
                    "⚠️  Failed to persist semantic records: %s",
                    exc,
                )
        if episodic and len(episodic_records) > 0:
            try:
                episodic.add(episodic_records, agent_id="idea_agent")
            except Exception as exc:
                log_message(
                    self.logger,
                    None,
                    "warning",
                    "⚠️  Failed to persist episodic records: %s",
                    exc,
                )
        if procedural and len(procedural_records) > 0:
            try:
                procedural.add(procedural_records, agent_id="idea_agent")
            except Exception as exc:
                log_message(
                    self.logger,
                    None,
                    "warning",
                    "⚠️  Failed to persist procedural records: %s",
                    exc,
                )

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
        generation_prompt: str,
        evaluation_prompt: str,
        contract_prompt: Optional[str] = None,
        skill_generation_prompt: Optional[str] = None,
        anchor_refiner_prompt: Optional[str] = None,
        skill_repair_prompt: Optional[str] = None,
        config: Optional[MCTSConfig] = None,
        memory_accessor: Optional[LongTermMemoryAccessor] = None,
        logger: Optional[logging.Logger] = None,
        log_sink: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self.chat_fn = chat_fn
        self.generation_prompt = generation_prompt
        self.evaluation_prompt = evaluation_prompt
        self.contract_prompt = contract_prompt
        self.skill_generation_prompt = skill_generation_prompt
        self.anchor_refiner_prompt = anchor_refiner_prompt
        self.skill_repair_prompt = skill_repair_prompt
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
        self.contract: Optional[IdeaContract] = None
        self.contract_mode: bool = False

    def _operator_pool_for_depth(self, depth: int) -> List[EditOperator]:
        if depth <= self.config.conservative_depth:
            allowed = CONSERVATIVE_OPERATOR_NAMES
        elif depth >= self.config.aggressive_depth:
            allowed = CONSERVATIVE_OPERATOR_NAMES | MODERATE_OPERATOR_NAMES | AGGRESSIVE_OPERATOR_NAMES
        else:
            allowed = CONSERVATIVE_OPERATOR_NAMES | MODERATE_OPERATOR_NAMES
        return [op for op in EDIT_OPERATORS if op.name in allowed]

    def _build_contract(self, mature_idea: str) -> Optional[IdeaContract]:
        if not self.contract_prompt:
            return None
        prompt = self.contract_prompt.format(mature_idea=mature_idea)
        try:
            response = self.chat_fn(
                prompt,
                model=self.config.generation_model,
                temperature=0.01,
                max_output_tokens=min(2048, self.config.generation_max_tokens),
            )
            payload = parse_json_response(response)
            if isinstance(payload, list):
                payload = payload[0]
            return IdeaContract.from_payload(payload)
        except Exception as exc:
            log_message(self.logger, self.log_sink, "warning", "⚠️  Contract build failed: %s", exc)
            return None

    def _contract_node_summary(self, node: IdeaNode) -> str:
        if not node.state.skill_output:
            return node.state.describe()
        delta = node.state.skill_output.get("delta", {})
        exp_patch = node.state.skill_output.get("experiment_patch", {})
        return (
            f"Title: {node.state.title}\n"
            f"Delta intervention: {delta.get('intervention','')}\n"
            f"Delta mechanism: {delta.get('mechanism','')}\n"
            f"Experiment patch: {exp_patch}\n"
            f"Risks: {node.state.risks}"
        )

    def _normalize_list(self, raw: Any) -> List[str]:
        if isinstance(raw, list):
            return [str(item).strip() for item in raw if str(item).strip()]
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            if "," in text:
                return [seg.strip() for seg in text.split(",") if seg.strip()]
            return [text]
        return []

    def _match_invariant_key(self, key: str, contract: IdeaContract) -> Optional[str]:
        if not key:
            return None
        key = str(key).strip()
        if not key:
            return None
        ids = {inv.inv_id for inv in contract.invariants()}
        if key in ids:
            return key
        lowered = key.lower()
        for inv in contract.invariants():
            if lowered == inv.inv_id.lower():
                return inv.inv_id
        for inv in contract.invariants():
            text_lower = inv.text.lower()
            if lowered in text_lower or text_lower in lowered:
                return inv.inv_id
        return None

    def _normalize_anchor_mapping(
        self, raw: Any, contract: IdeaContract
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if isinstance(raw, dict):
            items = raw.items()
        elif isinstance(raw, list):
            items = []
            for item in raw:
                if isinstance(item, dict):
                    items.append((item.get("invariant_id") or item.get("invariant"), item.get("anchor") or item.get("mapping")))
        else:
            items = []
        for key, value in items:
            inv_id = self._match_invariant_key(str(key), contract)
            if not inv_id:
                continue
            mapping[inv_id] = str(value).strip()
        return mapping

    def _normalize_invariant_impact(
        self, raw: Any, contract: IdeaContract
    ) -> Dict[str, InvariantImpact]:
        impact: Dict[str, InvariantImpact] = {}
        entries: List[Tuple[Any, Any]] = []
        if isinstance(raw, dict):
            entries = list(raw.items())
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    key = item.get("invariant_id") or item.get("invariant")
                    entries.append((key, item))
        for key, value in entries:
            inv_id = self._match_invariant_key(str(key), contract)
            if not inv_id:
                continue
            if isinstance(value, dict):
                status = str(value.get("status", "")).strip().lower()
                reason = str(value.get("reason", "")).strip()
                compensation = str(value.get("compensation", "")).strip()
            else:
                status = str(value).strip().lower()
                reason = ""
                compensation = ""
            impact[inv_id] = InvariantImpact(status=status or "preserve", reason=reason, compensation=compensation)
        return impact

    def _parse_skill_output(self, data: Dict[str, Any], contract: IdeaContract) -> Optional[SkillOutput]:
        if not isinstance(data, dict):
            return None
        delta_raw = data.get("delta") or {}
        exp_raw = data.get("experiment_patch") or {}
        delta = SkillDelta(
            intervention=clip_text(delta_raw.get("intervention", ""), MAX_IDEA_TEXT),
            mechanism=clip_text(delta_raw.get("mechanism", ""), MAX_IDEA_TEXT),
            implementation_notes=clip_text(delta_raw.get("implementation_notes", ""), MAX_IDEA_TEXT),
        )
        regression = self._normalize_list(
            exp_raw.get("regression_tests") or exp_raw.get("regression_test") or exp_raw.get("regression")
        )
        ablation = self._normalize_list(
            exp_raw.get("ablation_tests") or exp_raw.get("ablation_test") or exp_raw.get("ablation")
        )
        stress = self._normalize_list(
            exp_raw.get("stress_tests") or exp_raw.get("stress_test") or exp_raw.get("stress")
        )
        experiment_patch = ExperimentPatch(
            regression_tests=regression,
            ablation_tests=ablation,
            stress_tests=stress,
        )
        risk_patch = self._normalize_list(data.get("risk_patch"))
        introduced = self._normalize_list(data.get("introduced_concepts"))
        anchor_mapping = self._normalize_anchor_mapping(data.get("anchor_mapping"), contract)
        invariant_impact = self._normalize_invariant_impact(data.get("invariant_impact"), contract)
        memory_refs = self._normalize_list(data.get("memory_refs"))
        budget_impact = data.get("budget_impact") if isinstance(data.get("budget_impact"), dict) else {}
        return SkillOutput(
            delta=delta,
            experiment_patch=experiment_patch,
            risk_patch=risk_patch,
            introduced_concepts=introduced,
            anchor_mapping=anchor_mapping,
            invariant_impact=invariant_impact,
            memory_refs=memory_refs,
            budget_impact=budget_impact,
            raw=data,
        )

    def _estimate_mechanism_count(self, text: str) -> int:
        cleaned = (text or "").strip().lower()
        if not cleaned:
            return 0
        separators = [" and ", " + ", " plus ", " coupled with ", " combined with ", " & ", ";"]
        count = 1
        for sep in separators:
            count += cleaned.count(sep)
        return count

    def _compute_complexity_penalty(self, skill: SkillOutput) -> float:
        text = " ".join(
            [
                skill.delta.intervention,
                skill.delta.mechanism,
                skill.delta.implementation_notes,
                " ".join(skill.experiment_patch.regression_tests),
                " ".join(skill.experiment_patch.ablation_tests),
                " ".join(skill.experiment_patch.stress_tests),
            ]
        ).lower()
        penalty = 0
        for keyword in [
            "new module",
            "additional module",
            "auxiliary head",
            "extra head",
            "new head",
            "objective",
            "loss",
            "data channel",
            "dataset",
            "compute",
            "latency",
            "memory",
            "params",
        ]:
            if keyword in text:
                penalty += 1
        penalty += max(0, self._estimate_mechanism_count(skill.delta.mechanism) - 1) * 2
        penalty += len(skill.introduced_concepts)
        if skill.budget_impact:
            penalty += 1
        return min(5.0, float(penalty))

    def _compute_alignment_score(self, skill: SkillOutput, contract: IdeaContract, anchor_coverage: float) -> float:
        total = len(contract.invariants())
        if total == 0:
            return 5.0
        preserve = 0
        modify = 0
        for inv in contract.invariants():
            impact = skill.invariant_impact.get(inv.inv_id)
            if not impact:
                continue
            status = impact.status.lower()
            if status == "preserve":
                preserve += 1
            elif status == "modify":
                modify += 1
        ratio = (preserve + 0.5 * modify) / max(1, total)
        score = 5.0 * (0.6 * ratio + 0.4 * anchor_coverage)
        return max(0.0, min(5.0, score))

    def _budget_exceeded(self, contract: IdeaContract, skill: SkillOutput) -> bool:
        if not contract.budget_ceiling:
            return False
        budget = contract.budget_ceiling
        if skill.budget_impact:
            for key, ceiling in budget.items():
                impact = skill.budget_impact.get(key)
                if impact is None:
                    continue
                if isinstance(ceiling, (int, float)) and isinstance(impact, (int, float)):
                    if impact > ceiling:
                        return True
                if isinstance(impact, str) and any(token in impact.lower() for token in ["exceed", "over", ">", "increase"]):
                    return True
            return False
        # fallback heuristic based on implementation notes
        notes = skill.delta.implementation_notes.lower()
        if any(token in notes for token in ["exceed", "over budget", "too expensive", "budget breach"]):
            return True
        return False

    def _validate_skill_output(self, skill: SkillOutput, contract: IdeaContract) -> SkillValidation:
        errors: List[str] = []
        warnings: List[str] = []

        if len(skill.introduced_concepts) > 2:
            errors.append("introduced_concepts exceeds 2")

        if not skill.experiment_patch.regression_tests:
            errors.append("experiment_patch missing regression_tests")
        if not skill.experiment_patch.ablation_tests:
            errors.append("experiment_patch missing ablation_tests")
        if not skill.experiment_patch.stress_tests:
            errors.append("experiment_patch missing stress_tests")

        invariant_ids = contract.invariant_ids()
        mapped_ids = set(skill.anchor_mapping.keys())
        total_invariants = len(invariant_ids)
        anchor_coverage = (
            len(mapped_ids & set(invariant_ids)) / total_invariants if total_invariants else 1.0
        )
        if anchor_coverage < self.config.min_anchor_coverage:
            errors.append("anchor_mapping coverage below threshold")

        for inv in invariant_ids:
            impact = skill.invariant_impact.get(inv)
            if not impact:
                errors.append(f"invariant_impact missing entry for {inv}")
                continue
            status = impact.status.lower()
            if status == "violate":
                errors.append(f"invariant {inv} marked violate")
            if status == "modify" and not impact.compensation:
                errors.append(f"invariant {inv} modify requires compensation")

        mechanism_count = self._estimate_mechanism_count(skill.delta.mechanism)
        if mechanism_count == 0:
            errors.append("delta.mechanism missing")
        elif mechanism_count > 1:
            errors.append("main mechanism count > 1")

        if self._budget_exceeded(contract, skill):
            errors.append("budget_ceiling exceeded")

        alignment_score = self._compute_alignment_score(skill, contract, anchor_coverage)
        complexity_penalty = self._compute_complexity_penalty(skill)

        return SkillValidation(
            ok=not errors,
            errors=errors,
            warnings=warnings,
            alignment_score=alignment_score,
            complexity_penalty=complexity_penalty,
            anchor_coverage=anchor_coverage,
        )

    def _attempt_skill_repair(
        self,
        skill_output: SkillOutput,
        errors: List[str],
        parent_state: IdeaState,
        contract: IdeaContract,
    ) -> Optional[SkillOutput]:
        if not (self.config.enable_skill_repair and self.skill_repair_prompt):
            return None
        prompt = self.skill_repair_prompt.format(
            idea_contract=contract.to_prompt_block(),
            parent_idea=json.dumps(parent_state.to_payload(), ensure_ascii=False, indent=2),
            skill_output=json.dumps(skill_output.to_dict(), ensure_ascii=False, indent=2),
            errors="; ".join(errors),
        )
        try:
            response = self.chat_fn(
                prompt,
                model=self.config.generation_model,
                temperature=self.config.generation_temperature,
                max_output_tokens=min(2048, self.config.generation_max_tokens),
            )
            payload = parse_json_response(response)
            return self._parse_skill_output(payload, contract)
        except Exception as exc:
            log_message(self.logger, self.log_sink, "warning", "⚠️  Skill repair failed: %s", exc)
            return None

    def _materialize_child_state(
        self,
        parent_state: IdeaState,
        skill_output: SkillOutput,
        contract: IdeaContract,
        operator: str,
        target_defects: List[str],
    ) -> Optional[IdeaState]:
        if not self.anchor_refiner_prompt:
            return None
        prompt = self.anchor_refiner_prompt.format(
            idea_contract=contract.to_prompt_block(),
            parent_idea=json.dumps(parent_state.to_payload(), ensure_ascii=False, indent=2),
            skill_output=json.dumps(skill_output.to_dict(), ensure_ascii=False, indent=2),
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
            tags = payload.get("tags")
            return IdeaState(
                title=str(payload.get("title", "Untitled idea")).strip(),
                abstract=str(payload.get("abstract", "")).strip(),
                core_contribution=str(payload.get("core_contribution", "")).strip(),
                method=str(payload.get("method", "")).strip(),
                experiments=str(payload.get("experiments", "")).strip(),
                risks=str(payload.get("risks", "")).strip(),
                tags=tags if isinstance(tags, list) else [str(tags)] if tags else [operator],
                operator=operator,
                target_defects=target_defects,
                rationale=skill_output.delta.intervention,
                memory_refs=skill_output.memory_refs,
                skill_output=skill_output.to_dict(),
            )
        except Exception as exc:
            log_message(self.logger, self.log_sink, "warning", "⚠️  Anchor refiner failed: %s", exc)
            return None

    def _fallback_skill_payloads(
        self,
        node: IdeaNode,
        bundle: MemoryBundle,
        operators: List[EditOperator],
        contract: IdeaContract,
    ) -> List[Dict[str, Any]]:
        referenced_ids = bundle.referenced_ids()
        invariants = contract.invariant_ids()
        anchor_mapping = {inv: "preserve via parent mechanism/experiment" for inv in invariants}
        invariant_impact = {
            inv: {"status": "preserve", "reason": "contract preservation", "compensation": ""}
            for inv in invariants
        }
        experiment_patch = {
            "regression_tests": [
                "Reproduce parent key metrics under the same protocol/invariants."
            ],
            "ablation_tests": ["Remove the delta intervention and compare to parent baseline."],
            "stress_tests": ["Stress test failure modes aligned to evaluation invariants."],
        }
        payloads: List[Dict[str, Any]] = []
        for idx, op in enumerate(operators[: self.config.branching_factor]):
            payloads.append(
                {
                    "operator": op.name,
                    "target_defects": op.defects,
                    "skill_output": {
                        "delta": {
                            "intervention": f"Apply {op.name} to refine the parent mechanism without leaving allowed_axes.",
                            "mechanism": f"Single {op.name} intervention anchored to {node.state.title}.",
                            "implementation_notes": "Keep implementation within budget ceiling; no extra modules beyond the single intervention.",
                        },
                        "experiment_patch": experiment_patch,
                        "risk_patch": ["Fallback delta; validate carefully for contract alignment."],
                        "introduced_concepts": [],
                        "anchor_mapping": anchor_mapping,
                        "invariant_impact": invariant_impact,
                        "memory_refs": referenced_ids,
                    },
                }
            )
        return payloads

    def search(self, topic: str, context: Dict[str, Any]) -> SearchResult:
        reset_search_state(self)
        self.topic = topic
        self.analysis_blob = format_analysis_blob(context.get("analysis", []))
        self.paper_context = context.get("paper_context") or "No curated papers available yet."
        self.contract = None
        self.contract_mode = False
        mature_idea = (context.get("mature_idea") or "").strip()
        if mature_idea:
            contract = self._build_contract(mature_idea)
            if contract:
                self.contract = contract
                self.contract_mode = True
                context = dict(context)
                context["idea_contract"] = contract
            else:
                log_message(
                    self.logger,
                    self.log_sink,
                    "warning",
                    "⚠️  Contract mode requested but contract build failed; falling back to legacy MCTS.",
                )
        root_state = build_root_state(topic, context, IdeaState)
        root = new_node(
            root_state,
            depth=0,
            parent=None,
            signature_nodes=self.signature_nodes,
            id_iter=self._id_iter,
            idea_node_cls=IdeaNode,
            operator_application_cls=OperatorApplication,
        )
        experiences = []

        for iteration in tqdm(range(self.config.max_iterations)):
            leaf, path = self._select(root)
            current_depth = len(path) - 1
            if current_depth >= self.config.max_depth:
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Iteration %d generation skipped (max_depth=%d) leaf_id=%s depth=%d",
                    iteration,
                    self.config.max_depth,
                    leaf.node_id,
                    current_depth,
                )
                target = leaf
                rollout_path = path
            else:
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Iteration %d generation start leaf_id=%s depth=%d",
                    iteration,
                    leaf.node_id,
                    current_depth,
                )
                target, rollout_path = self._expand(leaf, path)
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Iteration %d generation done target_id=%s path_depth=%d children=%d",
                    iteration,
                    target.node_id if target else "None",
                    (len(rollout_path) - 1) if rollout_path else -1,
                    len(leaf.children),
                )
            if target is None:
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Iteration %d evaluate skipped (no target)",
                    iteration,
                )
                continue
            log_message(
                self.logger,
                self.log_sink,
                "info",
                "[MCTS] Iteration %d evaluate start node_id=%s path_depth=%d",
                iteration,
                target.node_id,
                len(rollout_path) - 1,
            )
            evaluation = self._simulate(target, rollout_path, experiences)
            if evaluation is None:
                log_message(
                    self.logger,
                    self.log_sink,
                    "info",
                    "[MCTS] Iteration %d evaluate skipped (no evaluation)",
                    iteration,
                )
                continue
            self._backpropagate(rollout_path, evaluation)
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
                    "skill_output": target.state.skill_output,
                    "skill_metrics": target.state.skill_metrics,
                }
            )

        best = best_candidate(root, SearchCandidate)
        pareto = pareto_candidates(root, SearchCandidate)
    
        cache_entries = sum(len(entries) for entries in self.evaluation_cache.values())
        return SearchResult(
            best=best,
            pareto=pareto,
            trace=self.trace,
            cache_size=cache_entries,
            experiences=experiences,
            idea_contract=self.contract.to_dict() if self.contract else None,
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
        if self.contract_mode:
            return self._expand_contract(node, path)
        return self._expand_legacy(node, path)

    def _expand_legacy(self, node: IdeaNode, path: List[IdeaNode]) -> Tuple[Optional[IdeaNode], List[IdeaNode]]:
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
        children_payload: List[Dict[str, Any]]
        try:
            response = self.chat_fn(
                prompt,
                model=self.config.generation_model,
                temperature=self.config.generation_temperature,
                max_output_tokens=min(2048, self.config.generation_max_tokens),
            )
            log_message(self.logger, self.log_sink, "info", "[MCTS] Generation response: %s", response)
            if not response or not response.strip():
                raise ValueError("Empty response from generation model")
            payload = parse_json_response(response)
            if isinstance(payload, list):
                payload = payload[0] 
            children_payload = payload.get("children", [])[: self.config.branching_factor]
        except Exception as exc:
            log_message(
                self.logger,
                self.log_sink,
                "warning",
                "⚠️  Expansion failed: %s. Falling back to heuristic children.",
                exc,
            )
            children_payload = fallback_child_payloads(
                node,
                bundle,
                EDIT_OPERATORS,
                self.config.branching_factor,
            )

        new_child: Optional[IdeaNode] = None
        for child_data in children_payload:
            state = parse_child_state(child_data, IdeaState)
            child_node = attach_child(
                node,
                state,
                signature_nodes=self.signature_nodes,
                id_iter=self._id_iter,
                idea_node_cls=IdeaNode,
                operator_application_cls=OperatorApplication,
                logger=self.logger,
                log_sink=self.log_sink,
            )
            if child_node is None:
                continue
            cached_eval = get_best_cached_evaluation(state.signature, self.evaluation_cache)
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

    def _expand_contract(self, node: IdeaNode, path: List[IdeaNode]) -> Tuple[Optional[IdeaNode], List[IdeaNode]]:
        if not self.contract:
            return self._expand_legacy(node, path)
        bundle = self.memory_accessor.retrieve_bundle(
            query=f"{self.topic}\n{node.state.title}\n{node.state.core_contribution}"
        )
        operator_pool = self._operator_pool_for_depth(node.depth)
        prompt_template = self.skill_generation_prompt or self.generation_prompt
        prompt = prompt_template.format(
            topic=self.topic,
            current_summary=self._contract_node_summary(node),
            paper_context=self.paper_context,
            memory_bundle=bundle.to_prompt_block(),
            edit_operators=format_edit_operators(operator_pool),
            max_children=self.config.branching_factor,
            constraints="\n".join(ANTI_PATTERN_CONSTRAINTS),
            idea_contract=self.contract.to_prompt_block(),
        )
        children_payload: List[Dict[str, Any]]
        try:
            response = self.chat_fn(
                prompt,
                model=self.config.generation_model,
                temperature=self.config.generation_temperature,
                max_output_tokens=min(2048, self.config.generation_max_tokens),
            )
            if not response or not response.strip():
                raise ValueError("Empty response from generation model")
            payload = parse_json_response(response)
            if isinstance(payload, list):
                payload = payload[0]
            children_payload = payload.get("children", [])[: self.config.branching_factor]
        except Exception as exc:
            log_message(
                self.logger,
                self.log_sink,
                "warning",
                "⚠️  Contract expansion failed: %s. Falling back to heuristic skill deltas.",
                exc,
            )
            children_payload = self._fallback_skill_payloads(
                node,
                bundle,
                operator_pool,
                self.contract,
            )

        new_child: Optional[IdeaNode] = None
        allowed_names = {op.name for op in operator_pool}
        for child_data in children_payload:
            operator = str(child_data.get("operator", "")).strip()
            if operator not in allowed_names:
                continue
            target_defects = self._normalize_list(child_data.get("target_defects")) or ["unspecified_defect"]
            skill_data = child_data.get("skill_output") or {}
            skill_output = self._parse_skill_output(skill_data, self.contract)
            if not skill_output:
                continue
            validation = self._validate_skill_output(skill_output, self.contract)
            if not validation.ok:
                repaired = self._attempt_skill_repair(
                    skill_output,
                    validation.errors,
                    node.state,
                    self.contract,
                )
                if repaired:
                    validation = self._validate_skill_output(repaired, self.contract)
                    if validation.ok:
                        skill_output = repaired
                if not validation.ok:
                    log_message(
                        self.logger,
                        self.log_sink,
                        "info",
                        "[MCTS] Pruned child from operator=%s due to violations: %s",
                        operator,
                        "; ".join(validation.errors),
                    )
                    continue
            child_state = self._materialize_child_state(
                parent_state=node.state,
                skill_output=skill_output,
                contract=self.contract,
                operator=operator,
                target_defects=target_defects,
            )
            if not child_state:
                continue
            child_state.skill_metrics = {
                "alignment_score": validation.alignment_score,
                "complexity_penalty": validation.complexity_penalty,
                "anchor_coverage": validation.anchor_coverage,
            }
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
        if new_child is None and node.children:
            new_child = min(node.children, key=lambda c: c.visits)
        if new_child:
            return new_child, path + [new_child]
        return node, path

    def _simulate(self, node: IdeaNode, path: List[IdeaNode], experiences: List[Dict[str, Any]]) -> Optional[IdeaEvaluation]:
        path_summary_text = path_summary(path)
        path_key = path_cache_key(node.state.signature, path_summary_text)
        cached_evaluation = get_cached_evaluation(
            node.state.signature,
            path_key,
            self.evaluation_cache,
        )
        if cached_evaluation:
            log_message(
                self.logger,
                self.log_sink,
                "info",
                "[MCTS] Simulate cache hit for signature=%s path_key=%s",
                node.state.signature,
                path_key,
            )
            cached_evaluation.alignment_weight = self.config.alignment_weight
            cached_evaluation.complexity_weight = self.config.complexity_weight
            if self.contract_mode and node.state.skill_metrics:
                cached_evaluation.alignment_score = node.state.skill_metrics.get(
                    "alignment_score", cached_evaluation.alignment_score
                )
                cached_evaluation.complexity_penalty = node.state.skill_metrics.get(
                    "complexity_penalty", cached_evaluation.complexity_penalty
                )
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

        log_message(
            self.logger,
            self.log_sink,
            "info",
            "[MCTS] Simulate start for signature=%s path_key=%s",
            node.state.signature,
            path_key,
        )
        prompt = self.evaluation_prompt.format(
            topic=self.topic,
            analysis=self.analysis_blob,
            paper_context=self.paper_context,
            idea_contract=self.contract.to_prompt_block() if self.contract else "None",
            skill_output=json.dumps(
                node.state.skill_output, ensure_ascii=False, indent=2
            )
            if node.state.skill_output
            else "null",
            idea=json.dumps(node.state.to_payload(), ensure_ascii=False, indent=2),
            path_summary=path_summary_text,
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
                payload = payload[0] # Sometimes returns a list of evaluations, we take the first one.
            evaluation = IdeaEvaluation.from_payload(payload)
            evaluation.alignment_weight = self.config.alignment_weight
            evaluation.complexity_weight = self.config.complexity_weight
        except Exception as exc:
            log_message(self.logger, self.log_sink, "warning", "⚠️  Simulation failed: %s", exc)
            log_message(
                self.logger,
                self.log_sink,
                "info",
                "[MCTS] Simulate returning None for signature=%s path_key=%s",
                node.state.signature,
                path_key,
            )
            return None

        if self.contract_mode and node.state.skill_metrics:
            evaluation.alignment_score = node.state.skill_metrics.get(
                "alignment_score", evaluation.alignment_score
            )
            evaluation.complexity_penalty = node.state.skill_metrics.get(
                "complexity_penalty", evaluation.complexity_penalty
            )

        cache_evaluation(node.state.signature, path_key, evaluation, self.evaluation_cache)
        node.evaluation = evaluation
        node.latest_path_summary = path_summary_text
        log_message(
            self.logger,
            self.log_sink,
            "info",
            "[MCTS] Simulate success for signature=%s path_key=%s score=%.4f",
            node.state.signature,
            path_key,
            evaluation.composite,
        )

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

    def _backpropagate(self, path: List[IdeaNode], evaluation: IdeaEvaluation) -> None:
        score = evaluation.composite
        for hop in reversed(path):
            hop.visits += 1
            hop.value_sum += score
