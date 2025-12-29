from __future__ import annotations

import json
import math
import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
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

logger = get_logger()


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
]

ANTI_PATTERN_CONSTRAINTS = [
    "No feature dumping: every add-on must map to a measured defect.",
    "Always declare baseline + ablation protocols for fairness.",
    "Describe at least one deliberate failure-mode surfacing plan.",
    "Constrain resource usage; note instrumentation for guardrails.",
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
    ) -> None:
        self.semantic_cfg = semantic_cfg or {}
        self.episodic_cfg = episodic_cfg or {}
        self.procedural_cfg = procedural_cfg or {}
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
                logger.warning(
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
            logger.warning("⚠️  Memory query failed (%s): %s", prefix, exc)
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
            logger.info("ℹ️ Skipping persistence because semantic store is unavailable.")
            return
        
        slot_process = SlotProcess(llm_name="mimo-v2-flash", llm_backend="openai") # lazy loading
        try:
            # 1. Multi-threaded run for contexts transformation
            working_slots = slot_process.transfer_idea_agent_context_to_working_slots(experience)
            print("[Info] Transferred experience to working slots, total slots:", len(working_slots))
            # 2. Multi-threaded run for slots filter and route
            _multi_thread_run(slot_process._multi_thread_filter_and_route_slot, working_slots, max_workers)
            # 3. Multi-threaded run for experience persistence
            _multi_thread_run(slot_process._multi_thread_transfer_slot_to_memory, slot_process.routed_slot_container, max_workers)
        except Exception as exc:
            logger.warning("⚠️  Failed to persist experience: %s", exc)
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
                logger.warning("⚠️  Failed to persist semantic records: %s", exc)
        if episodic and len(episodic_records) > 0:
            try:
                episodic.add(episodic_records, agent_id="idea_agent")
            except Exception as exc:
                logger.warning("⚠️  Failed to persist episodic records: %s", exc)
        if procedural and len(procedural_records) > 0:
            try:
                procedural.add(procedural_records, agent_id="idea_agent")
            except Exception as exc:
                logger.warning("⚠️  Failed to persist procedural records: %s", exc)

        print(f"[Debug] Size of semantic_records: {len(semantic_records)}, episodic_records: {len(episodic_records)}, procedural_records: {len(procedural_records)}")


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
            0.25 * self.novelty
            + 0.25 * self.impact
            + 0.25 * self.feasibility
            + 0.15 * self.clarity
            + 0.10 * self.conciseness
        )
        penalty = 0.25 * self.risk
        return positive - penalty


@dataclass
class OperatorApplication:
    operator: str
    defects: List[str]
    rationale: str
    memory_refs: List[str]


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
        steps = []
        for hop in self.path():
            steps.append(
                f"{hop.state.title} [{hop.transformation.operator}] -> defects {hop.transformation.defects}"
            )
        return " | ".join(steps)


@dataclass
class MCTSConfig:
    max_iterations: int = 5
    max_depth: int = 3
    branching_factor: int = 3
    exploration_constant: float = 1.2
    generation_model: str = "mimo-v2-flash"
    evaluation_model: str = "mimo-v2-flash"
    generation_temperature: float = 0.4
    evaluation_temperature: float = 0.0
    min_confidence_for_memory: float = 0.0
    pareto_top_k: int = 3


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
    ) -> None:
        self.chat_fn = chat_fn
        self.generation_prompt = generation_prompt
        self.evaluation_prompt = evaluation_prompt
        self.config = config or MCTSConfig()
        self.memory_accessor = memory_accessor or LongTermMemoryAccessor()
        self._id_iter = itertools.count()
        self.signature_nodes: Dict[str, List[IdeaNode]] = {}
        self.evaluation_cache: Dict[str, IdeaEvaluation] = {}
        self.trace: List[Dict[str, Any]] = []
        self.topic: str = ""
        self.analysis_blob: str = ""

    def search(self, topic: str, context: Dict[str, Any]) -> SearchResult:
        self.topic = topic
        self.analysis_blob = self._format_analysis(context.get("analysis", []))
        root_state = self._build_root_state(topic, context)
        root = self._new_node(root_state, depth=0, parent=None)
        experiences = []
        self.trace = []

        for iteration in tqdm(range(self.config.max_iterations)):
            leaf = self._select(root)
            if leaf.depth >= self.config.max_depth:
                target = leaf
            else:
                target = self._expand(leaf)
            if target is None:
                continue
            evaluation = self._simulate(target, experiences)
            if evaluation is None:
                continue
            self._backpropagate(target, evaluation)
            self.trace.append(
                {
                    "iteration": iteration,
                    "node_id": target.node_id,
                    "title": target.state.title,
                    "score": evaluation.composite,
                    "visits": target.visits,
                    "path": target.path_summary(),
                }
            )

        best = self._best_candidate(root)
        pareto = self._pareto_candidates(root)
    

        return SearchResult(
            best=best,
            pareto=pareto,
            trace=self.trace,
            cache_size=len(self.evaluation_cache),
            experiences=experiences,
        )

    def _new_node(
        self,
        state: IdeaState,
        depth: int,
        parent: Optional[IdeaNode],
    ) -> IdeaNode:
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
        self.signature_nodes.setdefault(state.signature, []).append(node)
        if parent:
            parent.children.append(node)
        return node

    def _format_analysis(self, analysis: List[Any]) -> str:
        if not analysis:
            return "No prior analysis."
        try:
            if isinstance(analysis[-1], dict):
                return json.dumps(analysis[-1], ensure_ascii=False, indent=2)
            return str(analysis[-1])
        except Exception:
            return str(analysis)

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

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """
        LLM responses occasionally include code fences or extra commentary.
        This helper strips the noise and extracts the first JSON object/array well-formed enough for json.loads.
        """
        text = (raw or "").strip()
        if not text:
            raise ValueError("Empty response")
        if text.startswith("```"):
            fence_end = text.find("\n")
            if fence_end != -1:
                text = text[fence_end + 1 :]
            if text.endswith("```"):
                text = text[: -3]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for idx, ch in enumerate(text):
                if ch in "{[":
                    try:
                        parsed, _ = decoder.raw_decode(text[idx:])
                        return parsed
                    except json.JSONDecodeError:
                        continue
        raise ValueError(f"Unable to parse JSON from response: {text[:200]}")

    def _select(self, node: IdeaNode) -> IdeaNode:
        current = node
        while current.children and current.expanded:
            current = max(
                current.children,
                key=lambda child: child.uct_value(
                    parent_visits=current.visits or 1,
                    exploration_constant=self.config.exploration_constant,
                ),
            )
        return current

    def _expand(self, node: IdeaNode) -> Optional[IdeaNode]:
        bundle = self.memory_accessor.retrieve_bundle(
            query=f"{self.topic}\n{node.state.title}\n{node.state.core_contribution}"
        )
        prompt = self.generation_prompt.format(
            topic=self.topic,
            current_summary=node.state.describe(),
            memory_bundle=bundle.to_prompt_block(),
            edit_operators=self._format_edit_ops(),
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
                max_tokens=4096,
            )
            print(f"[Debug] Generation response: {response}")
            if not response or not response.strip():
                raise ValueError("Empty response from generation model")
            payload = self._parse_json_response(response)
            children_payload = payload.get("children", [])[: self.config.branching_factor]
        except Exception as exc:
            logger.warning("⚠️  Expansion failed: %s. Falling back to heuristic children.", exc)
            children_payload = self._fallback_child_payloads(node, bundle)

        new_child: Optional[IdeaNode] = None
        for child_data in children_payload:
            state = self._parse_child_state(child_data)
            existing_nodes = self.signature_nodes.get(state.signature)
            if existing_nodes:
                # Reuse canonical state to avoid diverging copies.
                state = existing_nodes[0].state
            child_node = self._new_node(state, depth=node.depth + 1, parent=node)
            if existing_nodes and existing_nodes[0].evaluation:
                child_node.evaluation = existing_nodes[0].evaluation
                self.evaluation_cache[state.signature] = existing_nodes[0].evaluation
            if new_child is None:
                new_child = child_node
        node.expanded = True
        return new_child or (node.children[0] if node.children else node)

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

    def _simulate(self, node: IdeaNode, experiences: List[Dict[str, Any]]) -> Optional[IdeaEvaluation]:
        if node.state.signature in self.evaluation_cache:
            evaluation = self.evaluation_cache[node.state.signature]
            node.evaluation = evaluation
            return evaluation

        prompt = self.evaluation_prompt.format(
            topic=self.topic,
            analysis=self.analysis_blob,
            idea=json.dumps(node.state.to_payload(), ensure_ascii=False, indent=2),
            path_summary=node.path_summary(),
        )
        try:
            response = self.chat_fn(
                prompt,
                model=self.config.evaluation_model,
                temperature=self.config.evaluation_temperature,
                max_tokens=4096,
            )
            payload = self._parse_json_response(response)
            evaluation = IdeaEvaluation.from_payload(payload)
        except Exception as exc:
            logger.warning("⚠️  Simulation failed: %s", exc)
            return None

        self.evaluation_cache[node.state.signature] = evaluation
        node.evaluation = evaluation

        experience = self._harvest_experience(node, evaluation)
        if experience:
            self.memory_accessor.persist_experience(experience)
            experiences.append(experience)

        return evaluation

    def _backpropagate(self, node: IdeaNode, evaluation: IdeaEvaluation) -> None:
        score = evaluation.composite
        current: Optional[IdeaNode] = node
        while current:
            current.visits += 1
            current.value_sum += score
            current = current.parent

    def _best_candidate(self, root: IdeaNode) -> Optional[SearchCandidate]:
        candidates: List[SearchCandidate] = []
        stack = [root]
        while stack:
            node = stack.pop()
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
        visited: List[SearchCandidate] = []
        while stack:
            node = stack.pop()
            if node.evaluation:
                visited.append(SearchCandidate(node=node, evaluation=node.evaluation))
            stack.extend(node.children)

        for label, scorer in by_metric.items():
            if visited:
                pareto[label] = max(visited, key=lambda c, s=scorer: s(c.evaluation))
        return pareto

    def _harvest_experience(self, node: IdeaNode, evaluation: IdeaEvaluation) -> Optional[Dict[str, Any]]:
        if evaluation.confidence > self.config.min_confidence_for_memory:
            experience = {
                    "defect": ", ".join(node.state.target_defects) or evaluation.defect_fix_summary,
                    "action": node.state.operator,
                    "lift": round(evaluation.lift_estimate, 2),
                    "idea": node.state.title,
                    "context": node.path_summary(),
                    "feedback": evaluation.feedback,
                    "tags": node.state.tags + ["defect_fix"],
                }
            
            return experience
        else:
            return None

    def _format_edit_ops(self) -> str:
        lines = []
        for op in EDIT_OPERATORS:
            lines.append(
                f"- {op.name}: {op.description} | targets {', '.join(op.defects)} | guardrails: {', '.join(op.guardrails)}"
            )
        return "\n".join(lines)
