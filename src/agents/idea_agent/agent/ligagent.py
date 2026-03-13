from src.agents.idea_agent.agent.base import AgentBase
from src.agents.idea_agent.utils.core.chat_transport import ensure_default_max_output_tokens
from src.agents.idea_agent.agent import get_logger
from src.agents.idea_agent.utils.core.logger import get_or_create_mode_logger

import logging
from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import time
from copy import deepcopy
from dataclasses import fields

from src.agents.idea_agent.agent.tools import TOOLS
from src.agents.idea_agent.agent.artifacts import (
    artifact_append,
    artifact_get,
    artifact_init,
    artifact_set,
)
from src.agents.idea_agent.agent.prompts import PROMPTS
from src.agents.idea_agent.agent.mcts import (
    MemoryGuidedMCTS,
    MCTSConfig,
    VectorMemoryAccessor,
    apply_idea_taste_preset,
)
from src.agents.idea_agent.utils.papers.paper_repository import PaperRepository
from src.agents.idea_agent.utils.workflow import ligagent_handlers
from src.agents.idea_agent.utils.workflow.ligagent_flow import (
    build_action_workflow,
    make_stage_context,
    persist_final_idea,
)
from src.agents.idea_agent.utils.workflow.ligagent_utils import (
    collect_paper_context_entries,
    generate_idea_introduction,
    LigRuntime,
    LigSession,
    parse_json_response,
)
from src.agents.idea_agent.utils.workflow.ligagent_helpers import (
    build_action_lookup,
    generate_background_brief,
    sanitize_action_token,
    get_paper_content as load_paper_content,
)
from src.agents.idea_agent.utils.workflow.stage_contract import (
    StageContext,
    StageResult,
)
from src.agents.idea_agent.utils.workflow.workflow_runtime import (
    StageSpec,
    WorkflowEdge,
    WorkflowExecutor,
    WorkflowSpec,
)
from src.agents.idea_agent.utils.core.config_loader import get_config_value
from memory.api.component_taxonomy import (
    ContextSignature,
    extract_component_families,
)

logger = get_logger()

_ABLATION_RESULT_SIGN = {
    "positive": 1.0,
    "negative": -1.0,
    "inconclusive": 0.0,
    "mixed": 0.0,
    "neutral": 0.0,
}


def _normalize_ablation_result_label(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "pos": "positive",
        "beneficial": "positive",
        "good": "positive",
        "works": "positive",
        "neg": "negative",
        "harmful": "negative",
        "bad": "negative",
        "fails": "negative",
        "failure": "negative",
        "unclear": "inconclusive",
        "unknown": "inconclusive",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in _ABLATION_RESULT_SIGN else "inconclusive"


def _normalize_ablation_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _fallback_component_family(component: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(component or "").strip())
    normalized = "_".join(part for part in normalized.split("_") if part)
    return f"component.{normalized or 'unknown'}"


def _normalize_ablation_component_entry(
    component_name: str,
    payload: Dict[str, Any],
    *,
    run_summary: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    component = str(component_name or payload.get("component") or "").strip()
    if not component:
        return None

    result = _normalize_ablation_result_label(payload.get("result"))
    confidence = _normalize_ablation_confidence(payload.get("confidence"), default=0.5)
    normalized = {
        "component": component,
        "op": str(payload.get("op") or "remove").strip().lower() or "remove",
        "result": result,
        "metric": str(payload.get("metric") or "").strip(),
        "value": str(payload.get("value") or "").strip(),
        "analysis": str(payload.get("analysis") or payload.get("rationale") or "").strip(),
        "method_context": str(payload.get("method_context") or "").strip(),
        "confidence": confidence,
        "run_summary": dict(run_summary or {}),
        "support_count": max(1, int(payload.get("support_count", 1) or 1)),
    }
    return normalized


def normalize_ablation_results_payload(results: Any) -> List[Dict[str, Any]]:
    if not results:
        return []

    normalized: List[Dict[str, Any]] = []
    if isinstance(results, list):
        for entry in results:
            if not isinstance(entry, dict):
                continue
            item = _normalize_ablation_component_entry(
                str(entry.get("component") or ""),
                entry,
                run_summary=entry.get("run_summary") if isinstance(entry.get("run_summary"), dict) else None,
            )
            if item is not None:
                normalized.append(item)
        return normalized

    if not isinstance(results, dict):
        return []

    run_summary = results.get("summary") if isinstance(results.get("summary"), dict) else {}
    components = results.get("components") if isinstance(results.get("components"), dict) else {}
    for component_name, payload in components.items():
        if not isinstance(payload, dict):
            continue
        item = _normalize_ablation_component_entry(
            str(component_name),
            payload,
            run_summary=run_summary,
        )
        if item is not None:
            normalized.append(item)
    return normalized

class LigAgent(AgentBase):
    ACTION_ALIASES: Dict[str, Set[str]] = {
        "knowledge_aquisition": {
            "knowledge_aquisition",
            "knowledge aquisition",
            "knowledge acquisition",
            "knowledgeaquisition",
            "knowledgeacquisition",
            "knowledge-acquisition",
        },
        "advanced_analysis": {
            "advanced_analysis",
            "advanced analysis",
            "advancedanalysis",
            "advanced-analysis",
        },
        "idea_generation": {
            "idea_generation",
            "idea generation",
            "ideageneration",
            "idea-generation",
        },
        "re_analysis_replan": {
            "re_analysis_replan",
            "re analysis replan",
            "reanalysisreplan",
            "re-analysis-replan",
        },
    }

    def _canonical_action(self, action: Optional[str]) -> Optional[str]:
        if not action:
            return None
        sanitized = sanitize_action_token(action)
        return self._action_lookup.get(sanitized)

    def __init__(self, *args, **kwargs):
        # Configure from provided config or load from file
        config = kwargs.pop("config", None)
        self.config = config

        chat_max_retries = get_config_value(config, "agent.chat_max_retries", 3)
        chat_retry_backoff = get_config_value(config, "agent.chat_retry_backoff", 2.0)
        survey_config_path = kwargs.pop("survey_config_path", None)
        run_dir = kwargs.pop("run_dir", None)
        rag_config = kwargs.pop("rag_config", None)
        model = get_config_value(config, "agent.model", "gpt-5-mini")
        super().__init__(*args, **kwargs)
        self.model = str(model or "gpt-5-mini")
        self.action_space = [
            "knowledge_aquisition",
            "advanced_analysis",
            "idea_generation",
            "re_analysis_replan",
        ]
        self.tools = TOOLS
        self.artifact = artifact_init()
        self.session = LigSession(self.artifact)

        # Result persistence paths
        self.run_dir = Path(run_dir) if run_dir else Path(__file__).resolve().parent.parent
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.idea_result_path = self.run_dir / "idea_result.json"
        self.idea_result_path.parent.mkdir(parents=True, exist_ok=True)
        self._mode_loggers: Dict[str, logging.Logger] = {}

        self.chat_max_retries = chat_max_retries
        self.chat_retry_backoff = chat_retry_backoff
        self._action_lookup = build_action_lookup(self.ACTION_ALIASES)
        self.semantic_search_limit = get_config_value(config, "agent.semantic_search_limit", 5)
        self.logger = logger
        self.runtime = LigRuntime(self)
        self.workflow_executor = WorkflowExecutor(logger=logger)

        # Initialize mature_idea in artifact from config (user-provided) if available
        initial_mature_idea = get_config_value(config, "run.mature_idea", "")
        if isinstance(initial_mature_idea, str) and initial_mature_idea.strip():
            artifact_set(self.artifact, "mature_idea", initial_mature_idea.strip())

        mcts_config = MCTSConfig()
        for field in fields(MCTSConfig):
            override = get_config_value(config, f"mcts.{field.name}", None)
            if override is not None:
                setattr(mcts_config, field.name, override)
        legacy_symbolic_memory_path = get_config_value(
            config, "mcts.skill_prior_memory_path", None
        )
        if (
            legacy_symbolic_memory_path is not None
            and get_config_value(config, "mcts.symbolic_memory_path", None) is None
        ):
            setattr(mcts_config, "symbolic_memory_path", legacy_symbolic_memory_path)
        if bool(get_config_value(config, "run.LigAgent-Pro", False)):
            setattr(mcts_config, "enable_vector_memory", False)
        idea_taste_preset = apply_idea_taste_preset(mcts_config)
        if idea_taste_preset:
            artifact_set(self.artifact, "idea_taste_mode", idea_taste_preset.mode)
            artifact_set(self.artifact, "idea_taste_label", idea_taste_preset.label)
            logger.info(
                "[LigAgent] Applied idea taste preset %s (%s).",
                idea_taste_preset.mode,
                idea_taste_preset.label,
            )
        if bool(get_config_value(config, "run.LigAgent-Pro", False)):
            logger.info("[LigAgent] LigAgent-Pro enabled: vector memory disabled for all MCTS instances.")
        self.mcts = MemoryGuidedMCTS(
            chat_fn=self.chat,
            evaluation_prompt=PROMPTS.get("mcts_evaluation"),
            config=mcts_config,
            memory_accessor=self._build_memory_accessor(),
            logger=logger,
        )

        self.paper_enrichment_timeout = kwargs.pop(
            "paper_enrichment_timeout",
            get_config_value(config, "agent.paper_enrichment_timeout_sec", 7200),
        )
        self.paper_repository = PaperRepository(
            config_path=survey_config_path,
            logger=logger,
            rag_config=rag_config,
        )

    def chat(self, prompt: str, model: str = "gpt-5-mini", **kwargs) -> str:
        last_exc: Optional[Exception] = None
        request_kwargs = dict(kwargs)
        stage = str(request_kwargs.pop("stage", "") or "").strip()
        resolved_model = str(model or self.model or "gpt-5-mini")
        for attempt in range(1, self.chat_max_retries + 1):
            try:
                # Special handling for GPT-5 models
                if "gpt-5-mini" in resolved_model:
                    # Idea Generator: GPT-5 mini
                    request_kwargs["temperature"] = 1.0
                    effort = "high" if stage in {"mcts_expand", "idea_fusion", "advanced_analysis", "re_analysis_replan", "experiment_findings_extraction"} else "low"
                    return super().chat(
                        prompt,
                        model=resolved_model,
                        reasoning={"effort": effort},
                        **request_kwargs,
                    )
                elif "gpt-5" in resolved_model:
                    # Idea Evaluator: GPT-5.2
                    request_kwargs["temperature"] = 1.0
                    effort = "high" if stage in {"mcts_expand", "idea_fusion", "advanced_analysis", "re_analysis_replan", "experiment_findings_extraction"} else "low"
                    return super().chat(
                        prompt,
                        model=resolved_model,
                        reasoning={"effort": effort},
                        **request_kwargs,
                    )
                elif "kimi-k2.5" in resolved_model:
                    if stage != "mcts_expand":
                        request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
                    return super().chat(prompt, model=resolved_model, **request_kwargs)
            except Exception as exc:
                last_exc = exc
                wait = self.chat_retry_backoff ** (attempt - 1)
                logger.warning(
                    "⚠️ Chat attempt %d/%d failed (%s). Retrying in %.2fs...",
                    attempt,
                    self.chat_max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise last_exc if last_exc else RuntimeError("Chat failed without exception detail.")

    def _workflow_model_group(
        self,
        stage: Optional[str],
        workflow_name: Optional[str],
    ) -> str:
        workflow = str(workflow_name or "").strip()
        stage_name = str(stage or "").strip()
        if workflow.endswith("knowledge_acquisition") or stage_name.startswith("ka_"):
            return "knowledge_acquisition"
        return "main"

    def resolve_stage_model(
        self,
        *,
        stage: Optional[str],
        workflow_name: Optional[str] = None,
        requested_model: Optional[str] = None,
    ) -> str:
        base_model = str(self.model or "gpt-5-mini")
        explicit_model = str(requested_model or "").strip()
        if explicit_model and explicit_model != base_model:
            return explicit_model

        group = self._workflow_model_group(stage=stage, workflow_name=workflow_name)
        lookup_keys: List[str] = []
        if stage:
            lookup_keys.append(f"agent.workflow_models.{group}.{stage}")
            lookup_keys.append(f"agent.workflow_models.stage_overrides.{stage}")
        lookup_keys.append("agent.workflow_models.default")

        for key in lookup_keys:
            value = get_config_value(self.config, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return explicit_model or base_model

    def perform_action(self, action: str, **kwargs) -> Any:
        logger.info(f"🚀 Performing action: {action}...")
        resolved_action = self._canonical_action(action)
        if not resolved_action:
            raise ValueError(f"Unknown action: {action}")
        if resolved_action == "knowledge_aquisition":
            logger.info("🔍 Due to API and web request rate limits, this process may take some time...")
        return self._run_action_workflow(resolved_action, **kwargs)

    def bootstrap_topic(self, topic: str, retrieval_keywords: Optional[str] = None) -> None:
        normalized_topic = (topic or "").strip()
        if not normalized_topic:
            raise ValueError("Topic must be a non-empty string.")
        keywords = (
            retrieval_keywords.strip()
            if isinstance(retrieval_keywords, str)
            else normalized_topic
        )
        if not artifact_get(self.artifact, "run_topic", ""):
            artifact_set(self.artifact, "run_topic", normalized_topic)
        artifact_append(self.artifact, "topic", [normalized_topic])
        artifact_append(self.artifact, "retrieval_keywords", [keywords])
        background = generate_background_brief(
            normalized_topic,
            PROMPTS,
            self.chat,
            self.model,
            logger,
        )
        if background:
            artifact_append(self.artifact, "background_knowledge", [background])

    def knowledge_aquisition(self) -> str:
        return self._run_action_workflow("knowledge_aquisition")

    def get_paper_content(self, paper_id: str, include_markdown: bool = True) -> Dict[str, Any]:
        return load_paper_content(
            paper_id,
            include_markdown,
            self.artifact,
            self.paper_repository,
            logger,
        )

    def advanced_analysis(self, **kwargs) -> str:
        return self._run_action_workflow("advanced_analysis", **kwargs)

    def idea_generation(self, **kwargs) -> str:
        return self._run_action_workflow("idea_generation", **kwargs)

    def re_analysis_replan(self, **kwargs) -> str:
        return self._run_action_workflow("re_analysis_replan", **kwargs)

    def build_mcts_for_mode(self, idea_taste_mode: Optional[str]) -> MemoryGuidedMCTS:
        mcts_config = deepcopy(self.mcts.config)
        setattr(mcts_config, "idea_taste_mode", idea_taste_mode)
        apply_idea_taste_preset(mcts_config)
        mode_mcts = MemoryGuidedMCTS(
            chat_fn=self.chat,
            evaluation_prompt=PROMPTS.get("mcts_evaluation"),
            config=mcts_config,
            memory_accessor=self._build_memory_accessor(),
            logger=get_or_create_mode_logger(
                self._mode_loggers,
                self.logger,
                self.run_dir,
                idea_taste_mode,
            ),
        )
        mode_mcts.symbolic_memory = deepcopy(self.mcts.symbolic_memory)
        mode_mcts.persist_symbolic_memory = False
        return mode_mcts

    def _build_memory_accessor(self) -> VectorMemoryAccessor:
        model_path = str(
            get_config_value(
                self.config,
                "memory.vector_store.model_path",
                ".cache/all-MiniLM-L6-v2",
            )
            or ".cache/all-MiniLM-L6-v2"
        )
        return VectorMemoryAccessor(
            semantic_cfg={"model_path": model_path},
            episodic_cfg={"model_path": model_path},
            procedural_cfg={"model_path": model_path},
            llm_name=str(
                get_config_value(
                    self.config,
                    "memory.vector_store.llm_name",
                    "gpt-5-mini",
                )
                or "gpt-5-mini"
            ),
            llm_backend=str(
                get_config_value(
                    self.config,
                    "memory.vector_store.llm_backend",
                    "openai",
                )
                or "openai"
            ),
            logger=logger,
        )

    def _persist_final_idea(
        self, best_entry: Dict[str, Any], paper_entries: List[Dict[str, Any]]
    ) -> None:
        persist_final_idea(
            best_entry=best_entry,
            paper_entries=paper_entries,
            artifact=self.artifact,
            idea_result_path=self.idea_result_path,
            chat_fn=self.chat,
            model=self.model,
            logger=logger,
            prompts=PROMPTS,
        )

    def _run_action_workflow(self, action: str, **kwargs) -> str:
        spec = build_action_workflow(self, action)
        before_steps = len(artifact_get(self.artifact, "steps", []))
        self.workflow_executor.run(
            spec,
            make_stage_context(self, workflow_name=spec.name, **kwargs),
        )
        new_steps = artifact_get(self.artifact, "steps", [])[before_steps:]
        if new_steps:
            logger.info(new_steps[-1])
            return new_steps[-1]
        return ""

    def _execute_knowledge_acquisition_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.execute_knowledge_acquisition_stage(self, ctx)

    def _execute_advanced_analysis_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.execute_advanced_analysis_stage(self, ctx)

    def _execute_idea_generation_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.execute_idea_generation_stage(self, ctx)

    def _execute_reanalysis_replan_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.execute_reanalysis_replan_stage(self, ctx)

    def _build_knowledge_acquisition_workflow(self) -> WorkflowSpec:
        return WorkflowSpec(
            name="ligagent.knowledge_acquisition",
            entry_stage="ka_route",
            stages={
                "ka_route": StageSpec(
                    name="ka_route",
                    handler=self._ka_route_stage,
                ),
                "ka_seed_search": StageSpec(
                    name="ka_seed_search",
                    handler=self._ka_seed_search_stage,
                    fallback_stage="ka_seed_search_fallback",
                    allowed_artifact_namespaces={"retrieval"},
                ),
                "ka_seed_search_fallback": StageSpec(
                    name="ka_seed_search_fallback",
                    handler=self._ka_seed_search_fallback_stage,
                    allowed_artifact_namespaces={"retrieval"},
                ),
                "ka_query_generation": StageSpec(
                    name="ka_query_generation",
                    handler=self._ka_query_generation_stage,
                    allowed_artifact_namespaces={"retrieval"},
                ),
                "ka_outcome_rag": StageSpec(
                    name="ka_outcome_rag",
                    handler=self._ka_outcome_rag_stage,
                    allowed_artifact_namespaces={"retrieval"},
                ),
                "ka_citation_expansion": StageSpec(
                    name="ka_citation_expansion",
                    handler=self._ka_citation_expansion_stage,
                    allowed_artifact_namespaces={"retrieval"},
                ),
                "ka_enrichment": StageSpec(
                    name="ka_enrichment",
                    handler=self._ka_enrichment_stage,
                    allowed_artifact_namespaces={"retrieval"},
                ),
                "ka_paper_triage": StageSpec(
                    name="ka_paper_triage",
                    handler=self._ka_paper_triage_stage,
                    allowed_artifact_namespaces={"retrieval"},
                ),
            },
            transitions={
                "ka_query_generation": [WorkflowEdge("ka_outcome_rag")],
                "ka_outcome_rag": [WorkflowEdge("ka_citation_expansion")],
                "ka_citation_expansion": [
                    WorkflowEdge("ka_enrichment", when=lambda stage_ctx, _result: bool(stage_ctx.state.get("combined_papers"))),
                ],
                "ka_enrichment": [WorkflowEdge("ka_paper_triage")],
            },
        )

    def _ka_route_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_route_stage(self, ctx)

    def _ka_seed_search_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_seed_search_stage(self, ctx)

    def _ka_seed_search_fallback_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_seed_search_fallback_stage(self, ctx)

    def _ka_query_generation_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_query_generation_stage(self, ctx)

    def _ka_outcome_rag_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_outcome_rag_stage(self, ctx)

    def _ka_citation_expansion_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_citation_expansion_stage(self, ctx)

    def _ka_enrichment_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_enrichment_stage(self, ctx)

    def _ka_paper_triage_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_paper_triage_stage(self, ctx)

    def _generate_idea_introduction(
        self, best_entry: Dict[str, Any], paper_entries: List[Dict[str, Any]]
    ) -> str:
        entries = paper_entries or collect_paper_context_entries(
            self.artifact,
            artifact_get(self.artifact, "references", []),
        )
        topic_history = artifact_get(self.artifact, "topic", [])
        topic = topic_history[-1] if topic_history else "unspecified topic"
        return generate_idea_introduction(
            chat_fn=self.chat,
            prompt_template=PROMPTS["idea_introduction"],
            model=self.model,
            topic=topic,
            best_entry=best_entry,
            paper_entries=entries,
            logger=logger,
        )

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        return parse_json_response(raw)

    def _inject_symbolic_priors(
        self,
        topic: str,
        context: Dict[str, Any],
    ) -> None:
        """Inject component×op priors into ``self.mcts.symbolic_memory``.

        This is the single entry-point where external signals are translated
        into :class:`SymbolicRecord` instances that downstream
        symbolic-memory-aware stages can consume.

        Two signal sources may exist in the artifact, but only
        **experiment-agent ablation results** are injected by the default
        implementation:

        1. **Experiment-agent ablation results** – stored under
           ``self.artifact["ideation"]["ablation_results_raw"]`` as the raw
           ablation report, and normalized into
           ``self.artifact["ideation"]["ablation_results"]``. Each normalized
           entry is a dict::

               {
                   "component": "uncertainty_weighted_loss",
                   "op": "remove",
                   "result": "negative",
                   "metric": "...",
                   "value": "...",
                   "analysis": "...",
                   "method_context": "...",  # idea intro with this component removed
                   "confidence": 0.8,
                   "run_summary": {},
               }

        2. **Paper-graph conclusions** – stored under
           ``self.artifact["ideation"]["paper_graph_priors"]``.  They are not injected
           into symbolic memory by the default implementation.

        Override or extend this method to plug in additional signal sources.
        """
        if not self.mcts.config.enable_symbolic_memory:
            return
        sym = self.mcts.symbolic_memory
        injected = 0

        raw_payload = artifact_get(self.artifact, "ablation_results_raw", {})
        entries = normalize_ablation_results_payload(raw_payload)
        if not entries:
            entries = normalize_ablation_results_payload(
                artifact_get(self.artifact, "ablation_results", [])
            )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            component = str(entry.get("component", "")).strip()
            main_op = str(entry.get("op", "")).strip().lower() or "remove"
            if not component or not main_op:
                continue

            method_text = str(entry.get("method_context", ""))
            families = extract_component_families([component], method_text)
            component_family = (
                str(families[0].get("family") or "").strip()
                if families else ""
            ) or _fallback_component_family(component)

            confidence = _normalize_ablation_confidence(entry.get("confidence"), default=0.5)
            result_label = _normalize_ablation_result_label(entry.get("result"))
            metric_name = str(entry.get("metric", "")).strip()
            metric_value = str(entry.get("value", "")).strip()
            analysis_text = str(entry.get("analysis", "")).strip()
            run_summary = dict(entry.get("run_summary") or {}) if isinstance(entry.get("run_summary"), dict) else {}

            try:
                record = sym.instantiate_symbolic_record(
                    component=component,
                    component_family=component_family,
                    op=main_op,
                    result=result_label,
                    metric=metric_name,
                    value=metric_value,
                    analysis=analysis_text,
                    method_context=method_text,
                    confidence=confidence,
                    run_summary=run_summary,
                    metadata={
                        "topic": topic,
                        "raw_entry": entry,
                    },
                    support_count=int(entry.get("support_count", 1)),
                )
                sym.upsert_normal_records([record], agent_id="idea_agent")
                injected += 1
            except Exception as exc:
                logger.warning(
                    "⚠️  Failed to inject symbolic prior for %s/%s: %s",
                    component, main_op, exc,
                )

        if injected:
            logger.info(
                "[LigAgent] Injected %d symbolic prior(s) into MCTS symbolic memory.",
                injected,
            )


    def ingest_ablation_results(self, results: Any) -> None:
        """Write experiment-agent ablation results into the artifact so that
        ``_inject_symbolic_priors`` can consume them on the next
        ``idea_generation`` call.

        Preferred input schema is the raw ablation report::

            {
              "components": {
                "<component_name>": {
                  "result": "positive|negative|inconclusive",
                  "metric": "...",
                  "value": "...",
                  "confidence": 0.95,
                  "analysis": "...",
                  "method_context": "..."
                }
              },
              "summary": {...}
            }

        The normalized internal representation for each component is::

            {
                "component":        str,   # e.g. "uncertainty_weighted_loss"
                "op":               str,   # usually "remove"
                "result":           str,
                "metric":           str,
                "value":            str,
                "analysis":         str,
                "method_context":   str,   # idea intro with this component removed
                "confidence":       float, # [0, 1]
                "run_summary":      dict,  # top-level summary payload
                "support_count":    int,   # optional, default 1
            }
        """
        if not results:
            return
        if isinstance(results, dict):
            artifact_set(self.artifact, "ablation_results_raw", dict(results))

        normalized_results = normalize_ablation_results_payload(results)
        valid: List[Dict[str, Any]] = []
        for entry in normalized_results:
            if not isinstance(entry, dict):
                logger.warning("[LigAgent] ingest_ablation_results: skipping non-dict entry %r", entry)
                continue
            if not entry.get("component") or not entry.get("op"):
                logger.warning(
                    "[LigAgent] ingest_ablation_results: skipping entry missing required fields "
                    "(component/op): %r",
                    entry,
                )
                continue
            valid.append(dict(entry))
        if not isinstance(results, dict):
            artifact_set(self.artifact, "ablation_results_raw", {})
        artifact_append(self.artifact, "ablation_results", valid)
        logger.info(
            "[LigAgent] ingest_ablation_results: added %d record(s), total=%d",
            len(valid),
            len(artifact_get(self.artifact, "ablation_results", [])),
        )
