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
from src.agents.idea_agent.agent.artifacts import artifact_init
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
    extract_context_signature,
)

logger = get_logger()

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
            self.artifact["mature_idea"] = initial_mature_idea.strip()

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
            self.artifact["idea_taste_mode"] = idea_taste_preset.mode
            self.artifact["idea_taste_label"] = idea_taste_preset.label
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
                    effort = "high" if stage == "mcts_expand" else "low"
                    return super().chat(
                        prompt,
                        model=resolved_model,
                        reasoning={"effort": effort},
                        **request_kwargs,
                    )
                elif "gpt-5" in resolved_model:
                    # Idea Evaluator: GPT-5.4
                    request_kwargs["temperature"] = 1.0
                    effort = "high" if stage == "idea_fusion" else "low"
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
        if not self.artifact.get("run_topic"):
            self.artifact["run_topic"] = normalized_topic
        self.artifact["topic"].append(normalized_topic)
        self.artifact["retrieval_keywords"].append(keywords)
        background = generate_background_brief(
            normalized_topic,
            PROMPTS,
            self.chat,
            self.model,
            logger,
        )
        if background:
            self.artifact["background_knowledge"].append(background)

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
        before_steps = len(self.artifact.get("steps", []))
        self.workflow_executor.run(
            spec,
            make_stage_context(self, workflow_name=spec.name, **kwargs),
        )
        new_steps = self.artifact.get("steps", [])[before_steps:]
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
                "ka_route": StageSpec(name="ka_route", handler=self._ka_route_stage),
                "ka_seed_search": StageSpec(
                    name="ka_seed_search",
                    handler=self._ka_seed_search_stage,
                    fallback_stage="ka_seed_search_fallback",
                ),
                "ka_seed_search_fallback": StageSpec(
                    name="ka_seed_search_fallback",
                    handler=self._ka_seed_search_fallback_stage,
                ),
                "ka_query_generation": StageSpec(
                    name="ka_query_generation",
                    handler=self._ka_query_generation_stage,
                ),
                "ka_outcome_rag": StageSpec(
                    name="ka_outcome_rag",
                    handler=self._ka_outcome_rag_stage,
                ),
                "ka_citation_expansion": StageSpec(
                    name="ka_citation_expansion",
                    handler=self._ka_citation_expansion_stage,
                ),
                "ka_enrichment": StageSpec(
                    name="ka_enrichment",
                    handler=self._ka_enrichment_stage,
                ),
                "ka_paper_triage": StageSpec(
                    name="ka_paper_triage",
                    handler=self._ka_paper_triage_stage,
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
            self.artifact.get("references", []),
        )
        topic = self.artifact["topic"][-1] if self.artifact["topic"] else "unspecified topic"
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
        into :class:`SymbolicRecord` instances that ``compute_action_priors``
        can consume during MCTS expand.

        Two signal sources may exist in the artifact, but only
        **experiment-agent ablation results** are injected by the default
        implementation:

        1. **Experiment-agent ablation results** – stored under
           ``self.artifact["ablation_results"]``.  Each entry is a dict::

               {
                   "component": "uncertainty_weighted_loss",
                   "op": "remove",          # add | remove | replace | ...
                   "delta_score": -0.12,     # observed Δ metric
                   "method_context": "...",  # optional, for family extraction
                   "context_signature": {},   # optional ContextSignature dict
                   "confidence": 0.8,
               }

        2. **Paper-graph conclusions** – stored under
           ``self.artifact["paper_graph_priors"]``.  They are not injected
           into symbolic memory by the default implementation.

        Override or extend this method to plug in additional signal sources.
        """
        if not self.mcts.config.enable_symbolic_memory:
            return
        sym = self.mcts.symbolic_memory
        injected = 0

        entries = self.artifact.get("ablation_results", [])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            component = str(entry.get("component", "")).strip()
            main_op = str(entry.get("op", "")).strip().lower()
            delta = entry.get("delta_score")
            if not component or not main_op or delta is None:
                continue

            method_text = str(entry.get("method_context", ""))
            families = extract_component_families([component], method_text)
            component_family = families[0].get("family", "") if families else ""

            ctx_sig_raw = entry.get("context_signature")
            ctx_sig_dict = dict(ctx_sig_raw) if isinstance(ctx_sig_raw, dict) else {}

            confidence = max(0.0, min(1.0, float(entry.get("confidence", 0.5))))

            try:
                record = sym.instantiate_symbolic_record(
                    summary=f"{main_op} on {component} -> delta={float(delta):+.4f}",
                    pattern=f"component={component}; op={main_op}",
                    conditions=[],
                    actions=[f"{main_op}:{component}"],
                    rationale=str(entry.get("rationale", "")),
                    expected_outcomes=[],
                    anti_patterns=[],
                    tags=["experiment_ablation", main_op, component],
                    priority=max(0.0, min(1.0, abs(float(delta)))),
                    confidence=confidence,
                    source="experiment_ablation",
                    support_count=int(entry.get("support_count", 1)),
                    metadata={"topic": topic, "raw_entry": entry},
                    component_family=component_family,
                    main_op=main_op,
                    context_signature=ctx_sig_dict,
                    delta_score=float(delta),
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


    def ingest_ablation_results(self, results: List[Dict[str, Any]]) -> None:
        """Write experiment-agent ablation results into the artifact so that
        ``_inject_symbolic_priors`` can consume them on the next
        ``idea_generation`` call.

        Expected schema for each entry in ``results``::

            {
                "component":        str,   # e.g. "uncertainty_weighted_loss"
                "op":               str,   # add | remove | replace | ...
                "delta_score":      float, # observed Δ metric
                "method_context":   str,   # optional free-text for family extraction
                "context_signature": dict, # optional ContextSignature fields
                "confidence":       float, # [0, 1]
                "rationale":        str,   # optional
                "support_count":    int,   # optional, default 1
            }
        """
        if not results:
            return
        valid: List[Dict[str, Any]] = []
        for entry in results:
            if not isinstance(entry, dict):
                logger.warning("[LigAgent] ingest_ablation_results: skipping non-dict entry %r", entry)
                continue
            if not entry.get("component") or not entry.get("op") or entry.get("delta_score") is None:
                logger.warning(
                    "[LigAgent] ingest_ablation_results: skipping entry missing required fields "
                    "(component/op/delta_score): %r",
                    entry,
                )
                continue
            valid.append(dict(entry))
        existing: List[Dict[str, Any]] = self.artifact.get("ablation_results", [])
        self.artifact["ablation_results"] = existing + valid
        logger.info(
            "[LigAgent] ingest_ablation_results: added %d record(s), total=%d",
            len(valid),
            len(self.artifact["ablation_results"]),
        )
