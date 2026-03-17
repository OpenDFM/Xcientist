from src.agents.idea_agent.agent.base import AgentBase
from src.agents.idea_agent.agent import get_logger
from src.agents.idea_agent.utils.core.logger import get_or_create_mode_logger
from src.agents.idea_agent.utils.core.chat_router import prepare_ligagent_chat_request

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import time
from copy import deepcopy
from dataclasses import fields

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
    generate_background_brief,
    normalize_ablation_results_payload,
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
from src.agents.idea_agent.utils.papers.paper_graph_vector_store import (
    PaperGraphComponentVectorStore,
)

logger = get_logger()

class LigAgent(AgentBase):
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
        self._shared_paper_graph_vector_store = self._build_shared_paper_graph_vector_store(
            mcts_config
        )
        self.mcts = MemoryGuidedMCTS(
            chat_fn=self.chat,
            evaluation_prompt=PROMPTS.get("mcts_evaluation"),
            config=mcts_config,
            memory_accessor=self._build_memory_accessor(),
            paper_graph_vector_store=self._shared_paper_graph_vector_store,
            logger=logger,
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
                routed_model, routed_kwargs = prepare_ligagent_chat_request(
                    model=resolved_model,
                    stage=stage,
                    kwargs=request_kwargs,
                )
                return super().chat(prompt, model=routed_model, **routed_kwargs)
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
            paper_graph_vector_store=self._shared_paper_graph_vector_store,
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

    def _build_shared_paper_graph_vector_store(
        self,
        mcts_config: MCTSConfig,
    ) -> PaperGraphComponentVectorStore:
        shared_store = PaperGraphComponentVectorStore(
            model_name_or_path=mcts_config.component_novelty_model,
            index_dir=mcts_config.component_novelty_index_dir or None,
        )
        try:
            shared_store.warmup(allow_stale_graph=True)
            logger.info(
                "[LigAgent] Eager-loaded component retrieval resources from model=%s index_dir=%s",
                mcts_config.component_novelty_model,
                shared_store.index_dir,
            )
            return shared_store
        except Exception as exc:
            logger.warning(
                "⚠️  Failed to eager-load component retrieval resources (%s). "
                "Component novelty and theory-transfer retrieval will stay in fallback mode.",
                exc,
            )
            return PaperGraphComponentVectorStore(
                model_name_or_path=mcts_config.component_novelty_model,
                index_dir=mcts_config.component_novelty_index_dir or None,
                disabled_error=str(exc),
            )

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
                "ka_graph_retrieval": StageSpec(
                    name="ka_graph_retrieval",
                    handler=self._ka_graph_retrieval_stage,
                    allowed_artifact_namespaces={"retrieval"},
                ),
                "ka_reference_selection": StageSpec(
                    name="ka_reference_selection",
                    handler=self._ka_reference_selection_stage,
                    allowed_artifact_namespaces={"retrieval"},
                ),
            },
            transitions={
                "ka_query_generation": [WorkflowEdge("ka_outcome_rag")],
                "ka_outcome_rag": [WorkflowEdge("ka_graph_retrieval")],
                "ka_graph_retrieval": [WorkflowEdge("ka_reference_selection")],
            },
        )

    def _ka_route_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_route_stage(self, ctx)

    def _ka_query_generation_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_query_generation_stage(self, ctx)

    def _ka_outcome_rag_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_outcome_rag_stage(self, ctx)

    def _ka_graph_retrieval_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_graph_retrieval_stage(self, ctx)

    def _ka_reference_selection_stage(self, ctx: StageContext) -> StageResult:
        return ligagent_handlers.ka_reference_selection_stage(self, ctx)

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

    def ingest_ablation_results(self, results: Any) -> None:
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
            if not entry.get("component"):
                logger.warning(
                    "[LigAgent] ingest_ablation_results: skipping entry missing required field "
                    "(component): %r",
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
