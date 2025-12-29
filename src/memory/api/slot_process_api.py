import json
import asyncio
import concurrent.futures
import threading

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Any, Callable
from collections import deque
from memory.memory_system.utils import (
    dump_slot_json, 
    _extract_json_between, 
    _hard_validate_slot_keys,
    _build_context_snapshot,
    _safe_dump_str,
    _truncate_text,
    compute_overlap_score,
    _chunks,
    _multi_thread_run,
    new_id,
    now_iso,
)
from memory.memory_system.user_prompt import (
    WORKING_SLOT_COMPRESS_USER_PROMPT,
    TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT,
    TRANSFER_IDEA_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT,
    TRANSFER_SLOT_TO_TEXT_PROMPT,
    TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_EXPEIRMENT,
    TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_EXPRIMENT,
    TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_EXPERIMENT,
    TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_IDEA,
    TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_IDEA,
    TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_IDEA,
)
from textwrap import dedent
from memory.memory_system import WorkingSlot, OpenAIClient, LLMClient
from memory.memory_system.models import (
    EpisodicRecord,
    SemanticRecord,
    ProceduralRecord,
)

EXPERIMENT_AGENT_STAGE_OPTIONS = [
    "pre_analysis",
    "code_plan",
    "code_implement",
    "code_judge",
    "experiment_execute",
    "experiment_analysis",
]

IDEA_AGENT_STAGE_OPTIONS = [
    "topic_alignment",
    "memory_retrieval",
    "analysis",
    "mcts_planning",
    "mcts_expansion",
    "mcts_simulation",
    "mcts_evaluation",
    "idea_selection",
    "memory_writeback",
]

class SlotProcess:
    def __init__(self, llm_name: str = "gpt-4o-mini", llm_backend: Literal["openai", "vllm"] = "openai"):
        self.slot_container: Dict[str, WorkingSlot] = {}
        self.filtered_slot_container: List[WorkingSlot] = []
        self.routed_slot_container: List[Dict] = []
        self.llm_model = OpenAIClient(model=llm_name, backend=llm_backend)
        self.memory_dict = []

    def add_slot(self, slot: WorkingSlot) -> None:
        self.slot_container[slot.to_dict().get('id')] = slot
    
    def clear_container(self) -> None:
        self.slot_container = {}

    def get_container_size(self) -> int:
        return len(self.slot_container)

    def query(self, query_text: str, slots: Optional[List[WorkingSlot]] = None, limit: int = 5, key_words: Optional[List[str]] = None) -> List[Tuple[float, WorkingSlot]]:
        if slots is None:
            slots = list(self.slot_container.values())
    
        k = min(limit, len(self.slot_container))

        scored_slots: List[Tuple[float, WorkingSlot]] = []
        for slot in self.slot_container.values():
            score = compute_overlap_score(query_text, slot.summary, key_words)
            scored_slots.append((score, slot))
        scored_slots.sort(key=lambda x: x[0], reverse=True)
        
        return scored_slots[:k]
        
    async def filter_and_route_slots(self, task: Literal["experiment", "idea"] = "experiment") -> List[Dict[str, WorkingSlot]]:
        for slot in self.slot_container.values():
            check_result = await slot.slot_filter(self.llm_model, task=task)
            if check_result == True:
                self.filtered_slot_container.append(slot)
        
        try:
            for filtered_slot in self.filtered_slot_container:
                route_result = await filtered_slot.slot_router(self.llm_model, task=task)
                pair = {
                    "memory_type": route_result,
                    "slot": filtered_slot
                }
                self.routed_slot_container.append(pair)
        except Exception as e:
            print(f"Routing error: {e}")
        
        return self.routed_slot_container

    def _multi_thread_filter_and_route_slot(self, slot: WorkingSlot, task: Literal["experiment", "idea"] = "experiment"):
        check_result = asyncio.run(slot.slot_filter(self.llm_model, task=task))
        if check_result == True:
            try:
                route_result = asyncio.run(slot.slot_router(self.llm_model, task=task))
                pair = {
                        "memory_type": route_result,
                        "slot": slot
                    }
                self.routed_slot_container.append(pair)
            except Exception as e:
                print(f"Routing error: {e}")
        else:
            return
    
    async def compress_slots(self, sids: List[str] = None) -> WorkingSlot:
        slot_json_blobs = []
        if sids is None:
            for idx, slot in enumerate(self.slot_container.values()):
                slot_json_blobs.append(f"### Slot {idx}\n{dump_slot_json(slot)}")
        else:
            for idx, slot_id in enumerate(sids):
                slot_json_blobs.append(f"### Slot {idx}\n{dump_slot_json(self.slot_container[slot_id])}")
        slots_block = "\n\n".join(slot_json_blobs)

        system_prompt = (
            "You are an expert research assistant and memory compressor. "
            "Given multiple WorkingSlot JSON dumps, produce a single, compact summary "
            "that preserves non-redundant, reusable knowledge while discarding noise."
            "Be precise, consistent, and avoid hallucinations. Output only the requested JSON inside the tags."
        )
        user_prompt = WORKING_SLOT_COMPRESS_USER_PROMPT.format(slots_block=slots_block)

        response = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = _extract_json_between(response, "compressed-slot", "compressed-slot")
        print(f"Compressed slot payload: {payload}")
        try:
            _hard_validate_slot_keys(payload, allowed_keys={"stage", "topic", "summary", "attachments", "tags"})
        except Exception as e:
            raise ValueError(f"Compressed slot validation error: {e}")
        
        stage = str(payload.get("stage", ""))
        topic = str(payload.get("topic", ""))
        summary = str(payload.get("summary", ""))
        attachments = payload.get("attachments", {})
        tags = payload.get("tags", [])

        compressed_slot = WorkingSlot(
            stage=stage,
            topic=topic,
            summary=summary,
            attachments=attachments,
            tags=tags
        )

        return compressed_slot
    
    async def transfer_slot_to_text(self, slot: WorkingSlot) -> str:
        system_prompt = (
            "You are an expert assistant that converts WorkingSlot JSON data into a clear, concise text summary. "
            "Focus on key insights, important metrics, and actionable items. Output only the requested text inside the tags."
        )

        user_prompt = TRANSFER_SLOT_TO_TEXT_PROMPT.format(dump_slot_json=dump_slot_json(slot))

        text = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        return text

    def _retry_llm_to_slots(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict,
        schema_name: str,
        allowed_keys: set,
        max_slots: int,
        max_retries: int = 5,
        max_tokens: int = 8192,
        post_process_slot: Optional[Callable[[dict, str], dict]] = None,
        context: Optional[str] = None,
        is_async: bool = False,
    ) -> List[WorkingSlot]:
        """
        Generic retry wrapper for LLM -> WorkingSlot conversion.
        
        Args:
            system_prompt: System prompt for LLM.
            user_prompt: User prompt for LLM.
            json_schema: JSON schema for structured output.
            schema_name: Name of the schema.
            allowed_keys: Allowed keys in slot dict for validation.
            max_slots: Maximum number of slots to return.
            max_retries: Number of retry attempts.
            max_tokens: Max tokens for LLM response.
            post_process_slot: Optional callable(slot_dict, context) -> slot_dict for per-slot post-processing.
            context: Original context string (passed to post_process_slot if provided).
            is_async: If True, use asyncio.run; otherwise assume already in async context.
        
        Returns:
            List of valid WorkingSlot objects.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                if is_async:
                    response = asyncio.run(self.llm_model.complete(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        json_schema=json_schema,
                        schema_name=schema_name,
                        strict=False,
                        max_tokens=max_tokens
                    ))
                else:
                    # For async methods, we need to await directly
                    # This branch is used when called from sync context
                    response = asyncio.run(self.llm_model.complete(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        json_schema=json_schema,
                        schema_name=schema_name,
                        strict=False,
                        max_tokens=max_tokens
                    ))
            except Exception as e:
                last_error = e
                print(f"[Retry {attempt}/{max_retries}] LLM call error: {e}")
                continue

            try:
                data = json.loads(response)
                if not data:
                    raise ValueError("Empty JSON response")
            except Exception as e:
                last_error = e
                print(f"[Retry {attempt}/{max_retries}] JSON parsing error: {e}")
                continue

            slots_data = data.get("slots", [])
            if not isinstance(slots_data, list):
                last_error = ValueError("`slots` must be a list.")
                print(f"[Retry {attempt}/{max_retries}] Invalid schema: `slots` must be a list.")
                continue

            working_slots: List[WorkingSlot] = []

            for slot_dict in slots_data[:max_slots]:
                if not isinstance(slot_dict, dict):
                    continue

                try:
                    _hard_validate_slot_keys(slot_dict, allowed_keys=allowed_keys)

                    # Apply post-processing if provided
                    if post_process_slot is not None and context is not None:
                        slot_dict = post_process_slot(slot_dict, context)

                    stage = str(slot_dict.get("stage", "")).strip()
                    topic = str(slot_dict.get("topic", "")).strip()
                    summary = str(slot_dict.get("summary", "")).strip()
                    attachments = slot_dict.get("attachments") or {}
                    tags = slot_dict.get("tags") or []

                    slot = WorkingSlot(
                        stage=stage,
                        topic=topic,
                        summary=summary,
                        attachments=attachments,
                        tags=list(tags),
                    )
                    working_slots.append(slot)

                except Exception as e:
                    last_error = e
                    print(f"[Retry {attempt}/{max_retries}] Error creating WorkingSlot: {e}")
                    continue

            if len(working_slots) != 0:
                return working_slots

            print(f"[Retry {attempt}/{max_retries}] No valid WorkingSlot created; retrying...")

        print(f"Failed to create any WorkingSlot after {max_retries} retries. Last error: {last_error}")
        return []

    @staticmethod
    def _post_process_chat_slot(slot_dict: dict, context: str) -> dict:
        """Post-process chat slot to inject extracted session_id."""
        try:
            extracted_session_id = _extract_session_id_from_context(context)
        except Exception as e:
            print(f"Session ID extraction error: {e}")
            raise

        attachments = slot_dict.get("attachments") or {}
        if not isinstance(attachments, dict):
            attachments = {}
        
        session_ids = attachments.get("session_ids")
        if not isinstance(session_ids, dict):
            session_ids = {"items": []}
            attachments["session_ids"] = session_ids
        
        session_ids["items"] = [extracted_session_id]
        slot_dict["attachments"] = attachments
        
        return slot_dict

    def _retry_llm_to_record(
        self,
        system_prompt: str,
        user_prompt: str,
        record_tag: str,
        slot: WorkingSlot,
        post_process_payload: Callable[[dict, WorkingSlot], Dict[str, Any]],
        max_retries: int = 5,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Generic retry wrapper for LLM -> Record (semantic/episodic/procedural) conversion.
        
        Args:
            system_prompt: System prompt for LLM.
            user_prompt: User prompt for LLM.
            record_tag: The XML-like tag to extract JSON from (e.g., "semantic-record").
            slot: The source WorkingSlot (used as fallback for missing fields).
            post_process_payload: Callable(payload_dict, slot) -> final_record_dict.
            max_retries: Number of retry attempts.
            max_tokens: Max tokens for LLM response.
        
        Returns:
            The final record dict.
        
        Raises:
            ValueError: If all retries fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                response = asyncio.run(self.llm_model.complete(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens
                ))
            except Exception as e:
                last_error = e
                print(f"[Retry {attempt}/{max_retries}] LLM call error: {e}")
                continue

            try:
                # Clean up response
                response = response.strip()
                response = response.replace("<think>", "").replace("</think>", "")

                payload = json.loads(response)
                
                if not payload or not isinstance(payload, dict):
                    raise ValueError(f"Empty or invalid payload extracted from <{record_tag}>")

                # Apply post-processing to build final record
                record = post_process_payload(payload, slot)
                return record

            except Exception as e:
                last_error = e
                print(f"[Retry {attempt}/{max_retries}] Record extraction/processing error: {e}")
                continue

        raise ValueError(f"Failed to create record after {max_retries} retries. Last error: {last_error}")

    def transfer_experiment_agent_context_to_working_slots(self, context, state: str, max_slots: int = 50) -> List[WorkingSlot]:
        
        if state not in set(EXPERIMENT_AGENT_STAGE_OPTIONS):
            return []

        snapshot = _build_context_snapshot(context, state)

        system_prompt = (
            "You are an expert workflow archivist. "
            "Transform the provided Experiment Agent context into WorkingSlot JSON objects. "
            "Each slot must capture the stage, topic, summary (≤120 words), attachments, and tags. "
            "Summaries must follow a Situation→Action→Result narrative whenever possible. "
            "You MUST output at least one slot."
        )

        user_prompt = TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT.format(
            max_slots=max_slots,
            snapshot=snapshot,
        )

        schema = Schema(max_slots=max_slots)
        experiment_task_slot_schema = schema.EXPERIMENT_TASK_SLOT_SCHEMA
        allowed_keys = {"stage", "topic", "summary", "attachments", "tags"}

        working_slots = self._retry_llm_to_slots(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=experiment_task_slot_schema,
            schema_name="EXPERIMENT_TASK_SLOT_SCHEMA",
            allowed_keys=allowed_keys,
            max_slots=max_slots,
            max_retries=5,
            max_tokens=4096,
            post_process_slot=None,
            context=snapshot,
            is_async=False,
        )

        return working_slots

    def transfer_idea_agent_context_to_working_slots(self, context: Dict[str, Any], max_slots: int = 10) -> List[WorkingSlot]:
        snapshot = _safe_dump_str(context)

        system_prompt = (
            "You are the recorder for ResearchAgent's memory-guided MCTS idea workflow. "
            "Transform the Idea Agent context into WorkingSlot JSON objects that capture operator usage, "
            "retrieved memories, structured ideas, evaluator feedback, and write-back actions. "
            "Each slot must retain stage, topic, ≤130 word SAR summary, attachments, and tags. "
            "Always emit at least one slot summarizing the best or most novel outcome."
        )

        user_prompt = TRANSFER_IDEA_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT.format(
            max_slots=max_slots,
            snapshot=snapshot,
            stage_enums=", ".join(IDEA_AGENT_STAGE_OPTIONS),
        )

        schema = Schema(max_slots=max_slots)
        idea_task_slot_schema = schema.IDEA_TASK_SLOT_SCHEMA
        allowed_keys = {"stage", "topic", "summary", "attachments", "tags"}

        working_slots = self._retry_llm_to_slots(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=idea_task_slot_schema,
            schema_name="IDEA_TASK_SLOT_SCHEMA",
            allowed_keys=allowed_keys,
            max_slots=max_slots,
            max_retries=5,
            max_tokens=4096,
            post_process_slot=None,
            context=snapshot,
            is_async=False,
        )

        return working_slots

    async def generate_long_term_memory(self, routed_slots: List[Dict[str, WorkingSlot]]) -> List[Dict[str, Any]]:
        allowed_types = {"semantic", "episodic", "procedural"}
        inputs: List[Dict[str, Any]] = []

        for pair in routed_slots:
            memory_type = pair.get("memory_type")
            slot = pair.get("slot")

            if memory_type not in allowed_types or not isinstance(slot, WorkingSlot):
                continue

            try:
                if memory_type == "semantic":
                    input_dict = self.transfer_slot_to_semantic_record(slot)
                elif memory_type == "episodic":
                    input_dict = self.transfer_slot_to_episodic_record(slot)
                else:
                    input_dict = self.transfer_slot_to_procedural_record(slot)
            except Exception as exc:
                print(
                    f"[MEMORY] Failed to convert slot {getattr(slot, 'id', 'unknown')} "
                    f"({memory_type}): {exc}"
                )
                continue

            if inputs is not None:
                inputs.append({"memory_type": memory_type, "input": input_dict})

        return inputs

    def _multi_thread_transfer_slot_to_memory(self, pair: Dict[str, WorkingSlot]) -> List[Dict[str, Any]]:
        allowed_types = {"semantic", "episodic", "procedural"}
        memory_type = pair.get("memory_type")
        slot = pair.get("slot")

        if memory_type not in allowed_types or not isinstance(slot, WorkingSlot):
            return

        try:
            if memory_type == "semantic":
                input_dict = self.transfer_slot_to_semantic_record(slot)
            elif memory_type == "episodic":
                input_dict = self.transfer_slot_to_episodic_record(slot)
            elif memory_type == "procedural":
                input_dict = self.transfer_slot_to_procedural_record(slot)
        except Exception as exc:
            print(
                f"[MEMORY] Failed to convert slot {getattr(slot, 'id', 'unknown')} "
                f"({memory_type}): {exc}"
            )
            return

        self.memory_dict.append({"memory_type": memory_type, "input": input_dict})

    def transfer_slot_to_semantic_record(self, slot: WorkingSlot, task: Literal["experiment", "idea"] = "idea") -> Dict[str, Any]:
        if task == "experiment":
            system_prompt = (
                "You are a senior research archivist. Convert the WorkingSlot into a reusable "
                "semantic memory entry that captures enduring, generalizable insights."
            )
            user_prompt = TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_EXPEIRMENT.format(dump_slot_json=dump_slot_json(slot))
        elif task == "idea":
            system_prompt = (
                "You are a senior research archivist curating long-term memory. "
                "Convert an IdeaAgent WorkingSlot into a semantic memory record that preserves durable, "
                "generalizable guidance. Focus on reusable defect→fix heuristics, anti-pattern guardrails, "
                "field knowledge, and evaluation protocols—not run-specific chatter. "
                "Return only the structured output requested by the user prompt."
            )
            user_prompt = TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_IDEA.format(dump_slot_json=dump_slot_json(slot))
        user_prompt += " /no_think"

        def post_process_semantic(payload: dict, slot: WorkingSlot) -> Dict[str, Any]:
            summary = payload.get("summary") or slot.summary
            detail = payload.get("detail") or slot.summary
            tags = payload.get("tags") or slot.tags
            sem_id = new_id(prefix="sem")
            created_at = now_iso()
            updated_at = now_iso()

            return {
                "id": sem_id,
                "summary": summary.strip() if isinstance(summary, str) else str(summary),
                "detail": detail.strip() if isinstance(detail, str) else str(detail),
                "tags": list(tags) if tags else [],
                "created_at": created_at,
                "updated_at": updated_at,
            }

        return self._retry_llm_to_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            record_tag="semantic-record",
            slot=slot,
            post_process_payload=post_process_semantic,
            max_retries=1,
            max_tokens=2048,
        )

    def transfer_slot_to_episodic_record(self, slot: WorkingSlot, task: Literal["experiment", "idea"] = "idea") -> Dict[str, Any]:
        if task == "experiment":
            system_prompt = (
                "You are a scientific lab journal assistant. Convert the WorkingSlot into an episodic "
                "memory capturing Situation → Action → Result, including measurable outcomes."
            )
            user_prompt = TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_EXPRIMENT.format(dump_slot_json=dump_slot_json(slot), stage=slot.stage)
        elif task == "idea":
            system_prompt = (
                "You are a lab-notebook style recorder. "
                "Convert an IdeaAgent WorkingSlot into an episodic record capturing a specific MCTS traversal segment. "
                "Preserve Situation → Action → Result with concrete operator applications, targeted defects, "
                "retrieved memory IDs, evaluation metrics, Pareto role, and any fairness/failure-mode instrumentation. "
                "Return only the structured output requested by the user prompt."
            )
            user_prompt = TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_IDEA.format(dump_slot_json=dump_slot_json(slot), stage=slot.stage)
        user_prompt += " /no_think"


        def post_process_episodic(payload: dict, slot: WorkingSlot) -> Dict[str, Any]:
            stage = payload.get("stage") or slot.stage
            summary = payload.get("summary") or slot.summary
            detail = payload.get("detail") or {}
            tags = payload.get("tags") or slot.tags
            epi_id = new_id(prefix="epi")
            created_at = now_iso()
            
            if not isinstance(detail, dict):
                detail = {"notes": detail}

            return {
                "id": epi_id,
                "stage": stage.strip() if isinstance(stage, str) else str(stage),
                "summary": summary.strip() if isinstance(summary, str) else str(summary),
                "detail": detail,
                "tags": list(tags) if tags else [],
                "created_at": created_at,
            }

        return self._retry_llm_to_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            record_tag="episodic-record",
            slot=slot,
            post_process_payload=post_process_episodic,
            max_retries=1,
            max_tokens=2048,
        )

    def transfer_slot_to_procedural_record(self, slot: WorkingSlot, task: Literal["experiment", "idea"] = "idea") -> Dict[str, Any]:
        if task == "experiment":
            system_prompt = (
                "You are an expert operations documenter. Convert the WorkingSlot into a procedural "
                "memory entry describing reproducible steps/commands."
            )
            user_prompt = TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_EXPERIMENT.format(dump_slot_json=dump_slot_json(slot))
        elif task == "idea":
            system_prompt = (
                "You are an expert operations documenter. "
                "Convert an IdeaAgent WorkingSlot into a procedural memory entry that a future agent can execute to "
                "reproduce the workflow or evaluation harness. Emphasize trigger conditions (when to use), "
                "step-by-step actions (memory retrieval, operator application, reproducibility spec), "
                "evaluation/ablation requirements, and guardrails for failure modes. "
                "Return only the structured output requested by the user prompt."
            )
            user_prompt = TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_IDEA.format(dump_slot_json=dump_slot_json(slot))
        user_prompt += " /no_think"

        def post_process_procedural(payload: dict, slot: WorkingSlot) -> Dict[str, Any]:
            name = payload.get("name") or slot.topic or "skill"
            description = payload.get("description") or slot.summary
            steps = payload.get("steps") or []
            code = payload.get("code")
            tags = payload.get("tags") or slot.tags
            proc_id = new_id(prefix="proc")
            created_at = now_iso()
            updated_at = now_iso()

            if isinstance(steps, str):
                steps = [steps]

            clean_steps = [step.strip() for step in steps if isinstance(step, str) and step.strip()]

            return {
                "id": proc_id,
                "name": name.strip() if isinstance(name, str) else str(name),
                "description": description.strip() if isinstance(description, str) else str(description),
                "steps": clean_steps,
                "code": code.strip() if isinstance(code, str) else None,
                "tags": list(tags) if tags else [],
                "created_at": created_at,
                "updated_at": updated_at,
            }

        return self._retry_llm_to_record(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            record_tag="procedural-record",
            slot=slot,
            post_process_payload=post_process_procedural,
            max_retries=1,
            max_tokens=2048,
        )

    async def multi_thread_transfer_dicts_to_memories(self, is_abstract: bool = False):
        semantic_records = []
        episodic_records = []

        for i in self.slot_process.memory_dict:
            if i['memory_type'] == 'semantic':
                semantic_records.append(self.semantic_memory_system.instantiate_sem_record(**i['input']))
            elif i['memory_type'] == 'episodic':
                episodic_records.append(self.episodic_memory_system.instantiate_epi_record(**i['input']))
        
        if is_abstract and len(episodic_records) > 0:
            await self.abstract_episodic_records_to_semantic_record(episodic_records)

        if len(semantic_records) > 0:
            try:
                self.semantic_memory_system.upsert_normal_records(semantic_records)
            except Exception as e:
                import traceback
                print("[ERROR] upsert_normal_records for semantic_records failed:", repr(e))
                traceback.print_exc()
        if len(episodic_records) > 0:
            try:
                self.episodic_memory_system.upsert_normal_records(episodic_records)
            except Exception as e:
                import traceback
                print("[ERROR] upsert_normal_records for episodic_records failed:", repr(e))
                traceback.print_exc()

    def multi_thread_process(self, func, max_workers: int = 5, **kwargs) -> None:
        '''
        Multi-thread process for: 1. transfer context to working slots; 2. filter and route working slots; 3. transfer working slots to long-term memories.
        '''
        # func is your processing function
        try:
            func(**kwargs) # call your processing function here, your processing function MUST load context to working slots first!
            working_slots = self.slot_container.values()
            num_slots = len(working_slots)

            # filter and route
            print(f"[Info] Filtering and routing {num_slots} slots")
            _multi_thread_run(self.multi_thread_filter_and_route_slot, working_slots, max_workers=max_workers)
            routed_slots = llm_client.slot_process.routed_slot_container
            num_routed_slots = len(routed_slots)
            print(f"[Info] Transferring memories from {num_routed_slots} slots to memory systems")

            # generate memories in multi-threaded way
            _multi_thread_run(self.multi_thread_transfer_slot_to_memory, routed_slots, max_workers=max_workers)

            # transfer memories to records
            asyncio.run(self.multi_thread_transfer_dicts_to_memories(is_abstract=args.abstract_memories))

        except Exception as e:
            print(f"Error processing batch: {e}")

        print(f"[Info] Semantic memory size: {llm_client.semantic_memory_system.size}")
        print(f"[Info] Episodic memory size: {llm_client.episodic_memory_system.size}")
        print(f"[Info] Procedural memory size: {llm_client.procedural_memory_system.size}")        


class Schema:
    """
    Simple helper to build json_schema payloads for LLM structured outputs.
    """

    def __init__(self, max_slots: int = 20) -> None:
        self.max_slots = max_slots
        self.EXPERIMENT_TASK_SLOT_SCHEMA = self._build_slot_schema(EXPERIMENT_AGENT_STAGE_OPTIONS)
        self.IDEA_TASK_SLOT_SCHEMA = self._build_slot_schema(IDEA_AGENT_STAGE_OPTIONS)

    def _build_slot_schema(self, stage_enums: List[str]) -> Dict[str, Any]:
        slot_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "stage": {"type": "string", "enum": stage_enums},
                "topic": {"type": "string", "minLength": 1, "maxLength": 80},
                "summary": {"type": "string", "minLength": 1, "maxLength": 600},
                "attachments": {
                    "type": "object",
                    "additionalProperties": True,
                    "default": {},
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 32},
                    "maxItems": 5,
                    "default": [],
                },
            },
            "required": ["stage", "topic", "summary"],
            "additionalProperties": False,
        }

        return {
            "type": "object",
            "properties": {
                "slots": {
                    "type": "array",
                    "items": slot_schema,
                    "minItems": 1,
                    "maxItems": self.max_slots,
                }
            },
            "required": ["slots"],
            "additionalProperties": False,
        }
