import json

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Any
from collections import deque
from memory.memory_system.utils import (
    dump_slot_json, 
    _extract_json_between, 
    _hard_validate_slot_keys,
    _build_context_snapshot,
    _safe_dump,
    _truncate_text,
    compute_overlap_score,
)
from memory.memory_system.user_prompt import (
    WORKING_SLOT_COMPRESS_USER_PROMPT,
    TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT,
    TRANSFER_SLOT_TO_TEXT_PROMPT,
    TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT,
    TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT,
    TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT,
)
from textwrap import dedent
from memory.memory_system import WorkingSlot, OpenAIClient, LLMClient
from memory.memory_system.models import (
    EpisodicRecord,
    SemanticRecord,
    ProceduralRecord,
)

class SlotProcess:
    def __init__(self):
        self.slot_container: Dict[str, WorkingSlot] = {}
        self.filtered_slot_container: List[WorkingSlot] = []
        self.routed_slot_container: List[Dict] = []
        self.llm_model = OpenAIClient()

    def add_slot(self, slot: WorkingSlot) -> None:
        self.slot_container[slot.to_dict().get('id')] = slot
    
    def clear_container(self) -> None:
        self.slot_container = {}

    def get_container_size(self) -> int:
        return len(self.slot_container)

    def query(self, query_text: str, limit: int = 5, key_words: Optional[List[str]] = None) -> List[Tuple[float, WorkingSlot]]:
        k = min(limit, len(self.slot_container))

        scored_slots: List[Tuple[float, WorkingSlot]] = []
        for slot in self.slot_container.values():
            score = compute_overlap_score(query_text, slot.summary, key_words)
            scored_slots.append((score, slot))
        scored_slots.sort(key=lambda x: x[0], reverse=True)
        
        return scored_slots[:k]
        
    async def filter_and_route_slots(self) -> List[Dict[str, WorkingSlot]]:
        for slot in self.slot_container.values():
            check_result = await slot.slot_filter(self.llm_model)
            print(check_result)
            if check_result == True:
                self.filtered_slot_container.append(slot)
        
        try:
            for filtered_slot in self.filtered_slot_container:
                route_result = await filtered_slot.slot_router(self.llm_model)
                pair = {
                    "memory_type": route_result,
                    "slot": filtered_slot
                }
                self.routed_slot_container.append(pair)
        except Exception as e:
            print(f"Routing error: {e}")
        
        return self.routed_slot_container
    
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

    async def transfer_experiment_agent_context_to_working_slots(self, context, state: str, max_slots: int = 50) -> List[WorkingSlot]:
        
        '''if not isinstance(context, WorkflowContext):
            raise TypeError("context must be an instance of WorkflowContext")'''

        if stage not in {"pre_analysis", "code_plan", "code_implement", "code_judge", "experiment_execute", "experiment_analysis"}:
            return []

        snapshot = _build_context_snapshot(context, state)

        system_prompt = (
            "You are an expert workflow archivist. "
            "Transform the provided Experiment Agent context into WorkingSlot JSON objects. "
            "Each slot must capture the stage, topic, summary (≤120 words), attachments, and tags. "
            "Summaries must follow a Situation→Action→Result narrative whenever possible."
        )

        user_prompt = TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT.format(
            max_slots=max_slots,
            snapshot=snapshot,
        )

        response = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = _extract_json_between(response, "working-slots", "working-slots")
        if not payload:
            return []

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse WorkingSlot JSON: {exc}") from exc

        slots_data = data.get("slots", [])
        if not isinstance(slots_data, list):
            raise ValueError("`slots` must be a list.")

        working_slots: List[WorkingSlot] = []
        allowed_keys = {"stage", "topic", "summary", "attachments", "tags"}

        for slot_dict in slots_data[:max_slots]:
            if not isinstance(slot_dict, dict):
                continue
            _hard_validate_slot_keys(slot_dict, allowed_keys=allowed_keys)

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
                    input_dict = await self.transfer_slot_to_semantic_record(slot)
                elif memory_type == "episodic":
                    input_dict = await self.transfer_slot_to_episodic_record(slot)
                else:
                    input_dict = await self.transfer_slot_to_procedural_record(slot)
            except Exception as exc:
                print(
                    f"[MEMORY] Failed to convert slot {getattr(slot, 'id', 'unknown')} "
                    f"({memory_type}): {exc}"
                )
                continue

            if inputs is not None:
                inputs.append({"memory_type": memory_type, "input": input_dict})

        return inputs

    async def transfer_slot_to_semantic_record(self, slot: WorkingSlot) -> Dict[str, Any]:
        system_prompt = (
            "You are a senior research archivist. Convert the WorkingSlot into a reusable "
            "semantic memory entry that captures enduring, generalizable insights."
        )

        user_prompt = TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT.format(dump_slot_json=dump_slot_json(slot))

        response = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = _extract_json_between(response, "semantic-record", "semantic-record")

        summary = payload.get("summary") or slot.summary
        detail = payload.get("detail") or slot.summary
        tags = payload.get("tags") or slot.tags

        return {
            "summary": summary.strip(),
            "detail": detail.strip(),
            "tags": list(tags),
        }

    async def transfer_slot_to_episodic_record(self, slot: WorkingSlot) -> Dict[str, Any]:
        system_prompt = (
            "You are a scientific lab journal assistant. Convert the WorkingSlot into an episodic "
            "memory capturing Situation → Action → Result, including measurable outcomes."
        )

        user_prompt = TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT.format(dump_slot_json=dump_slot_json(slot), stage=slot.stage)

        response = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = _extract_json_between(response, "episodic-record", "episodic-record")

        stage = payload.get("stage") or slot.stage
        summary = payload.get("summary") or slot.summary
        detail = payload.get("detail") or {}
        tags = payload.get("tags") or slot.tags

        if not isinstance(detail, dict):
            detail = {"notes": detail}

        return {
            "stage": stage.strip(),
            "summary": summary.strip(),
            "detail": detail,
            "tags": list(tags),
        }

    async def transfer_slot_to_procedural_record(self, slot: WorkingSlot) -> Dict[str, Any]:
        system_prompt = (
            "You are an expert operations documenter. Convert the WorkingSlot into a procedural "
            "memory entry describing reproducible steps/commands."
        )

        user_prompt = TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT.format(dump_slot_json=dump_slot_json(slot))

        response = await self.llm_model.complete(system_prompt=system_prompt, user_prompt=user_prompt)
        payload = _extract_json_between(response, "procedural-record", "procedural-record")

        name = payload.get("name") or slot.topic or "skill"
        description = payload.get("description") or slot.summary
        steps = payload.get("steps") or []
        code = payload.get("code")
        tags = payload.get("tags") or slot.tags

        if isinstance(steps, str):
            steps = [steps]

        clean_steps = [step.strip() for step in steps if isinstance(step, str) and step.strip()]

        return {
            "name": name.strip(),
            "description": description.strip(),
            "steps": clean_steps,
            "code": code.strip() if isinstance(code, str) else None,
            "tags": list(tags),
        }
