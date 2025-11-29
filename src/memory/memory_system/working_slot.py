import asyncio

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Protocol
from memory.memory_system.utils import new_id, dump_slot_json
from pydantic import BaseModel, Field, field_validator, validate_call
from openai import OpenAI
from textwrap import dedent
from memory.memory_system.user_prompt import WORKING_SLOT_FILTER_USER_PROMPT, WORKING_SLOT_ROUTE_USER_PROMPT
from memory.memory_system.llm import OpenAIClient, LLMClient

class SlotPayload(BaseModel):
    id: str = Field(default_factory=lambda: new_id("work"))
    stage: str = Field("", description="Stage of the working.")
    topic: str = Field("", description="Topic of the working slot.")
    summary: str = Field("", description="Summary of the working slot.")
    attachments: Dict[str, Dict] = Field(
        default_factory=dict,
        description="List of attachment identifiers associated with the slot.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags associated with the slot.",
    )

class WorkingSlot(SlotPayload):
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "stage": self.stage,
            "topic": self.topic,
            "summary": self.summary,
            "attachments": self.attachments,
            "tags": self.tags,
        }
    
    async def slot_filter(self, llm: LLMClient) -> bool:
        system_prompt = "You are a memory access reviewer. Only output 'yes' or 'no'."
        user_prompt = WORKING_SLOT_FILTER_USER_PROMPT.format(slot_dump=dump_slot_json(self))
        out = await llm.complete(system_prompt, user_prompt)

        if out.strip().lower() not in ["yes", "no"]:
            raise ValueError(f"Invalid slot filter output: {out}")

        return True if out.strip().lower() == "yes" else False
    
    async def slot_router(self, llm: LLMClient) -> Literal["semantic", "procedural", "episodic"]:
        system_prompt = "You are a memory type classifier. Only output legal string: 'semantic', 'procedural', or 'episodic'."
        user_prompt = WORKING_SLOT_ROUTE_USER_PROMPT.format(slot_dump=dump_slot_json(self))
        out = await llm.complete(system_prompt, user_prompt)
        if out.strip() not in ["semantic", "procedural", "episodic"]:
            raise ValueError(f"Invalid slot type: {out}")
        return out
