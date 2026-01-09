from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


def _slot_attachments_schema() -> Dict[str, Any]:
    """
    Shared attachments schema used by both ExperimentAgent and IdeaAgent WorkingSlots.

    Attachments are OPTIONAL in prompts, and may include several optional groups.
    We keep the schema permissive (additionalProperties allowed) so the LLM
    can include only what it has.
    """
    return {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "notes": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            "metrics": {
                "type": "object",
                # metrics values can be number/string/bool depending on the source
                "additionalProperties": {"type": ["number", "string", "boolean", "null"]},
            },
            "issues": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "list": {"type": "array", "items": {"type": "string"}},
                },
            },
            "actions": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "list": {"type": "array", "items": {"type": "string"}},
                },
            },
            # IdeaAgent-specific optional groups
            "ideas": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            "operators": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            "memories": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "items": {"type": "array", "items": {"type": "string"}},
                },
            },
            # Allow additional agent-specific attachment groups (e.g., artifacts/procedures)
            "artifacts": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                },
            },
            "procedures": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }


def _working_slot_schema(stage_enum: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["stage", "topic", "summary", "tags"],
        "properties": {
            "stage": {"type": "string", "enum": stage_enum},
            "topic": {"type": "string", "minLength": 1},
            "summary": {"type": "string", "minLength": 1},
            "attachments": _slot_attachments_schema(),
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
            },
        },
    }


def _slots_envelope_schema(stage_enum: List[str], max_slots: int) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["slots"],
        "properties": {
            "slots": {
                "type": "array",
                "maxItems": max_slots,
                "items": _working_slot_schema(stage_enum),
            }
        },
    }


@dataclass
class Schema:
    """
    Provides JSON schemas for LLM structured outputs, aligned with:
    - TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT
    - TRANSFER_IDEA_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT

    Note:
      Prompts require "always emit at least one slot" for IdeaAgent in narrative rules,
      but schema cannot enforce minItems reliably when the prompt says it MAY be empty
      in other contexts. Therefore we keep minItems=0 and enforce cardinality in code.
    """

    max_slots: int = 50

    @property
    def EXPERIMENT_TASK_SLOT_SCHEMA(self) -> Dict[str, Any]:
        stage_enum = [
            "pre_analysis",
            "code_plan",
            "code_implement",
            "code_judge",
            "experiment_execute",
            "experiment_analysis",
            "meta",
        ]
        return _slots_envelope_schema(stage_enum=stage_enum, max_slots=self.max_slots)

    @property
    def IDEA_TASK_SLOT_SCHEMA(self) -> Dict[str, Any]:
        # The prompt injects stage_enums dynamically; this is a sane default set.
        # If your IdeaAgent uses different stages, update this list accordingly.
        stage_enum = [
            "knowledge_aquisition",
            "advanced_analysis",
            "idea_generation",
            "idea_evaluation",
            "re_analysis_replan",
            "meta",
        ]
        return _slots_envelope_schema(stage_enum=stage_enum, max_slots=self.max_slots)