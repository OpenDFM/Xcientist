from typing import List

ARTIFACT_FORMAT = {
    "topic": str, # research topic
    "run_topic": str, # original topic from launcher/env
    "survey": str, # survey of the topic
    "mature_idea": str, # mature idea text (user-provided or from re_analysis_replan)
    "background_knowledge": List[str], # list of background knowledge strings
    "analysis":  List[str], # list of analysis strings
    "references": List[dict], # list of reference dicts
    "rag_query": List[str], # list of refined queries for outcome RAG
    "rag_hits": List[dict], # list of outcome RAG hits
    "rag_contents": List[str], # list of survey content strings
    "paper_contents": dict, # mapping from paper_id -> parsed content metadata
    "idea_pool": List[dict], # list of canonical idea payloads
    "ligagent_pro_candidates": List[dict], # raw per-mode best ideas from LigAgent-Pro
    "fusion_result": dict, # latest fuse-agent output and fused candidate
    "dialogue": dict, # dialogue history
    "steps": List[str], # list of steps taken
    "artifact_structure": dict, # structure of the artifact
    "workflow_trace": List[dict], # explicit stage execution trace
    "workflow_state": dict, # latest workflow/stage status
    "context_slots": dict, # lightweight named cross-stage context
    "operation_trace": List[dict], # llm/tool op-level trace
}

def artifact_init() -> dict:
    artifact = {
        "topic": [],
        "run_topic": "",
        "survey": "",
        "mature_idea": "",
        "background_knowledge": [],
        "analysis": [],
        "references": [],
        "rag_query": [],
        "rag_hits": [],
        "rag_contents": [],
        "idea_pool": [],
        "evaluations": [],
        "retrieval_keywords": [],
        "paper_contents": {},
        "dialogue": {},
        "steps": [],
        "ligagent_pro_candidates": [],
        "fusion_result": {},
        "artifact_structure": {},
        "workflow_trace": [],
        "workflow_state": {},
        "context_slots": {},
        "operation_trace": [],
    }
    return artifact
