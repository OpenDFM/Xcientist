from typing import List

ARTIFACT_FORMAT = {
    "topic": str, # research topic
    "run_topic": str, # original topic from launcher/env
    "survey": str, # survey of the topic
    "background_knowledge": List[str], # list of background knowledge strings
    "analysis":  List[str], # list of analysis strings
    "references": List[dict], # list of reference dicts
    "rag_query": List[str], # list of refined queries for outcome RAG
    "rag_hits": List[dict], # list of outcome RAG hits
    "rag_contents": List[str], # list of survey content strings
    "paper_contents": dict, # mapping from paper_id -> parsed content metadata
    "idea_pool": List[str], # list of research ideas
    "dialogue": dict, # dialogue history
    "steps": List[str], # list of steps taken
    "artifact_structure": dict, # structure of the artifact
}

def artifact_init() -> dict:
    artifact = {
        "topic": [],
        "run_topic": "",
        "survey": "",
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
        "artifact_structure": {}
    }
    return artifact
