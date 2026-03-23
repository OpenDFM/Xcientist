from typing import List

MEMORY_FORMAT = {
    "topic": str, # research topic
    "survey": str, # survey of the topic
    "background_knowledge": List[str], # list of background knowledge strings
    "analysis":  List[str], # list of analysis strings
    "references": List[dict], # list of reference dicts
    "rag_query": List[str], # list of refined queries for outcome RAG
    "rag_hits": List[dict], # list of outcome RAG hits
    "paper_contents": dict, # mapping from paper_id -> parsed content metadata
    "idea_pool": List[str], # list of research ideas
    "dialogue": dict, # dialogue history
    "steps": List[str], # list of steps taken
    "memory_structure": dict # structure of the memory
}

def memory_init() -> dict:
    memory = {
        "topic": [],
        "survey": "",
        "background_knowledge": [],
        "analysis": [],
        "references": [],
        "rag_query": [],
        "rag_hits": [],
        "idea_pool": [],
        "evaluations": [],
        "retrieval_keywords": [],
        "paper_contents": {},
        "dialogue": {},
        "steps": [],
        "memory_structure": {}
    }
    return memory
