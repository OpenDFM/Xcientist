from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence

import networkx as nx

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "the",
    "to",
    "via",
    "with",
    "without",
    "method",
    "model",
    "approach",
    "framework",
    "system",
}


@dataclass(frozen=True)
class MethodPaperNode:
    node_id: str
    title: str
    paper_title: str
    keywords: str
    problem: str
    innovation: str
    scenarios: str
    degree: float
    tokens: Counter

# Currently saving the graph as .gexf in the idea_agent.
def _default_graph_path() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paper_graph.gexf"))


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = [token for token in _TOKEN_RE.findall(text.lower()) if token and token not in _STOPWORDS]
    return tokens


def _cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    for token, count in a.items():
        dot += count * b.get(token, 0)
    norm_a = math.sqrt(sum(count * count for count in a.values()))
    norm_b = math.sqrt(sum(count * count for count in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _node_title(node_id: str, data: Dict[str, str]) -> str:
    for key in ("paper_title", "full_name", "label"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(node_id)


def _is_method_node(data: Dict[str, str]) -> bool:
    group = data.get("group")
    return isinstance(group, str) and group.strip().lower() == "method"


def _is_method_paper_node(data: Dict[str, str]) -> bool:
    if not _is_method_node(data):
        return False
    paper_title = data.get("paper_title")
    keywords = data.get("keywords")
    return bool(isinstance(paper_title, str) and paper_title.strip()) and bool(
        isinstance(keywords, str) and keywords.strip()
    )


@lru_cache(maxsize=2)
def _load_graph(graph_path: str) -> nx.Graph:
    return nx.read_gexf(graph_path)


@lru_cache(maxsize=2)
def _load_method_paper_nodes(graph_path: str) -> List[MethodPaperNode]:
    graph = _load_graph(graph_path)
    nodes: List[MethodPaperNode] = []
    for node_id, data in graph.nodes(data=True):
        if not _is_method_paper_node(data):
            continue
        paper_title = (data.get("paper_title") or "").strip()
        keywords = (data.get("keywords") or "").strip()
        title = _node_title(node_id, data)
        tokens = Counter(_tokenize(paper_title))
        nodes.append(
            MethodPaperNode(
                node_id=node_id,
                title=title,
                paper_title=paper_title,
                keywords=keywords,
                problem=(data.get("problem") or "").strip(),
                innovation=(data.get("innovation") or "").strip(),
                scenarios=(data.get("scenarios") or "").strip(),
                degree=float(graph.degree(node_id)),
                tokens=tokens,
            )
        )
    return nodes


def rank_method_paper_nodes_weighted(
    topic: str,
    graph_path: Optional[str] = None,
    top_k: int = 20,
    degree_weight: float = 0.5,
    similarity_weight: float = 0.5,
) -> List[Dict[str, float]]:
    path = graph_path or _default_graph_path()
    nodes = _load_method_paper_nodes(path)
    if not nodes:
        return []
    max_degree = max((node.degree for node in nodes), default=0.0) or 1.0
    topic_tokens = Counter(_tokenize(topic))
    scored = []
    for node in nodes:
        degree_score = node.degree / max_degree
        similarity_score = _cosine_similarity(topic_tokens, node.tokens)
        combined = degree_weight * degree_score + similarity_weight * similarity_score
        scored.append((combined, degree_score, similarity_score, node))
    scored.sort(key=lambda item: (-item[0], item[3].paper_title or item[3].title))
    return [
        {
            "node_id": node.node_id,
            "title": node.title,
            "paper_title": node.paper_title,
            "keywords": node.keywords,
            "problem": node.problem,
            "innovation": node.innovation,
            "scenarios": node.scenarios,
            "degree": node.degree,
            "degree_score": degree_score,
            "similarity_score": similarity_score,
            "score": combined,
        }
        for combined, degree_score, similarity_score, node in scored[: max(top_k, 0)]
    ]


def score_component_explanation_embeddings_for_novelty(
    explanation_embeddings: Sequence[Sequence[float]],
    components_with_explanations: Optional[Sequence[Dict[str, str]]] = None,
    graph_path: Optional[str] = None,
) -> float:
    """Placeholder hook for paper-graph novelty scoring.

    The evaluator embeds each component explanation with ``all-MiniLM-L6-v2`` and
    then calls this function. A future implementation can project the embedding
    set into the paper graph and return a calibrated novelty score in ``[0, 5]``.
    """
    raise NotImplementedError(
        "paper-graph component novelty scoring is not implemented yet."
    )


def supports_component_novelty_scoring() -> bool:
    """Whether the paper-graph novelty hook is implemented and safe to call."""
    return False
