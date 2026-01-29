RAG_QUERY_PROMPT = """
You are refining the research focus for a follow-up retrieval step.

Original topic: {topic}

Mature idea (optional; if empty, ignore):
{mature_idea}

The agent has read these papers (title, abstract, short keynote):
{papers}

Task:
- Propose ONE search query string aimed at surveying METHODS/MECHANISMS/ALGORITHMS (not benchmarks).
- The query should combine:
  (1) a narrower subtopic + (2) the most critical unresolved technical problem + (3) a method family / mechanism keyword.
- Prefer method-oriented terms such as: "architecture", "training objective", "representation", "alignment", "reasoning mechanism",
  "planning", "retrieval strategy", "memory", "optimization", "calibration", "uncertainty", "self-consistency".
- Avoid evaluation/benchmark/dataset framing. Do NOT center the query on: "benchmark", "dataset", "leaderboard", "evaluation",
  "metrics", "ablation", "human study", "user study", "baseline comparison".
- Keep it specific and technical; avoid generic phrasing.
- The query must be a single string no longer than 25 words.
- If mature_idea is provided, bias the query toward its mechanism keywords and invariants.

Return STRICT JSON (no prose):
{{
  "query": "..."
}}
"""
