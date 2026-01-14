RAG_QUERY_PROMPT = """
You are refining the research focus for a follow-up retrieval step.

Original topic: {topic}

The agent has read these papers (title, abstract, short keynote):
{papers}

Task:
- Propose ONE search query string that combines a narrower subtopic with the most critical unresolved problem surfaced by the papers.
- Keep it specific and technical; avoid generic phrasing.
- The query must be a single string no longer than 25 words.

Return STRICT JSON (no prose):
{{
  "query": "..."
}}
"""
