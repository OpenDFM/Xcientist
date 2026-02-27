TOPIC_BACKGROUND_PROMPT = """
You are preparing a concise technical brief before launching a research agent to explore "{topic}".
Summarize distilled field knowledge that can prime downstream idea search.

Return STRICT JSON with:
{{
  "background": "≤120 word overview of the topic's status quo and why it matters now",
  "key_questions": ["3 critical unsolved questions phrased concisely"],
  "canonical_methods": ["core method families to anchor the search"]
}}

Rules:
- Do not include Markdown.
- Keep bullet strings terse but specific enough for an ICML reviewer.
"""
