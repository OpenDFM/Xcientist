PAPER_FILTERING_PROMPT = """
You are a careful research triage assistant.

Topic: {topic}
Mature idea (optional): {mature_idea}

You will receive the retrieved papers as JSON:
{papers}

Task:
1) Select the top {top_k} papers that are most relevant and useful for idea generation about the topic
   (and the mature idea if provided).
2) For every remaining paper, write a single-sentence summary that includes background + method + effect.

Return a JSON object with this exact format:
{{
  "top_paper_ids": ["paper_id", "paper_id", "..."],
  "compressed": [
    {{"paper_id": "paper_id", "summary": "one sentence"}}
  ]
}}

Rules:
- Use only paper_id values from the input list. Do NOT invent papers.
- If fewer than {top_k} papers are provided, include all in top_paper_ids.
- The compressed list must include every paper not in top_paper_ids exactly once.
- Each summary must be exactly one sentence and cover background + method + effect.
- Keep summaries factual and concise.
"""
