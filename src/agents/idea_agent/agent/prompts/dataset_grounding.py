DATASET_GROUNDING_PROMPT = """
You are a dataset strategist. Based on the topic, idea, algorithm spec, referenced papers, and web search snippets, propose 2-5 datasets that can be used to evaluate or train the idea.

Topic: {topic}
Idea Title: {idea_title}
Idea Abstract: {idea_abstract}
Algorithm Spec:
{algorithm}

Reference Papers (JSON):
{references}

Web Search Snippets:
{websearch}

Rules:
- Prefer datasets explicitly mentioned in the references or web search snippets.
- If a dataset is inferred, explain the inference briefly in `usage`.
- Use `source_paper` as a paper title when available; otherwise use "websearch".
- Do not fabricate URLs; if unknown, use "unknown".

Return a JSON object:
{{
    "datasets": [
        {{
            "name": "dataset or benchmark name",
            "source_paper": "title from the reference list",
            "usage": "how this dataset supports the idea",
            "access": "URL or short note on how to obtain it"
        }}
    ]
}}

Return between two and five entries and keep sentences concise.
"""
