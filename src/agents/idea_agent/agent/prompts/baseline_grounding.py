BASELINE_GROUNDING_PROMPT = """
You are a baseline scout. Based on the topic, idea, algorithm spec, referenced papers, and web search snippets, propose 2-5 baselines for fair comparison.

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
- Prefer baselines with a GitHub repository link; include it in `repo_url` if available.
- If a baseline comes from a paper, set `source` to the paper title; otherwise use "websearch".
- Do not fabricate URLs; if not available, leave `repo_url` empty.
- Keep descriptions concise and grounded in the input snippets.

Return a JSON object:
{{
    "baselines": [
        {{
            "name": "baseline method/system name",
            "source": "paper title or websearch",
            "repo_url": "https://github.com/owner/repo or empty string",
            "usage": "why this baseline is a relevant comparison"
        }}
    ]
}}

Return between two and five entries.
"""
