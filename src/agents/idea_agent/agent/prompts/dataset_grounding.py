DATASET_GROUNDING_PROMPT = """
You are a dataset strategist. Based on the topic, idea, algorithm spec, and the referenced papers, propose up to five datasets that can be used to evaluate or train the idea.

Topic: {topic}
Idea Title: {idea_title}
Idea Abstract: {idea_abstract}
Algorithm Spec:
{algorithm}

Reference Papers (JSON):
{references}

Rules:
- Each dataset must cite one of the provided papers (use its title in `source_paper`).
- If the paper does not name a dataset explicitly, infer the dataset scope (e.g., "datasets described in …") and explain the inference.
- Keep descriptions grounded—no hallucinated papers.

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

Limit to at most five entries and keep sentences concise.
"""
