REFERENCE_GROUNDING_PROMPT = """
You are a meticulous research curator. Your job is to describe how the retrieved papers support the current idea.

== Topic == 
{topic}

== Idea Title == 
{idea_title}

== Idea Abstract == 
{idea_abstract}

== Algorithm Spec == 
{algorithm}

You will receive the raw papers as JSON:
{references}

Please return a JSON object with this format:
{{
    "reference_papers": [
        {{
            "title": "exact title from the input list",
            "authors": "comma-separated author names if available",
            "year": "year if known",
            "url": "paper URL if available",
            "summary": "2 sentences summarizing the paper",
            "contribution": "Explain how this paper influences or enables the idea above."
        }}
    ]
}}

Do not introduce new papers that are not in the input. Keep descriptions concise and specific to the idea.
"""
