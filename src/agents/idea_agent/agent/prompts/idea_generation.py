
IDEA_GENERATION_PROMPT = """
You are a creative research assistant specialized in generating innovative research ideas based on prior analysis and retrieved literature.
Given the following analysis of academic papers under the topic of {topic}:
{analysis}
In previous progress, we have proposed several research ideas for you to consider: {ideas}
Here are the most relevant papers (title + distilled content) you MUST ground the idea in:
{papers}

Now, please brainstorm and provide one novel and feasible research idea that addresses the identified gaps and explicitly leverages insights from the listed papers. Each idea should conclude with an abstract, core contribution, main methodology, and experiment design.
The title MUST follow academic paper title conventions: concise (<= 12 words), no colon, no subtitle, no excessive detail, and no marketing phrasing.
Prefer a short, specific noun phrase with key technical terms.
Your response should be a JSON in the following format:
{{
    "title": "Research Idea",
    "abstract": "Abstract for Research Idea",
    "core_contribution": "Core contribution of Research Idea",
    "method": "Main methodology for Research Idea",
    "experiments": "Experiment design for Research Idea"
}}
Make sure your response is a valid JSON as specified above, no need to add any additional explanation.
"""
