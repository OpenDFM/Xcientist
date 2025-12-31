
IDEA_GENERATION_PROMPT = """
You are a creative research assistant specialized in generating innovative research ideas based on prior analysis and retrieved literature.
Given the following analysis of academic papers under the topic of {topic}:
{analysis}
In previous progress, we have proposed several research ideas for you to consider: {ideas}
Here are the most relevant papers (title + distilled content) you MUST ground the idea in:
{papers}

Now, please brainstorm and provide one novel and feasible research idea that addresses the identified gaps and explicitly leverages insights from the listed papers. Each idea should conclude with an abstract, core contribution, main methodology, and experiment design.
Your response should be a JSON in the following format:
{{
    "title": "Research Idea",
    "abstract": "Abstract for Research Idea",
    "core_contribute": "Core contribution of Research Idea",
    "methodology": "Main methodology for Research Idea",
    "experiment_design": "Experiment design for Research Idea"
}}
Make sure your response is a valid JSON as specified above, no need to add any additional explanation.
"""
