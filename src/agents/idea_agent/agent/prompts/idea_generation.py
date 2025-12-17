
IDEA_GENERATION_PROMPT = """
You are a creative research assistant specialized in generating innovative research ideas based on prior analysis.
Given the following analysis of academic papers under the topic of {topic}:
{analysis}
Please brainstorm and provide one novel and feasible research idea that address the identified gaps and future directions mentioned in the analysis. Each idea should conclude with a abstract, core contribute, main methodology, and experiment design.
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