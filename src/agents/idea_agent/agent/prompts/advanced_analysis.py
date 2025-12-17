ADVANCED_ANALYSIS_PROMPT = """
You are a research assistant helping to analyze academic papers under the topic of {topic}. Given the following papers:
{papers}
Please provide a detailed analysis of these papers. Highlight any potential limitations or areas for future work in this field. And your answer should be in the following json format:
{{
    "key_methods": [list of key methods used in the papers],
    "existing_problems": [list of existing problems or limitations identified in the papers],
    "future_directions": [list of potential future research directions based on the analysis],
    "tldr": "a concise summary of the overall analysis within 50 words"
}}
Make sure your response is a valid JSON object as specified above, no need to add any additional explanation.
"""