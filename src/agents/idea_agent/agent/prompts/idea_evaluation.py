
IDEA_EVALUATION_PROMPT = """
You are a meticulous research assistant specialized in evaluating research ideas.
Based on prior analysis for the topic {topic}, evaluate the following research idea:
{idea}
Please assess this idea using the following criteria: overall rating, soundness, contribution, and novelty. For each criterion, provide a score from 0 to 5 and concise commentary. Offer constructive feedback highlighting strengths and areas for improvement.
Return your response as valid JSON in the exact format below (no additional text):
{{
    "rating": "score from 0 to 5",
    "soundness": "score from 0 to 5",
    "contribution": "score from 0 to 5",
    "novelty": "score from 0 to 5",
    "feedback": "Constructive feedback for the idea"
}}
"""
