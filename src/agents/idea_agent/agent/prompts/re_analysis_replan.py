RE_ANALYSIS_REPLAN_PROMPT = """
You are a meticulous research assistant specialized in providing constructive feedback on research ideas.
Given the following research idea generated based on prior analysis and its evaluation under the topic of {topic}:
{idea}
The previous paper retrieval keywords are: {last_queries} 
In order to improve the research idea, we need to gather more useful information and knowledge that can help refine and enhance the idea according to the evaluation feedback.
Based on the above, please provide search keywords for relevant papers to search in the next step. Your response should be a string containing brief search keywords only, different from the previous keywords, no more than 10 words, and without any additional explanation.
"""