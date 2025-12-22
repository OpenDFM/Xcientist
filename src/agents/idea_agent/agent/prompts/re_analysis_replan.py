RE_ANALYSIS_REPLAN_PROMPT = """
You are a meticulous research assistant specialized in providing constructive feedback on research ideas.
Given the following research idea generated based on prior analysis and its evaluation under the topic of {topic}:
{idea}
The all previous paper retrieval keywords are: {last_queries} 
And the all previous topics are: {topics}
In order to improve the research idea, we need to change our research topic and gather more useful information and knowledge that can help refine and enhance the idea according to the evaluation feedback.
Based on the above, please provide search keywords for relevant papers to search in the next step. Your response should be a JSON format as follows:
{{
'new_topic': 'string',  # The new research topic(only one) that can help improve the idea, no more than 15 words.
'search_keywords': 'string'  # The new search keywords only, different from the previous keywords, no more than 10 words, and without any additional explanation.
}} 
Make sure your response is a valid JSON as specified above, no need to add any additional explanation.
"""