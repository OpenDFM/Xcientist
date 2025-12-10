# from utils.api_call import ChatAgent
# from omegaconf import OmegaConf
# from utils.chat_utils import load_prompt

# config = OmegaConf.load("config/deep_survey.yaml")
# chat_agent = ChatAgent(config)

# seed_title = "AutoSurvey: Large Language Models Can Automatically Write Surveys"
# seed_abstract = """This paper introduces AutoSurvey, a speedy and well-organized methodology for automating the creation of comprehensive literature surveys in rapidly evolving fields like artificial intelligence. Traditional survey paper creation faces challenges due to the vast volume and complexity of information, prompting the need for efficient survey methods. While large language models (LLMs) offer promise in automating this process, challenges such as context window limitations, parametric knowledge constraints, and the lack of evaluation benchmarks remain. AutoSurvey addresses these challenges through a systematic approach that involves initial retrieval and outline generation, subsection drafting by specialized LLMs, integration and refinement, and rigorous evaluation and iteration. Our contributions include a comprehensive solution to the survey problem, a reliable evaluation method, and experimental validation demonstrating AutoSurvey's effectiveness.We open our resources at \\url{https://github.com/AutoSurveys/AutoSurvey}."""
# candidate_title = "ResearchQA: Evaluating Scholarly Question Answering at Scale Across 75 Fields with Survey-Mined Questions and Rubrics"
# candidate_abstract = """Evaluating long-form responses to research queries heavily relies on expert annotators, restricting attention to areas like AI where researchers can conveniently enlist colleagues. Yet, research expertise is widespread: survey articles synthesize knowledge distributed across the literature. We introduce ResearchQA, a resource for evaluating LLM systems by distilling survey articles from 75 research fields into 21K queries and 160K rubric items. Each rubric, derived jointly with queries from survey sections, lists query-specific answer evaluation criteria, i.e., citing papers, making explanations, and describing limitations. Assessments by 31 Ph.D. annotators in 8 fields indicate 96% of queries support Ph.D. information needs and 87% of rubric items should be addressed in system responses by a sentence or more. Using our rubrics, we are able to construct an automatic pairwise judge obtaining 74% agreement with expert judgments. We leverage ResearchQA to analyze competency gaps in 18 systems in over 7.6K pairwise evaluations. No parametric or retrieval-augmented system we evaluate exceeds 70% on covering rubric items, and the highest-ranking agentic system shows 75% coverage. Error analysis reveals that the highest-ranking system fully addresses less than 11% of citation rubric items, 48% of limitation items, and 49% of comparison items. We release our data to facilitate more comprehensive multi-field evaluations."""

# prompt = load_prompt(
#     "modules/prompts/paper_relatedness.md",
#     seed_title=seed_title,
#     seed_abstract=seed_abstract,
#     candidate_title=candidate_title,
#     candidate_abstract=candidate_abstract,
# )
# response = chat_agent.remote_chat(prompt)
# print("Paper Relatedness LLM Response:")
# print(response)
# import json

# text = """```json
# {
#   "relevance_score": 4,
#   "category": "related",
#   "reason": "Both papers focus on leveraging large language models for automating processes in research, but the candidate paper centers on evaluating LLM systems rather than creating surveys."
# }
# ```"""
# import re, json


# def extract_json(text):
#     # 抓{...}之间的内容
#     m = re.search(r"\{[\s\S]*\}", text)
#     if not m:
#         raise ValueError("No JSON found")
#     return json.loads(m.group())


# print(extract_json(text))

# import diskcache as dc

# cache = dc.Cache(
#     "/hpc_stor03/sjtu_home/da.ma/src/deep-survey/database/workcollector_relatedness_cache"
# )

# for key in cache:
#     print("Key:", key)
#     print("Value:", cache[key])
#     print("------")


from utils.api_call import ChatAgent
from omegaconf import OmegaConf
from modules.pe import PAPER_DEEP_READING

config = OmegaConf.load("config/deep_survey.yaml")
chat_agent = ChatAgent(config)

with open("database/parsed_papers/2503.15573/auto/2503.15573.md", "r") as f:
    paper_markdown_text = f.read()
prompt = PAPER_DEEP_READING.format(paper_markdown_text=paper_markdown_text)
response = chat_agent.remote_chat(prompt, temperature=0.3)
print("Paper Deep Reading LLM Response:")
print(response)
