from agent.prompts.action_selection import ACTION_SELECTION_PROMPT
from agent.prompts.action_retry import ACTION_RETRY_PROMPT
from agent.prompts.advanced_analysis import ADVANCED_ANALYSIS_PROMPT
from agent.prompts.idea_generation import IDEA_GENERATION_PROMPT
from agent.prompts.idea_evaluation import IDEA_EVALUATION_PROMPT
from agent.prompts.mcts_generation import MCTS_IDEA_GENERATION_PROMPT
from agent.prompts.mcts_evaluation import MCTS_IDEA_EVALUATION_PROMPT
from agent.prompts.re_analysis_replan import RE_ANALYSIS_REPLAN_PROMPT
from agent.prompts.reference_grounding import REFERENCE_GROUNDING_PROMPT
from agent.prompts.dataset_grounding import DATASET_GROUNDING_PROMPT
from agent.prompts.baseline_grounding import BASELINE_GROUNDING_PROMPT
from agent.prompts.algorithm_structuring import ALGORITHM_STRUCTURING_PROMPT
from agent.prompts.algorithm_alignment import ALGORITHM_ALIGNMENT_PROMPT
from agent.prompts.idea_introduction import IDEA_INTRODUCTION_PROMPT
from agent.prompts.topic_background import TOPIC_BACKGROUND_PROMPT
from agent.prompts.rag_query import RAG_QUERY_PROMPT


PROMPTS = {
    "action_selection": ACTION_SELECTION_PROMPT,
    "action_retry": ACTION_RETRY_PROMPT,
    "advanced_analysis": ADVANCED_ANALYSIS_PROMPT,
    "idea_generation": IDEA_GENERATION_PROMPT,
    "idea_evaluation": IDEA_EVALUATION_PROMPT,
    "mcts_generation": MCTS_IDEA_GENERATION_PROMPT,
    "mcts_evaluation": MCTS_IDEA_EVALUATION_PROMPT,
    "re_analysis_replan": RE_ANALYSIS_REPLAN_PROMPT,
    "reference_grounding": REFERENCE_GROUNDING_PROMPT,
    "dataset_grounding": DATASET_GROUNDING_PROMPT,
    "baseline_grounding": BASELINE_GROUNDING_PROMPT,
    "algorithm_structuring": ALGORITHM_STRUCTURING_PROMPT,
    "algorithm_alignment": ALGORITHM_ALIGNMENT_PROMPT,
    "idea_introduction": IDEA_INTRODUCTION_PROMPT,
    "topic_background": TOPIC_BACKGROUND_PROMPT,
    "rag_query": RAG_QUERY_PROMPT,
}
