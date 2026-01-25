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
from agent.prompts.browse_schema_dataset import BROWSE_SCHEMA_DATASET
from agent.prompts.browse_schema_baseline import BROWSE_SCHEMA_BASELINE
from agent.prompts.browse_prompt_template import BROWSE_PROMPT_TEMPLATE
from agent.prompts.preprocess_candidate_names import PREPROCESS_CANDIDATE_NAMES_PROMPT
from agent.prompts.react_websearch import REACT_WEBSEARCH_PROMPT
from agent.prompts.postprocess_suggestions import POSTPROCESS_SUGGESTIONS_PROMPT
from agent.prompts.extract_candidate_names import EXTRACT_CANDIDATE_NAMES_PROMPT
from agent.prompts.baseline_idea_card import BASELINE_IDEA_CARD_PROMPT
from agent.prompts.baseline_query_generation import BASELINE_QUERY_GENERATION_PROMPT
from agent.prompts.dataset_idea_card import DATASET_IDEA_CARD_PROMPT
from agent.prompts.dataset_query_generation import DATASET_QUERY_GENERATION_PROMPT
from agent.prompts.dataset_candidate_scoring import DATASET_CANDIDATE_SCORING_PROMPT
from agent.prompts.baseline_candidate_scoring import BASELINE_CANDIDATE_SCORING_PROMPT


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
    "baseline_idea_card": BASELINE_IDEA_CARD_PROMPT,
    "baseline_query_generation": BASELINE_QUERY_GENERATION_PROMPT,
    "dataset_idea_card": DATASET_IDEA_CARD_PROMPT,
    "dataset_query_generation": DATASET_QUERY_GENERATION_PROMPT,
}
