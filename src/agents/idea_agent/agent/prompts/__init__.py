from src.agents.idea_agent.agent.prompts.advanced_analysis import ADVANCED_ANALYSIS_PROMPT
from src.agents.idea_agent.agent.prompts.component_novelty_evaluation import (
    COMPONENT_NOVELTY_EVALUATION_PROMPT,
)
from src.agents.idea_agent.agent.prompts.mcts_evaluation import MCTS_IDEA_EVALUATION_PROMPT
from src.agents.idea_agent.agent.prompts.re_analysis_replan import RE_ANALYSIS_REPLAN_PROMPT
from src.agents.idea_agent.agent.prompts.reference_grounding import REFERENCE_GROUNDING_PROMPT
from src.agents.idea_agent.agent.prompts.algorithm_structuring import ALGORITHM_STRUCTURING_PROMPT
from src.agents.idea_agent.agent.prompts.algorithm_alignment import ALGORITHM_ALIGNMENT_PROMPT
from src.agents.idea_agent.agent.prompts.idea_introduction import IDEA_INTRODUCTION_PROMPT
from src.agents.idea_agent.agent.prompts.idea_fusion import IDEA_FUSION_PROMPT
from src.agents.idea_agent.agent.prompts.experiment_findings_extraction import (
    EXPERIMENT_FINDINGS_EXTRACTION_PROMPT,
)
from src.agents.idea_agent.agent.prompts.topic_background import TOPIC_BACKGROUND_PROMPT
from src.agents.idea_agent.agent.prompts.rag_query import RAG_QUERY_PROMPT


PROMPTS = {
    "advanced_analysis": ADVANCED_ANALYSIS_PROMPT,
    "component_novelty_evaluation": COMPONENT_NOVELTY_EVALUATION_PROMPT,
    "mcts_evaluation": MCTS_IDEA_EVALUATION_PROMPT,
    "re_analysis_replan": RE_ANALYSIS_REPLAN_PROMPT,
    "reference_grounding": REFERENCE_GROUNDING_PROMPT,
    "algorithm_structuring": ALGORITHM_STRUCTURING_PROMPT,
    "algorithm_alignment": ALGORITHM_ALIGNMENT_PROMPT,
    "idea_introduction": IDEA_INTRODUCTION_PROMPT,
    "idea_fusion": IDEA_FUSION_PROMPT,
    "experiment_findings_extraction": EXPERIMENT_FINDINGS_EXTRACTION_PROMPT,
    "topic_background": TOPIC_BACKGROUND_PROMPT,
    "rag_query": RAG_QUERY_PROMPT,
}
