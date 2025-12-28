from agent.prompts.action_selection import ACTION_SELECTION_PROMPT
from agent.prompts.advanced_analysis import ADVANCED_ANALYSIS_PROMPT
from agent.prompts.idea_generation import IDEA_GENERATION_PROMPT
from agent.prompts.idea_evaluation import IDEA_EVALUATION_PROMPT
from agent.prompts.mcts_generation import MCTS_IDEA_GENERATION_PROMPT
from agent.prompts.mcts_evaluation import MCTS_IDEA_EVALUATION_PROMPT
from agent.prompts.re_analysis_replan import RE_ANALYSIS_REPLAN_PROMPT


PROMPTS = {
    "action_selection": ACTION_SELECTION_PROMPT,
    "advanced_analysis": ADVANCED_ANALYSIS_PROMPT,
    "idea_generation": IDEA_GENERATION_PROMPT,
    "idea_evaluation": IDEA_EVALUATION_PROMPT,
    "mcts_generation": MCTS_IDEA_GENERATION_PROMPT,
    "mcts_evaluation": MCTS_IDEA_EVALUATION_PROMPT,
    "re_analysis_replan": RE_ANALYSIS_REPLAN_PROMPT,
}
