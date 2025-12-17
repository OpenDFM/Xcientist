from doctest import testfile
from agent.tools.semantic_scholar_search import Semantic

semantic_tool = Semantic()

TOOLS = {
    "semantic_search": semantic_tool.search_papers,
    "semantic_recommend": semantic_tool.recommend_papers
}
