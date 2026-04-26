"""Search Paper Abstract Tool - OpenHands Format"""

import os
import sys
from typing import List, Dict, Any, Sequence
import logging
from pydantic import Field

from openhands.sdk import (
    Action,
    ImageContent,
    Observation,
    TextContent,
)
from openhands.sdk.tool import (
    ToolExecutor,
    register_tool,
    ToolDefinition,
)


logger = logging.getLogger(__name__)

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

# Import the search function
from blog_agent.utils.semantic_scholar import search_paper_and_get_abstract


# --- Action / Observation ---


class SearchPaperAbstractAction(Action):
    """Search for a paper on Semantic Scholar and retrieve its full abstract."""

    query: str = Field(
        description="Search query - can be paper title, keywords, or any text to find the paper"
    )


class SearchPaperAbstractObservation(Observation):
    """Result of searching paper abstract from Semantic Scholar."""

    status: str = Field(
        description="Status: 'success' or 'fail'"
    )
    detail: str = Field(
        description="Detail: failure reason or paper info"
    )
    paper: Dict[str, Any] = Field(
        default_factory=dict,
        description="Paper details including title, abstract, year, authors, venue, paper_id, url (only on success)"
    )

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.status == "fail":
            return [TextContent(text=f"Search failed: {self.detail}")]

        # Success case - show paper info
        p = self.paper
        lines = [
            f"Title: {p.get('title', 'N/A')}",
            f"Year: {p.get('year', 'N/A')}",
            f"Venue: {p.get('venue', 'N/A')}",
            f"Authors: {', '.join(p.get('authors', []))}",
            f"URL: {p.get('url', 'N/A')}",
            "",
            f"Abstract:",
            p.get('abstract', 'N/A')[:500] + "..." if p.get('abstract') and len(p.get('abstract', '')) > 500 else p.get('abstract', 'N/A'),
        ]

        return [TextContent(text="\n".join(lines))]


# --- Executor ---


class SearchPaperAbstractExecutor(ToolExecutor[SearchPaperAbstractAction, SearchPaperAbstractObservation]):
    """Executor that searches Semantic Scholar for paper abstracts."""

    def __call__(
        self,
        action: SearchPaperAbstractAction,
        conversation=None
    ) -> SearchPaperAbstractObservation:
        """Execute the search_paper_abstract action."""
        result = search_paper_and_get_abstract(
            query=action.query,
        )

        return SearchPaperAbstractObservation(
            status=result.get("status", "fail"),
            detail=result.get("detail", "Unknown error"),
            paper=result.get("paper", {})
        )


# --- Tool Description ---
_SEARCH_PAPER_ABSTRACT_DESCRIPTION = """Search Semantic Scholar for academic papers and retrieve full abstracts.
* Takes a search query (paper title, keywords, or any text)
* Returns paper details including: title, abstract, year, authors, venue, URL
* Useful for finding detailed paper information beyond what is available in the knowledge graph
* Supports fuzzy search - not limited to exact title matches
"""


# --- Tool Definition ---


class SearchPaperAbstractTool(ToolDefinition[SearchPaperAbstractAction, SearchPaperAbstractObservation]):
    """A custom tool for searching paper abstracts from Semantic Scholar."""

    @classmethod
    def create(cls, conv_state) -> Sequence[ToolDefinition]:
        """Create SearchPaperAbstractTool instance.

        Args:
            conv_state: Conversation state (not used but required by interface).

        Returns:
            A sequence containing a single SearchPaperAbstractTool instance.
        """
        executor = SearchPaperAbstractExecutor()

        return [
            cls(
                description=_SEARCH_PAPER_ABSTRACT_DESCRIPTION,
                action_type=SearchPaperAbstractAction,
                observation_type=SearchPaperAbstractObservation,
                executor=executor,
            )
        ]


# --- Registration ---
def _make_search_paper_abstract_tool(conv_state) -> list[ToolDefinition]:
    """Create the search paper abstract tool."""
    return list(SearchPaperAbstractTool.create(conv_state))


register_tool("SearchPaperAbstractTool", _make_search_paper_abstract_tool)
