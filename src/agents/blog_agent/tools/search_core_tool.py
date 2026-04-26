"""Search Core Nodes Tool - OpenHands Format"""

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

# Import the search function from search_core
from blog_agent.utils.search_core import get_core_nodes_within_hops


# --- Action / Observation ---


class SearchCoreNodesAction(Action):
    """Search for Core nodes within N hops from a root node in the knowledge graph."""

    root_label: str = Field(
        description="The label of the root node to start search from (supports exact and fuzzy matching)"
    )
    max_hops: int = Field(
        default=3,
        description="Maximum number of hops to traverse (default: 3)"
    )


class SearchCoreNodesObservation(Observation):
    """Result of searching core nodes within hops."""

    status: str = Field(
        description="Status: 'success' or 'fail'"
    )
    detail: str = Field(
        description="Detail: fail reason with optional query, or hop statistics on success"
    )
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of core nodes with id, paper_title, and hops (only on success)"
    )

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.status == "fail":
            return [TextContent(text=f"Search failed: {self.detail}")]

        # Success case - show preview
        if not self.results:
            return [TextContent(text="No core nodes found.")]

        # Show first 10 results as preview
        preview_lines = []
        for node in self.results[:10]:
            preview_lines.append(
                f"- [{node.get('hops')} hop] {node.get('paper_title', 'N/A')}"
            )

        preview = "\n".join(preview_lines)
        more = "\n... and more" if len(self.results) > 10 else ""

        ret = (
            f"{self.detail}\n\n"
            f"Results:\n{preview}{more}"
        )
        return [TextContent(text=ret)]


# --- Executor ---


class SearchCoreNodesExecutor(ToolExecutor[SearchCoreNodesAction, SearchCoreNodesObservation]):
    """Executor that searches the knowledge graph for Core nodes within N hops."""

    def __call__(
        self,
        action: SearchCoreNodesAction,
        conversation=None
    ) -> SearchCoreNodesObservation:
        """Execute the search_core_nodes action."""
        # Call the existing function from search_core.py
        result = get_core_nodes_within_hops(
            root_label=action.root_label,
            max_hops=action.max_hops
        )

        return SearchCoreNodesObservation(
            status=result.get("status", "fail"),
            detail=result.get("detail", "Unknown error"),
            results=result.get("results", [])
        )


# --- Tool Description ---
_SEARCH_CORE_NODES_DESCRIPTION = """Search knowledge graph for Core nodes within N hops.
* Finds papers (Core nodes) within a specified number of hops from a root node
* Supports both exact and fuzzy matching for the root node label
* Returns up to 100 papers sorted by hop distance (closest first)
* Each result includes: paper title, hop distance from root
* On fuzzy match failure, returns fail status with suggested query
* Useful for exploring related research papers in a knowledge graph
"""


# --- Tool Definition ---


class SearchCoreNodesTool(ToolDefinition[SearchCoreNodesAction, SearchCoreNodesObservation]):
    """A custom tool for searching Core nodes within N hops in a knowledge graph."""

    @classmethod
    def create(cls, conv_state) -> Sequence[ToolDefinition]:
        """Create SearchCoreNodesTool instance.

        Args:
            conv_state: Conversation state (not used but required by interface).

        Returns:
            A sequence containing a single SearchCoreNodesTool instance.
        """
        executor = SearchCoreNodesExecutor()

        return [
            cls(
                description=_SEARCH_CORE_NODES_DESCRIPTION,
                action_type=SearchCoreNodesAction,
                observation_type=SearchCoreNodesObservation,
                executor=executor,
            )
        ]


# --- Registration ---
def _make_search_core_nodes_tool(conv_state) -> list[ToolDefinition]:
    """Create the search core nodes tool."""
    return list(SearchCoreNodesTool.create(conv_state))


register_tool("SearchCoreNodesTool", _make_search_core_nodes_tool)
