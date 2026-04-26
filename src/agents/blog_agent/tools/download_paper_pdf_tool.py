"""Download Paper PDF Tool - OpenHands Format"""

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

# Import the download function
from blog_agent.utils.semantic_scholar import download_paper_pdf


# --- Action / Observation ---


class DownloadPaperPdfAction(Action):
    """Download a paper's PDF from Semantic Scholar."""

    query: str = Field(
        description="Search query - paper title, keywords, etc. to find the paper"
    )
    output_dir: str = Field(
        default="",
        description="Directory to save the PDF file (defaults to current directory)"
    )


class DownloadPaperPdfObservation(Observation):
    """Result of downloading paper PDF from Semantic Scholar."""

    status: str = Field(
        description="Status: 'success' or 'fail'"
    )
    detail: str = Field(
        description="Detail: failure reason or success message"
    )
    pdf_path: str = Field(
        default="",
        description="Path to downloaded PDF file (only on success)"
    )
    paper_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Paper metadata: title, year, authors, venue, paper_id, url, pdf_url"
    )

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if self.status == "fail":
            return [TextContent(text=f"Download failed: {self.detail}")]

        # Success case
        p = self.paper_info
        lines = [
            f"PDF downloaded successfully!",
            f"Path: {self.pdf_path}",
            "",
            f"Title: {p.get('title', 'N/A')}",
            f"Year: {p.get('year', 'N/A')}",
            f"Venue: {p.get('venue', 'N/A')}",
            f"Authors: {', '.join(p.get('authors', []))}",
            f"URL: {p.get('url', 'N/A')}",
            f"PDF URL: {p.get('pdf_url', 'N/A')}",
        ]

        return [TextContent(text="\n".join(lines))]


# --- Executor ---


class DownloadPaperPdfExecutor(ToolExecutor[DownloadPaperPdfAction, DownloadPaperPdfObservation]):
    """Executor that downloads paper PDFs from Semantic Scholar."""

    def __call__(
        self,
        action: DownloadPaperPdfAction,
        conversation=None
    ) -> DownloadPaperPdfObservation:
        """Execute the download_paper_pdf action."""
        output_dir = action.output_dir if action.output_dir else None
        result = download_paper_pdf(
            query=action.query,
            output_dir=output_dir,
        )

        return DownloadPaperPdfObservation(
            status=result.get("status", "fail"),
            detail=result.get("detail", "Unknown error"),
            pdf_path=result.get("pdf_path", ""),
            paper_info=result.get("paper_info", {})
        )


# --- Tool Description ---
_DOWNLOAD_PAPER_PDF_DESCRIPTION = """Download a paper's PDF from Semantic Scholar.
* Search for a paper by title/keywords and download its PDF if available
* Returns: PDF file path and paper metadata (title, authors, year, venue, etc.)
* Note: PDF may not be available for all papers (depends on open access status)
* Use this when you need the full paper content for detailed reading or citing
"""


# --- Tool Definition ---


class DownloadPaperPdfTool(ToolDefinition[DownloadPaperPdfAction, DownloadPaperPdfObservation]):
    """A custom tool for downloading paper PDFs from Semantic Scholar."""

    @classmethod
    def create(cls, conv_state) -> Sequence[ToolDefinition]:
        """Create DownloadPaperPdfTool instance.

        Args:
            conv_state: Conversation state (not used but required by interface).

        Returns:
            A sequence containing a single DownloadPaperPdfTool instance.
        """
        executor = DownloadPaperPdfExecutor()

        return [
            cls(
                description=_DOWNLOAD_PAPER_PDF_DESCRIPTION,
                action_type=DownloadPaperPdfAction,
                observation_type=DownloadPaperPdfObservation,
                executor=executor,
            )
        ]


# --- Registration ---
def _make_download_paper_pdf_tool(conv_state) -> list[ToolDefinition]:
    """Create the download paper PDF tool."""
    return list(DownloadPaperPdfTool.create(conv_state))


register_tool("DownloadPaperPdfTool", _make_download_paper_pdf_tool)
