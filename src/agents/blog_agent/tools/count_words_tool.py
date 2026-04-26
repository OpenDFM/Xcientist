"""Word Count Tool - OpenHands Format"""

import os
import sys
import re
import logging
from typing import List, Dict, Any, Sequence
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


# --- Action / Observation ---


class CountWordsAction(Action):
    """Count words in a markdown file."""

    file_path: str = Field(
        description="Path to the markdown file to count words in"
    )


class CountWordsObservation(Observation):
    """Result of counting words."""

    word_count: int = Field(description="Number of words in the file")
    file_path: str = Field(description="Path to the file that was counted")


# --- Executor ---


class CountWordsExecutor(ToolExecutor[CountWordsAction, CountWordsObservation]):
    """Executor for counting words in markdown files."""

    def __call__(
        self,
        action: CountWordsAction,
        conversation=None
    ) -> CountWordsObservation:
        """Execute word counting."""
        try:
            # Try to import markdown, fallback to simple regex if not available
            try:
                import markdown
                from bs4 import BeautifulSoup

                with open(action.file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                html = markdown.markdown(text)
                soup = BeautifulSoup(html, 'html.parser')
                clean_text = soup.get_text()

                words = re.findall(r'\b[a-zA-Z]+\b', clean_text)
                word_count = len(words)
            except ImportError:
                # Fallback: simple regex without markdown parsing
                with open(action.file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Remove code blocks
                text = re.sub(r'```[\s\S]*?```', '', text)
                # Remove inline code
                text = re.sub(r'`[^`]+`', '', text)
                # Remove markdown links
                text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
                # Remove markdown images
                text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
                # Remove headers markers
                text = re.sub(r'#+ ', '', text)
                # Remove bold/italic
                text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)

                words = re.findall(r'\b[a-zA-Z]+\b', text)
                word_count = len(words)

            return CountWordsObservation(
                content=[
                    TextContent(
                        type="text",
                        text=f"Word count: {word_count} in {action.file_path}"
                    )
                ],
                word_count=word_count,
                file_path=action.file_path
            )

        except FileNotFoundError:
            return CountWordsObservation(
                content=[
                    TextContent(
                        type="text",
                        text=f"File not found: {action.file_path}"
                    )
                ],
                word_count=0,
                file_path=action.file_path
            )
        except Exception as e:
            return CountWordsObservation(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error counting words: {str(e)}"
                    )
                ],
                word_count=0,
                file_path=action.file_path
            )


# --- Tool Definition ---

_COUNT_WORDS_DESCRIPTION = "Count the number of words in a markdown file. Returns the word count (English words only, excluding code blocks, links, and markdown formatting)."


class CountWordsTool(ToolDefinition[CountWordsAction, CountWordsObservation]):
    """A tool for counting words in markdown files."""

    @classmethod
    def create(cls, conv_state) -> Sequence[ToolDefinition]:
        """Create CountWordsTool instance."""
        executor = CountWordsExecutor()

        return [
            cls(
                description=_COUNT_WORDS_DESCRIPTION,
                action_type=CountWordsAction,
                observation_type=CountWordsObservation,
                executor=executor,
            )
        ]


# --- Registration ---
def _make_count_words_tool(conv_state) -> list[ToolDefinition]:
    """Create the count words tool."""
    return list(CountWordsTool.create(conv_state))


register_tool("CountWords", _make_count_words_tool)
