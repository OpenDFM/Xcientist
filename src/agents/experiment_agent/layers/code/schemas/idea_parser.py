import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .proposal import Proposal, Idea


def parse_idea_markdown(content: str) -> Proposal:
    """
    Parse an idea.md file content into a Proposal object.

    Args:
        content: The markdown content

    Returns:
        Proposal object
    """
    parser = IdeaMarkdownParser(content)
    return parser.parse()


def load_idea_file(file_path: str) -> Proposal:
    """
    Load an idea file (supports .md and .json formats).

    Args:
        file_path: Path to the idea file

    Returns:
        Proposal object
    """
    path = Path(file_path)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if path.suffix.lower() == ".json":
        # JSON format (original)
        return Proposal.model_validate_json(content)
    elif path.suffix.lower() in [".md", ".markdown"]:
        # Markdown format (new)
        return parse_idea_markdown(content)
    else:
        # Try to auto-detect
        content_stripped = content.strip()
        if content_stripped.startswith("{"):
            return Proposal.model_validate_json(content)
        else:
            return parse_idea_markdown(content)


class IdeaMarkdownParser:
    """Parser for idea.md format."""

    def __init__(self, content: str):
        self.content = content
        self.lines = content.split("\n")
        self.sections: Dict[str, str] = {}

    def parse(self) -> Proposal:
        """Parse the markdown content into a Proposal."""
        # Extract sections
        self._extract_sections()

        # Build Idea object
        title = self._extract_title()
        description = self._get_section_content(
            ["conceptual framework", "overview", "description", "summary"]
        )
        key_innovations = self._extract_innovations()
        methodology = self._extract_methodology()
        expected_outcomes = self._extract_outcomes()
        reference_papers = self._extract_references()

        idea = Idea(
            title=title,
            description=description,
            key_innovations=key_innovations,
            methodology=methodology,
            expected_outcomes=expected_outcomes,
        )

        return Proposal(
            idea=idea,
            reference_papers=reference_papers,
        )

    def _extract_sections(self):
        """Extract all sections from the markdown."""
        current_section = "header"
        current_content = []

        for line in self.lines:
            # Check for section headers (## or ###)
            if line.startswith("## ") or line.startswith("# "):
                # Save previous section
                if current_content:
                    self.sections[current_section.lower()] = "\n".join(
                        current_content
                    ).strip()

                # Start new section
                current_section = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            self.sections[current_section.lower()] = "\n".join(current_content).strip()

    def _extract_title(self) -> str:
        """Extract the title from the header."""
        # Look for first # heading
        for line in self.lines:
            if line.startswith("# ") and not line.startswith("## "):
                return line.lstrip("#").strip()

        # Or use header section
        header = self.sections.get("header", "")
        for line in header.split("\n"):
            line = line.strip()
            if line and not line.startswith("-"):
                return line

        return "Untitled Research Proposal"

    def _get_section_content(self, section_names: List[str]) -> str:
        """Get content from first matching section."""
        for name in section_names:
            for section_key, content in self.sections.items():
                if name in section_key.lower():
                    return content
        return ""

    def _extract_innovations(self) -> List[str]:
        """Extract key innovations as a list."""
        innovations = []

        # Find innovations section
        content = self._get_section_content(
            ["key innovations", "innovations", "contributions"]
        )

        if not content:
            return innovations

        # Parse numbered items or ### headings
        current_innovation = []

        for line in content.split("\n"):
            line = line.strip()

            # Check for numbered item or ### heading
            is_new_item = (
                re.match(r"^\d+\.", line)
                or re.match(r"^###\s+\d+\.", line)
                or line.startswith("### ")
                or line.startswith("- **")
            )

            if is_new_item:
                # Save previous innovation
                if current_innovation:
                    innovations.append(" ".join(current_innovation))

                # Start new innovation
                # Clean the line
                clean = re.sub(r"^(###\s+)?\d+\.\s*", "", line)
                clean = clean.lstrip("- ").lstrip("**").rstrip("**")
                current_innovation = [clean] if clean else []
            elif line and current_innovation:
                # Continue current innovation
                current_innovation.append(line.lstrip("- "))

        # Save last innovation
        if current_innovation:
            innovations.append(" ".join(current_innovation))

        return innovations

    def _extract_methodology(self) -> Dict[str, str]:
        """Extract methodology sections."""
        methodology = {}

        # Look for various methodology-related sections
        section_mappings = {
            "system architecture": "system_architecture",
            "algorithms": "algorithms",
            "mathematical formulations": "mathematical_formulations",
            "technical specifications": "technical_specifications",
            "implementation guidance": "implementation_guidance",
        }

        for section_name, key in section_mappings.items():
            content = self._get_section_content([section_name])
            if content:
                methodology[key] = content

        return methodology

    def _extract_outcomes(self) -> List[str]:
        """Extract expected outcomes."""
        outcomes = []

        # Look for outcomes/summary section
        content = self._get_section_content(
            ["expected outcomes", "outcomes", "summary"]
        )

        if not content:
            return outcomes

        # Parse bullet points
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                outcomes.append(line[2:])
            elif line.startswith("* "):
                outcomes.append(line[2:])

        # If no bullets, split by sentences
        if not outcomes and content:
            sentences = content.split(". ")
            outcomes = [s.strip() + "." for s in sentences if len(s) > 20]

        return outcomes[:5]  # Limit to 5

    def _extract_references(self) -> List[str]:
        """Extract reference papers/repositories."""
        references = []

        # Look for code repositories section (table format)
        content = self._get_section_content(
            ["code repositories", "repositories", "references"]
        )

        if not content:
            return references

        # Parse table format
        # | Repository | Purpose |
        for line in content.split("\n"):
            line = line.strip()

            # Skip table headers and separators
            if line.startswith("|") and "---" not in line and "Repository" not in line:
                parts = [p.strip() for p in line.split("|")]
                parts = [p for p in parts if p]  # Remove empty

                if len(parts) >= 2:
                    repo_name = parts[0].strip("*")
                    purpose = parts[1]
                    references.append(f"{repo_name}: {purpose}")

        # Also look for bullet points
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                ref = line.lstrip("- ").lstrip("* ")
                if ref and ref not in references:
                    references.append(ref)

        return references


def validate_idea_markdown(content: str) -> Tuple[bool, List[str]]:
    """
    Validate an idea.md file and return any issues.

    Args:
        content: The markdown content

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check for title
    has_title = any(line.startswith("# ") for line in content.split("\n"))
    if not has_title:
        issues.append("Missing main title (# Title)")

    # Check for key sections
    required_sections = ["key innovations", "implementation"]
    content_lower = content.lower()

    for section in required_sections:
        if section not in content_lower:
            issues.append(f"Missing recommended section: {section}")

    # Check for innovations
    parser = IdeaMarkdownParser(content)
    parser._extract_sections()
    innovations = parser._extract_innovations()

    if len(innovations) < 2:
        issues.append(
            f"Only {len(innovations)} innovations found, recommend at least 3"
        )

    return len(issues) == 0, issues
