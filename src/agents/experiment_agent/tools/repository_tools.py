"""
Repository analysis tools for experiment agents.

Provides tools for analyzing code repositories, generating code trees,
and extracting important code structures.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from agents import function_tool

try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from docling.document_converter import DocumentConverter

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@function_tool
def list_papers_in_directory(
    directory: str,
    file_extensions: List[str] = None,
) -> Dict[str, Any]:
    """
    List all paper files in a directory.

    Args:
        directory: Directory path to search for papers
        file_extensions: List of file extensions to include (default: [".tex", ".pdf", ".txt"])

    Returns:
        Dictionary with list of papers and their metadata
    """
    try:
        directory = os.path.expanduser(directory)

        if not os.path.exists(directory):
            return {
                "success": False,
                "error": f"Directory not found: {directory}",
            }

        if file_extensions is None:
            file_extensions = [".tex", ".pdf", ".txt", ".md"]

        papers = []
        path = Path(directory)

        for file_path in path.iterdir():
            if file_path.is_file() and file_path.suffix in file_extensions:
                papers.append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "extension": file_path.suffix,
                        "size": file_path.stat().st_size,
                        "size_kb": round(file_path.stat().st_size / 1024, 2),
                    }
                )

        # Sort by name
        papers.sort(key=lambda x: x["name"])

        return {
            "success": True,
            "directory": directory,
            "papers": papers,
            "total_count": len(papers),
            "extensions_found": list(set(p["extension"] for p in papers)),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error listing papers: {str(e)}",
        }



def _extract_pdf_with_pypdf2(pdf_path: str, max_chars: int = 50000) -> str:
    """
    Extract text from PDF using PyPDF2 (fallback method).

    Args:
        pdf_path: Path to PDF file
        max_chars: Maximum characters to extract

    Returns:
        Extracted text
    """
    if not PYPDF2_AVAILABLE:
        return ""

    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                if len(text) >= max_chars:
                    break
                text += page.extract_text() + "\n"
            return text[:max_chars]
    except Exception as e:
        print(f"PyPDF2 extraction failed: {e}")
        return ""


def _extract_pdf_with_docling(pdf_path: str, timeout: int = 60) -> str:
    """
    Extract text from PDF using Docling (preferred method).

    Args:
        pdf_path: Path to PDF file
        timeout: Timeout in seconds

    Returns:
        Extracted text in markdown format
    """
    if not DOCLING_AVAILABLE:
        return ""

    try:
        # Check if corresponding .md file exists
        md_path = pdf_path.rsplit(".", 1)[0] + ".md"
        if os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    print(f"Using cached markdown: {os.path.basename(pdf_path)}")
                    return content

        # Convert using Docling
        print(f"Converting PDF with Docling: {os.path.basename(pdf_path)}")
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        text = result.document.export_to_markdown()

        # Save to cache
        if text and text.strip():
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Successfully extracted: {os.path.basename(pdf_path)}")
            return text

        return ""

    except Exception as e:
        print(f"Docling extraction failed: {e}")
        return ""


def _truncate_text(text: str, max_tokens: int = 15000, model: str = "gpt-4") -> str:
    """
    Truncate text to fit within token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model: Model name for tokenization

    Returns:
        Truncated text
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback: roughly 4 characters per token
        max_chars = max_tokens * 4
        return text[:max_chars]

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return text


@function_tool
def read_pdf_paper(
    pdf_path: str,
    max_tokens: int = 0,  # 0 means no truncation
    use_docling: bool = True,
) -> Dict[str, Any]:
    """
    Read and extract text from a PDF paper file.

    Args:
        pdf_path: Path to the PDF file
        max_tokens: Maximum tokens to extract (default: 15000)
        use_docling: Whether to use Docling for extraction (default: True)

    Returns:
        Dictionary with extracted text and metadata
    """
    try:
        pdf_path = os.path.expanduser(pdf_path)

        if not os.path.exists(pdf_path):
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}",
            }

        # Try Docling first if available
        text = ""
        if use_docling and DOCLING_AVAILABLE:
            text = _extract_pdf_with_docling(pdf_path)

        # Fallback to PyPDF2
        if not text and PYPDF2_AVAILABLE:
            print(f"Falling back to PyPDF2 for: {os.path.basename(pdf_path)}")
            text = _extract_pdf_with_pypdf2(pdf_path)

        if not text:
            return {
                "success": False,
                "error": "Failed to extract text from PDF. PyPDF2 or Docling required.",
            }

        # Remove content before introduction
        intro_patterns = [
            r"(?i)^1\.?\s*introduction",
            r"(?i)^I\.?\s*introduction",
            r"(?i)^introduction",
        ]
        lines = text.split("\n")
        intro_idx = 0
        for i, line in enumerate(lines):
            if any(re.match(pattern, line.strip()) for pattern in intro_patterns):
                intro_idx = i
                break

        if intro_idx > 0:
            text = "\n".join(lines[intro_idx:])

        # Truncate to token limit (if max_tokens > 0)
        if max_tokens > 0:
            text = _truncate_text(text, max_tokens)

        # Get file stats
        file_stats = os.stat(pdf_path)

        return {
            "success": True,
            "pdf_path": pdf_path,
            "filename": os.path.basename(pdf_path),
            "text": text,
            "text_length": len(text),
            "estimated_tokens": len(text) // 4,  # Rough estimate
            "file_size": file_stats.st_size,
            "extraction_method": (
                "docling" if use_docling and DOCLING_AVAILABLE else "pypdf2"
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading PDF: {str(e)}",
        }
