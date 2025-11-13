"""
Research tools for paper and code search/download.

This module provides tools for:
- Searching papers on arXiv
- Downloading paper source files
- Searching GitHub repositories
- Cloning GitHub repositories
"""

import os
import re
import time
import tarfile
import urllib.parse
from typing import List, Dict, Optional

import feedparser
import requests


# =============================================================================
# ArXiv Paper Search and Download
# =============================================================================


def search_arxiv(query: str, max_results: int = 50) -> List[Dict]:
    """
    Search arxiv papers by query.

    Args:
        query: Search keyword or paper title
        max_results: Maximum number of results to return

    Returns:
        List of paper information dictionaries
    """
    # Temporarily disable proxy (feedparser doesn't support SOCKS5 proxy)
    old_http_proxy = os.environ.pop("http_proxy", None)
    old_https_proxy = os.environ.pop("https_proxy", None)
    old_HTTP_PROXY = os.environ.pop("HTTP_PROXY", None)
    old_HTTPS_PROXY = os.environ.pop("HTTPS_PROXY", None)

    try:
        # Build API URL
        base_url = "http://export.arxiv.org/api/query?"

        # Try exact title search first
        params = {
            "search_query": f"ti:{query}",  # Title search
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        # Build complete query URL
        query_url = base_url + urllib.parse.urlencode(params)

        # Use requests with timeout instead of feedparser.parse directly
        print(f"  Querying arXiv API (timeout: 30s)...")
        response = None

        # Add retry mechanism for API rate limiting
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Use requests to fetch with timeout
                http_response = requests.get(query_url, timeout=30)
                http_response.raise_for_status()

                # Parse the response with feedparser
                response = feedparser.parse(http_response.content)

                # If got results, break retry loop
                if len(response.entries) > 0:
                    print(f"  ✓ Received {len(response.entries)} results from arXiv")
                    break
                else:
                    print(f"  ⚠ No results found (attempt {retry + 1}/{max_retries})")

                # If no results and not last attempt, wait and retry
                if retry < max_retries - 1:
                    # Incremental wait time: 5s, 10s, 15s
                    wait_time = (retry + 1) * 5
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
            except requests.exceptions.Timeout:
                print(f"  ⚠ Request timeout (attempt {retry + 1}/{max_retries})")
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 5
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ All retry attempts failed due to timeout")
                    return []
            except requests.exceptions.RequestException as e:
                print(
                    f"  ⚠ Request error: {str(e)} (attempt {retry + 1}/{max_retries})"
                )
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 5
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ All retry attempts failed")
                    return []

        # If title search has no results, try full-text search
        if response is None or len(response.entries) == 0:
            print(f"  ⚠ No results from title search, trying simplified query...")
            # Try simplified title (remove subtitle)
            simplified_query = query.split(":")[0].strip()
            params["search_query"] = f"ti:{simplified_query}"
            query_url = base_url + urllib.parse.urlencode(params)

            try:
                print(f"  Trying simplified query: {simplified_query[:50]}...")
                http_response = requests.get(query_url, timeout=30)
                http_response.raise_for_status()
                response = feedparser.parse(http_response.content)
                time.sleep(1)
            except Exception as e:
                print(f"  ⚠ Error with simplified query: {str(e)}")
                response = feedparser.FeedParserDict()
                response.entries = []

            # If simplified title still has no results, try full-text search
            if len(response.entries) == 0:
                params["search_query"] = query  # Full-text search without ti: prefix
                query_url = base_url + urllib.parse.urlencode(params)
                try:
                    print(f"  Trying full-text search...")
                    http_response = requests.get(query_url, timeout=30)
                    http_response.raise_for_status()
                    response = feedparser.parse(http_response.content)
                    time.sleep(1)
                except Exception as e:
                    print(f"  ⚠ Error with full-text search: {str(e)}")
                    response = feedparser.FeedParserDict()
                    response.entries = []

        # Extract paper information
        papers = []
        for entry in response.entries:
            try:
                # Find PDF link, use None if not found
                pdf_url = next(
                    (
                        link.href
                        for link in entry.links
                        if link.type == "application/pdf"
                    ),
                    None,
                )

                paper = {
                    "title": entry.title,
                    "author": [author.name for author in entry.authors],
                    "published": entry.published,
                    "summary": entry.summary,
                    "url": entry.link,
                    "pdf_url": pdf_url,
                }
                papers.append(paper)

                # Respect API rate limit
                time.sleep(0.5)
            except Exception as e:
                # Skip if processing an entry fails
                print(f"Warning: Failed to process entry: {e}")
                continue

        # Title matching: try multiple matching strategies
        if papers:
            query_lower = query.lower().strip()
            query_normalized = " ".join(query_lower.split())  # Normalize whitespace

            # Store matching results
            exact_match = None
            exact_match_index = -1

            # Strategy 1: Exact match (word by word comparison)
            for i, paper in enumerate(papers):
                paper_title_lower = paper["title"].lower().strip()
                paper_title_normalized = " ".join(paper_title_lower.split())

                if paper_title_normalized == query_normalized:
                    exact_match = paper
                    exact_match_index = i
                    print(f"✓ Found exact match: {paper['title']}")
                    break

            # Strategy 2: If no exact match, try ignoring punctuation
            if not exact_match:
                import string

                query_no_punct = query_lower.translate(
                    str.maketrans("", "", string.punctuation)
                ).strip()
                query_no_punct = " ".join(query_no_punct.split())

                for i, paper in enumerate(papers):
                    paper_title_no_punct = (
                        paper["title"]
                        .lower()
                        .translate(str.maketrans("", "", string.punctuation))
                        .strip()
                    )
                    paper_title_no_punct = " ".join(paper_title_no_punct.split())

                    if paper_title_no_punct == query_no_punct:
                        exact_match = paper
                        exact_match_index = i
                        print(f"✓ Found match (ignoring punctuation): {paper['title']}")
                        break

            # Move matched paper to first position
            if exact_match and exact_match_index > 0:
                papers.pop(exact_match_index)
                papers.insert(0, exact_match)
            elif exact_match:
                print(f"✓ Best match already at position 0")
            else:
                print(f"⚠ No exact match found, using most relevant result")

        return papers

    finally:
        # Restore proxy settings
        if old_http_proxy is not None:
            os.environ["http_proxy"] = old_http_proxy
        if old_https_proxy is not None:
            os.environ["https_proxy"] = old_https_proxy
        if old_HTTP_PROXY is not None:
            os.environ["HTTP_PROXY"] = old_HTTP_PROXY
        if old_HTTPS_PROXY is not None:
            os.environ["HTTPS_PROXY"] = old_HTTPS_PROXY


def extract_tex_content(tar_path: str) -> str:
    """
    Extract all .tex file contents from tar.gz file.

    Args:
        tar_path: Path to tar.gz file

    Returns:
        Concatenated content of all .tex files, formatted as filename + content
    """
    try:
        all_content = []

        with tarfile.open(tar_path, "r:gz") as tar:
            # Get all .tex files
            tex_files = [f for f in tar.getmembers() if f.name.endswith(".tex")]

            for tex_file in tex_files:
                # Extract file content
                f = tar.extractfile(tex_file)
                if f is not None:
                    try:
                        # Try to decode as utf-8
                        content = f.read().decode("utf-8")
                    except UnicodeDecodeError:
                        # If utf-8 fails, try latin-1
                        f.seek(0)
                        content = f.read().decode("latin-1")

                    # Add filename and content
                    all_content.append(
                        f"\n{'='*50}\nFilename: {tex_file.name}\n{'='*50}\n"
                    )
                    all_content.append(content)
                    all_content.append("\n\n")

        # Concatenate all content into one string
        return "".join(all_content)

    except Exception as e:
        return f"Extract failed with error: {str(e)}"


def download_arxiv_pdf(arxiv_url: str, output_dir: str, title: str) -> Dict[str, any]:
    """
    Download arxiv paper PDF and convert to markdown/text.

    Args:
        arxiv_url: ArXiv paper URL, e.g. 'http://arxiv.org/abs/2006.11239v2'
        output_dir: Output directory for downloaded papers
        title: Paper title (used for filename)

    Returns:
        Dictionary with status, message, and path
    """
    try:
        # Extract paper ID from URL - try multiple patterns
        paper_id = None

        # Pattern 1: http://arxiv.org/abs/2006.11239v2 or https://arxiv.org/abs/2006.11239v2
        match = re.search(r"abs/([^/?#]+)", arxiv_url)
        if match:
            paper_id = match.group(1)

        # Pattern 2: http://arxiv.org/pdf/2006.11239v2.pdf
        if not paper_id:
            match = re.search(r"pdf/([^/?#]+)\.pdf", arxiv_url)
            if match:
                paper_id = match.group(1)

        # Pattern 3: Direct arxiv ID format (e.g., 2006.11239v2)
        if not paper_id:
            match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", arxiv_url)
            if match:
                paper_id = match.group(1)

        if not paper_id:
            return {
                "status": -1,
                "message": f"Could not extract paper ID from URL: {arxiv_url}",
                "path": None,
            }

        # Remove version suffix if present for PDF URL (use latest version)
        paper_id_base = paper_id.split("v")[0] if "v" in paper_id else paper_id

        # Build PDF URL
        pdf_url = f"http://arxiv.org/pdf/{paper_id_base}.pdf"

        # Download PDF
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url)

        # Check status code
        if response.status_code == 200:
            try:
                # Create directories
                paper_source_dir = os.path.join(output_dir, "paper_source")
                papers_dir = os.path.join(output_dir, "papers")
                os.makedirs(paper_source_dir, exist_ok=True)
                os.makedirs(papers_dir, exist_ok=True)

                # Save PDF file in paper_source
                safe_filename = title.replace(" ", "_").replace(":", "").lower()
                pdf_filepath = os.path.join(paper_source_dir, f"{safe_filename}.pdf")

                with open(pdf_filepath, "wb") as f:
                    f.write(response.content)

                print(f"PDF saved to: {pdf_filepath}")

                # Convert PDF to markdown using internal functions
                from src.agents.experiment_agent.tools.repository_tools import (
                    _extract_pdf_with_pypdf2,
                    _extract_pdf_with_docling,
                    PYPDF2_AVAILABLE,
                    DOCLING_AVAILABLE,
                )

                # Extract text from PDF
                text = ""
                if DOCLING_AVAILABLE:
                    text = _extract_pdf_with_docling(pdf_filepath)

                if not text and PYPDF2_AVAILABLE:
                    print(f"  Falling back to PyPDF2...")
                    text = _extract_pdf_with_pypdf2(pdf_filepath)

                if not text:
                    pdf_result = {
                        "success": False,
                        "error": "Failed to extract text from PDF",
                    }
                else:
                    # Remove content before introduction
                    intro_patterns = [
                        r"(?i)^1\.?\s*introduction",
                        r"(?i)^I\.?\s*introduction",
                        r"(?i)^introduction",
                    ]
                    lines = text.split("\n")
                    intro_idx = 0
                    for i, line in enumerate(lines):
                        if any(
                            re.match(pattern, line.strip())
                            for pattern in intro_patterns
                        ):
                            intro_idx = i
                            break

                    if intro_idx > 0:
                        text = "\n".join(lines[intro_idx:])

                    # Keep full text, no truncation
                    pdf_result = {
                        "success": True,
                        "text": text,
                        "text_length": len(text),
                        "extraction_method": (
                            "docling" if DOCLING_AVAILABLE else "pypdf2"
                        ),
                    }

                if pdf_result["success"]:
                    # Save markdown in papers directory for analysis
                    md_filepath = os.path.join(papers_dir, f"{safe_filename}.md")
                    with open(md_filepath, "w", encoding="utf-8") as f:
                        f.write(pdf_result["text"])

                    print(f"✓ Converted to markdown: {md_filepath}")
                    print(f"  Extracted {pdf_result['text_length']} chars")
                    print(f"  Method: {pdf_result['extraction_method']}")

                    return {
                        "status": 0,
                        "message": f"Downloaded and converted paper '{title}' successfully",
                        "path": md_filepath,
                        "pdf_path": pdf_filepath,
                        "text_length": pdf_result["text_length"],
                    }
                else:
                    # Even if conversion failed, we still have the PDF
                    print(
                        f"⚠ PDF downloaded but conversion failed: {pdf_result.get('error')}"
                    )
                    return {
                        "status": 0,
                        "message": f"Downloaded PDF '{title}' (conversion failed)",
                        "path": pdf_filepath,
                        "pdf_path": pdf_filepath,
                        "conversion_error": pdf_result.get("error"),
                    }

            except Exception as e:
                return {
                    "status": -1,
                    "message": f"Download paper '{title}' failed with error: {str(e)}",
                    "path": None,
                }
        else:
            return {
                "status": -1,
                "message": f"Download paper '{title}' failed with HTTP status code {response.status_code}",
                "path": None,
            }

    except Exception as e:
        return {
            "status": -1,
            "message": f"Download paper '{title}' failed with error: {str(e)}",
            "path": None,
        }


def download_arxiv_source(
    arxiv_url: str, output_dir: str, title: str
) -> Dict[str, any]:
    """
    Download arxiv paper (now downloads PDF instead of source).

    This function now calls download_arxiv_pdf for better compatibility.

    Args:
        arxiv_url: ArXiv paper URL, e.g. 'http://arxiv.org/abs/2006.11239v2'
        output_dir: Output directory for downloaded papers
        title: Paper title (used for filename)

    Returns:
        Dictionary with status, message, and path
    """
    return download_arxiv_pdf(arxiv_url, output_dir, title)


def download_papers_by_titles(
    paper_titles: List[str], output_dir: str
) -> List[Dict[str, any]]:
    """
    Download arxiv papers by titles.

    Args:
        paper_titles: List of paper titles to download
        output_dir: Output directory for downloaded papers

    Returns:
        List of download results (dictionaries with status, message, path)
    """
    results = []

    for title in paper_titles:
        print(f"\n{'='*60}")
        print(f"Searching for: {title}")
        print("=" * 60)

        # Search for paper (search_arxiv will automatically rank exact matches first)
        papers = search_arxiv(title, max_results=20)

        if len(papers) == 0:
            msg = f"❌ Cannot find the paper '{title}' in arxiv"
            print(msg)
            results.append({"status": -1, "message": msg, "path": None, "title": title})
            continue

        # Display search results
        print(f"Found {len(papers)} results from arxiv")
        if len(papers) > 1:
            print("Top 3 results:")
            for i, paper in enumerate(papers[:3]):
                print(f"  {i+1}. {paper['title']}")

        # Use first result (already sorted by search_arxiv, exact match is first)
        best_paper = papers[0]
        print(f"\n✓ Selected: {best_paper['title']}")

        # Download paper
        download_info = download_arxiv_source(best_paper["url"], output_dir, title)

        if download_info["status"] == 0:
            success_msg = (
                f"✓ Successfully downloaded: {title}\n"
                f"  Path: {download_info['path']}"
            )
            print(success_msg)

        results.append({**download_info, "title": title})

    return results


# =============================================================================
# GitHub Repository Search and Clone
# =============================================================================


def search_github_repos(query: str, limit: int = 5) -> List[Dict]:
    """
    Search GitHub public repositories based on a keyword.

    Args:
        query: The query to search for in repository names or descriptions
        limit: The total number of repositories to return

    Returns:
        List of dictionaries containing repository details
    """
    repos = []
    per_page = 10
    page = 1

    while len(repos) < limit:
        url = f"https://api.github.com/search/repositories?q={query}&per_page={per_page}&page={page}"

        response = requests.get(url)

        if response.status_code == 200:
            items = response.json().get("items", [])
            for item in items:
                formatted_repo = {
                    "name": f"{item['owner']['login']}/{item['name']}",
                    "author": item["owner"]["login"],
                    "description": item["description"],
                    "link": item["html_url"],
                    "clone_url": item["clone_url"],
                    "stars": item["stargazers_count"],
                    "language": item["language"],
                    "created_at": item["created_at"],
                }
                repos.append(formatted_repo)
                if len(repos) >= limit:
                    break

            if len(items) < per_page:  # Stop if there are no more repos to fetch
                break
            page += 1
        else:
            raise Exception(
                f"GitHub API request failed with status code {response.status_code}: {response.text}"
            )

    return repos


def clone_github_repo(clone_url: str, output_dir: str, repo_name: str) -> Dict:
    """
    Clone a GitHub repository to local directory.

    Args:
        clone_url: Git clone URL
        output_dir: Output directory for cloned repositories
        repo_name: Repository name (for directory naming)

    Returns:
        Dictionary with status and message
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Extract repository name from clone_url if not provided
        if not repo_name:
            repo_name = clone_url.split("/")[-1].replace(".git", "")

        # Full path for cloned repository
        repo_path = os.path.join(output_dir, repo_name)

        # Check if repository already exists
        if os.path.exists(repo_path):
            return {
                "status": 0,
                "message": f"Repository '{repo_name}' already exists at {repo_path}",
                "path": repo_path,
            }

        # Clone repository using git command
        import subprocess

        result = subprocess.run(
            ["git", "clone", clone_url, repo_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            return {
                "status": 0,
                "message": f"Successfully cloned repository '{repo_name}'",
                "path": repo_path,
            }
        else:
            return {
                "status": -1,
                "message": f"Failed to clone repository '{repo_name}': {result.stderr}",
                "path": None,
            }

    except subprocess.TimeoutExpired:
        return {
            "status": -1,
            "message": f"Clone operation timed out for repository '{repo_name}'",
            "path": None,
        }
    except Exception as e:
        return {
            "status": -1,
            "message": f"Failed to clone repository '{repo_name}': {str(e)}",
            "path": None,
        }


def extract_github_links_from_text(text: str) -> List[str]:
    """
    Extract GitHub repository URLs from text content.

    Args:
        text: Text content to search (e.g., paper content)

    Returns:
        List of GitHub repository URLs found in the text
    """
    github_patterns = [
        r"https?://github\.com/[\w\-\.]+/[\w\-\.]+",
        r"github\.com/[\w\-\.]+/[\w\-\.]+",
        r"https?://www\.github\.com/[\w\-\.]+/[\w\-\.]+",
    ]

    found_links = set()

    for pattern in github_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Normalize the URL
            if not match.startswith("http"):
                match = "https://" + match
            # Remove trailing slashes and fragments
            match = match.rstrip("/").split("#")[0].split("?")[0]
            found_links.add(match)

    return list(found_links)


def search_and_clone_repos_for_papers(
    paper_titles: List[str], output_dir: str, max_repos_per_paper: int = 3
) -> List[Dict]:
    """
    Search and clone GitHub repositories related to paper titles.

    Args:
        paper_titles: List of paper titles
        output_dir: Output directory for cloned repositories
        max_repos_per_paper: Maximum number of repositories to clone per paper

    Returns:
        List of clone results
    """
    results = []

    for title in paper_titles:
        print(f"\n{'='*60}")
        print(f"Searching GitHub repositories for: {title}")
        print("=" * 60)

        # Search GitHub with paper title
        try:
            # Add "-user:lucidrains" to exclude lucidrains repositories
            query = f"{title} -user:lucidrains"
            repos = search_github_repos(query, limit=max_repos_per_paper * 2)

            if len(repos) == 0:
                msg = f"❌ No GitHub repositories found for '{title}'"
                print(msg)
                results.append(
                    {
                        "status": -1,
                        "message": msg,
                        "path": None,
                        "paper_title": title,
                    }
                )
                continue

            # Display found repositories
            print(f"Found {len(repos)} repositories")
            for i, repo in enumerate(repos[:max_repos_per_paper]):
                print(
                    f"  {i+1}. {repo['name']} - {repo['stars']} stars - {repo['description']}"
                )

            # Clone top repositories
            cloned_count = 0
            for repo in repos[:max_repos_per_paper]:
                if cloned_count >= max_repos_per_paper:
                    break

                print(f"\nCloning {repo['name']}...")
                clone_result = clone_github_repo(
                    repo["clone_url"],
                    output_dir,
                    repo["name"].replace("/", "_"),
                )

                results.append(
                    {
                        **clone_result,
                        "paper_title": title,
                        "repo_name": repo["name"],
                        "repo_url": repo["link"],
                    }
                )

                if clone_result["status"] == 0:
                    print(f"✓ {clone_result['message']}")
                    cloned_count += 1
                else:
                    print(f"✗ {clone_result['message']}")

                time.sleep(1)  # Rate limiting

        except Exception as e:
            msg = f"Failed to search/clone repositories for '{title}': {str(e)}"
            print(f"❌ {msg}")
            results.append(
                {"status": -1, "message": msg, "path": None, "paper_title": title}
            )

    return results
