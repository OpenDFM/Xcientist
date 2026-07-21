"""
Semantic Scholar API Tool - Search papers and retrieve full abstracts
"""

import os
import sys
import requests
import time
from typing import Optional, Dict, Any

# Add path
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if agents_root not in sys.path:
    sys.path.insert(0, agents_root)

from src.config import load_config


def search_paper_and_get_abstract(
    query: str,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Search Semantic Scholar by query (title, keywords, etc.) and retrieve full abstract

    Args:
        query: Search query - can be paper title, keywords, or any text
        api_key: Semantic Scholar API key (optional, defaults to config)
        max_retries: Maximum number of retries

    Returns:
        Dict with:
        - status: "success" or "fail"
        - detail: str - failure reason or paper info
        - paper: dict - paper details (only on success)
          - title: Paper title
          - abstract: Full abstract
          - year: Publication year
          - authors: List of authors
          - venue: Conference/journal
          - paper_id: Semantic Scholar paper ID
          - url: Paper URL
    """
    # Get API key from config if not provided
    if not api_key:
        config = load_config().get("blog", {})
        api_key = config.get("semantic_scholar", {}).get("api_key", "")

    if not api_key:
        return {
            "status": "fail",
            "detail": "Semantic Scholar API key not found. Please configure it in config.",
            "paper": None
        }

    base_url = "http://api.semanticscholar.org/graph/v1"
    headers = {"x-api-key": api_key}

    # 1. Search for the paper first
    search_url = f"{base_url}/paper/search"
    search_params = {
        "query": query,
        "fields": "paperId,title,year,abstract,authors,venue,url",
        "limit": 5,
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                search_url,
                headers=headers,
                params=search_params,
                timeout=30
            )
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"Request failed: {str(e)}",
                "paper": None
            }

        if response.status_code == 200:
            break
        elif response.status_code == 429:
            # Rate limit
            time.sleep(60)
            continue
        else:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"Search failed with status {response.status_code}",
                "paper": None
            }

    if response.status_code != 200:
        return {
            "status": "fail",
            "detail": f"Search failed: {response.status_code}",
            "paper": None
        }

    search_result = response.json()
    papers = search_result.get("data", [])

    if not papers:
        return {
            "status": "fail",
            "detail": f"No papers found for query: {query}",
            "paper": None
        }

    # 2. Take the first (best match) result
    best_paper = papers[0]

    # Check if abstract is available
    abstract = best_paper.get("abstract")
    if not abstract:
        return {
            "status": "fail",
            "detail": f"Paper found but no abstract available: {best_paper.get('title')}",
            "paper": {
                "title": best_paper.get("title"),
                "abstract": None,
                "year": best_paper.get("year"),
                "authors": [a.get("name") for a in best_paper.get("authors", [])],
                "venue": best_paper.get("venue"),
                "paper_id": best_paper.get("paperId"),
                "url": best_paper.get("url"),
            }
        }

    return {
        "status": "success",
        "detail": f"Found paper: {best_paper.get('title')}",
        "paper": {
            "title": best_paper.get("title"),
            "abstract": abstract,
            "year": best_paper.get("year"),
            "authors": [a.get("name") for a in best_paper.get("authors", [])],
            "venue": best_paper.get("venue"),
            "paper_id": best_paper.get("paperId"),
            "url": best_paper.get("url"),
        }
    }


def get_paper_abstract_by_id(
    paper_id: str,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Retrieve paper abstract by Semantic Scholar paper ID

    Args:
        paper_id: Semantic Scholar paper ID (e.g., "ARXIV:2505.11711" or "paperId")
        api_key: Semantic Scholar API key
        max_retries: Maximum number of retries

    Returns:
        Dict with status, detail, paper (same as above)
    """
    if not api_key:
        config = load_config().get("blog", {})
        api_key = config.get("semantic_scholar", {}).get("api_key", "")

    if not api_key:
        return {
            "status": "fail",
            "detail": "Semantic Scholar API key not found",
            "paper": None
        }

    base_url = "http://api.semanticscholar.org/graph/v1"
    headers = {"x-api-key": api_key}

    # Normalize paper_id
    pid = str(paper_id).strip()
    if pid.lower().startswith("arxiv:"):
        pid = f"ARXIV:{pid.split(':', 1)[1].strip()}"
    elif "." in pid and pid.count(".") == 1:
        left, right = pid.split(".", 1)
        if left.isdigit() and right.isdigit():
            pid = f"ARXIV:{pid}"

    url = f"{base_url}/paper/{pid}"
    params = {
        "fields": "title,abstract,year,authors,venue,url"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"Request failed: {str(e)}",
                "paper": None
            }

        if response.status_code == 200:
            break
        elif response.status_code == 429:
            time.sleep(60)
            continue
        else:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"Request failed with status {response.status_code}",
                "paper": None
            }

    if response.status_code != 200:
        return {
            "status": "fail",
            "detail": f"Failed to get paper: {response.status_code}",
            "paper": None
        }

    paper = response.json()

    if not paper.get("abstract"):
        return {
            "status": "fail",
            "detail": "Paper found but no abstract available",
            "paper": {
                "title": paper.get("title"),
                "abstract": None,
                "year": paper.get("year"),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "venue": paper.get("venue"),
                "paper_id": paper.get("paperId"),
                "url": paper.get("url"),
            }
        }

    return {
        "status": "success",
        "detail": f"Found paper: {paper.get('title')}",
        "paper": {
            "title": paper.get("title"),
            "abstract": paper.get("abstract"),
            "year": paper.get("year"),
            "authors": [a.get("name") for a in paper.get("authors", [])],
            "venue": paper.get("venue"),
            "paper_id": paper.get("paperId"),
            "url": paper.get("url"),
        }
    }


def download_paper_pdf(
    query: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Search for a paper and download its PDF if available

    Args:
        query: Search query - paper title, keywords, etc.
        output_dir: Directory to save PDF (defaults to current dir)
        api_key: Semantic Scholar API key
        max_retries: Maximum number of retries

    Returns:
        Dict with:
        - status: "success" or "fail"
        - detail: str - failure reason or success message
        - pdf_path: str - path to downloaded PDF (only on success)
        - paper_info: dict - paper metadata (only on success)
    """
    if not api_key:
        config = load_config().get("blog", {})
        api_key = config.get("semantic_scholar", {}).get("api_key", "")

    if not api_key:
        return {
            "status": "fail",
            "detail": "Semantic Scholar API key not found",
            "pdf_path": "",
            "paper_info": {}
        }

    base_url = "http://api.semanticscholar.org/graph/v1"
    headers = {"x-api-key": api_key}

    # 1. Search for the paper with openAccessPdf and externalIds fields
    search_url = f"{base_url}/paper/search"
    search_params = {
        "query": query,
        "fields": "paperId,title,year,abstract,authors,venue,url,openAccessPdf,externalIds",
        "limit": 5,
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                search_url,
                headers=headers,
                params=search_params,
                timeout=30
            )
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"Search request failed: {str(e)}",
                "pdf_path": "",
                "paper_info": {}
            }

        if response.status_code == 200:
            break
        elif response.status_code == 429:
            time.sleep(60)
            continue
        else:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"Search failed with status {response.status_code}",
                "pdf_path": "",
                "paper_info": {}
            }

    if response.status_code != 200:
        return {
            "status": "fail",
            "detail": f"Search failed: {response.status_code}",
            "pdf_path": "",
            "paper_info": {}
        }

    search_result = response.json()
    papers = search_result.get("data", [])

    if not papers:
        return {
            "status": "fail",
            "detail": f"No papers found for query: {query}",
            "pdf_path": "",
            "paper_info": {}
        }

    best_paper = papers[0]

    # 2. Get PDF URL - try multiple sources
    pdf_url = best_paper.get("openAccessPdf", {}).get("url")

    # If no open access PDF, try arXiv
    if not pdf_url:
        external_ids = best_paper.get("externalIds", {})
        arxiv_id = external_ids.get("ArXiv")
        if arxiv_id:
            # Headers to avoid anti-bot
            test_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            # Try arXiv URLs
            for arxiv_url in [
                f"https://export.arxiv.org/pdf/{arxiv_id}.pdf",
                f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            ]:
                try:
                    test_resp = requests.head(arxiv_url, headers=test_headers, timeout=10, allow_redirects=True)
                    if test_resp.status_code == 200:
                        pdf_url = arxiv_url
                        break
                except:
                    continue

    if not pdf_url:
        return {
            "status": "fail",
            "detail": f"Paper found but no PDF available: {best_paper.get('title')}",
            "pdf_path": "",
            "paper_info": {
                "title": best_paper.get("title"),
                "year": best_paper.get("year"),
                "authors": [a.get("name") for a in best_paper.get("authors", [])],
                "venue": best_paper.get("venue"),
                "paper_id": best_paper.get("paperId"),
                "url": best_paper.get("url"),
            }
        }

    # 3. Download PDF
    if not output_dir:
        output_dir = os.getcwd()

    # Create safe filename
    safe_title = "".join(c for c in best_paper.get("title", "paper") if c.isalnum() or c in " -_").strip()[:50]
    filename = f"{safe_title}.pdf"
    pdf_path = os.path.join(output_dir, filename)

    # Add headers to avoid anti-bot (especially for arXiv)
    pdf_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(max_retries):
        try:
            pdf_response = requests.get(pdf_url, headers=pdf_headers, timeout=60)
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"PDF download failed: {str(e)}",
                "pdf_path": "",
                "paper_info": {}
            }

        if pdf_response.status_code == 200:
            break
        else:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"PDF download failed with status {pdf_response.status_code}",
                "pdf_path": "",
                "paper_info": {}
            }

    if pdf_response.status_code != 200:
        return {
            "status": "fail",
            "detail": f"PDF download failed: {pdf_response.status_code}",
            "pdf_path": "",
            "paper_info": {}
        }

    # Save PDF
    with open(pdf_path, "wb") as f:
        f.write(pdf_response.content)

    return {
        "status": "success",
        "detail": f"PDF downloaded: {best_paper.get('title')}",
        "pdf_path": pdf_path,
        "paper_info": {
            "title": best_paper.get("title"),
            "year": best_paper.get("year"),
            "authors": [a.get("name") for a in best_paper.get("authors", [])],
            "venue": best_paper.get("venue"),
            "paper_id": best_paper.get("paperId"),
            "url": best_paper.get("url"),
            "pdf_url": pdf_url,
        }
    }


def download_paper_pdf_by_id(
    paper_id: str,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Download paper PDF by Semantic Scholar paper ID

    Args:
        paper_id: Semantic Scholar paper ID (e.g., "ARXIV:2505.11711" or paperId)
        output_dir: Directory to save PDF
        api_key: Semantic Scholar API key
        max_retries: Maximum number of retries

    Returns:
        Dict with status, detail, pdf_path, paper_info
    """
    if not api_key:
        config = load_config().get("blog", {})
        api_key = config.get("semantic_scholar", {}).get("api_key", "")

    if not api_key:
        return {
            "status": "fail",
            "detail": "Semantic Scholar API key not found",
            "pdf_path": "",
            "paper_info": {}
        }

    base_url = "http://api.semanticscholar.org/graph/v1"
    headers = {"x-api-key": api_key}

    # Normalize paper_id
    pid = str(paper_id).strip()
    if pid.lower().startswith("arxiv:"):
        pid = f"ARXIV:{pid.split(':', 1)[1].strip()}"
    elif "." in pid and pid.count(".") == 1:
        left, right = pid.split(".", 1)
        if left.isdigit() and right.isdigit():
            pid = f"ARXIV:{pid}"

    # Get paper details including PDF URL
    url = f"{base_url}/paper/{pid}"
    params = {
        "fields": "title,year,authors,venue,url,openAccessPdf,externalIds"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"Request failed: {str(e)}",
                "pdf_path": "",
                "paper_info": {}
            }

        if response.status_code == 200:
            break
        elif response.status_code == 429:
            time.sleep(60)
            continue
        else:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"Request failed with status {response.status_code}",
                "pdf_path": "",
                "paper_info": {}
            }

    if response.status_code != 200:
        return {
            "status": "fail",
            "detail": f"Failed to get paper: {response.status_code}",
            "pdf_path": "",
            "paper_info": {}
        }

    paper = response.json()

    # Get PDF URL - try multiple sources
    pdf_url = paper.get("openAccessPdf", {}).get("url")

    # If no open access PDF, try arXiv
    if not pdf_url:
        external_ids = paper.get("externalIds", {})
        arxiv_id = external_ids.get("ArXiv")
        if arxiv_id:
            # Headers to avoid anti-bot
            test_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            # Try arXiv URLs
            for arxiv_url in [
                f"https://export.arxiv.org/pdf/{arxiv_id}.pdf",
                f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            ]:
                try:
                    test_resp = requests.head(arxiv_url, headers=test_headers, timeout=10, allow_redirects=True)
                    if test_resp.status_code == 200:
                        pdf_url = arxiv_url
                        break
                except:
                    continue

    if not pdf_url:
        return {
            "status": "fail",
            "detail": f"Paper found but no PDF available: {paper.get('title')}",
            "pdf_path": "",
            "paper_info": {
                "title": paper.get("title"),
                "year": paper.get("year"),
                "authors": [a.get("name") for a in paper.get("authors", [])],
                "venue": paper.get("venue"),
                "paper_id": paper.get("paperId"),
                "url": paper.get("url"),
            }
        }

    # Download PDF
    if not output_dir:
        output_dir = os.getcwd()

    safe_title = "".join(c for c in paper.get("title", "paper") if c.isalnum() or c in " -_").strip()[:50]
    filename = f"{safe_title}.pdf"
    pdf_path = os.path.join(output_dir, filename)

    # Add headers to avoid anti-bot (especially for arXiv)
    pdf_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(max_retries):
        try:
            pdf_response = requests.get(pdf_url, headers=pdf_headers, timeout=60)
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"PDF download failed: {str(e)}",
                "pdf_path": "",
                "paper_info": {}
            }

        if pdf_response.status_code == 200:
            break
        else:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "status": "fail",
                "detail": f"PDF download failed with status {pdf_response.status_code}",
                "pdf_path": "",
                "paper_info": {}
            }

    # Save PDF
    with open(pdf_path, "wb") as f:
        f.write(pdf_response.content)

    return {
        "status": "success",
        "detail": f"PDF downloaded: {paper.get('title')}",
        "pdf_path": pdf_path,
        "paper_info": {
            "title": paper.get("title"),
            "year": paper.get("year"),
            "authors": [a.get("name") for a in paper.get("authors", [])],
            "venue": paper.get("venue"),
            "paper_id": paper.get("paperId"),
            "url": paper.get("url"),
            "pdf_url": pdf_url,
        }
    }
