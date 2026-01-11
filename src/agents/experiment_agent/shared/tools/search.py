import os
import requests
import urllib.parse
import logging
from typing import List, Dict, Any
from agents import function_tool

from src.agents.experiment_agent.shared.utils.memory_middleware import (
    maybe_augment_tool_result,
)

logger = logging.getLogger(__name__)

@function_tool
def search_github_repos(query: str, limit: int = 5) -> Any:
    """
    Search GitHub repositories using the GitHub API.

    Args:
        query: The search query (e.g. "paper title keywords").
        limit: Maximum number of results to return (default: 5, max: 10).

    Returns:
        List of dictionaries containing repository details (name, link, description, stars, language).
    """
    if not query:
        return []

    # Safe limit
    limit = min(max(1, limit), 10)
    
    # Construct query
    q = urllib.parse.quote(query)
    url = f"https://api.github.com/search/repositories?q={q}&per_page={limit}&page=1"
    
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_AI_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            result = [{"error": f"GitHub API returned status {r.status_code}", "details": r.text}]
            return maybe_augment_tool_result(
                tool_name="search_github_repos",
                tool_args={"query": query, "limit": limit},
                result=result,
            )
        
        items = r.json().get("items", []) or []
        repos = []
        for it in items:
            name = f"{it['owner']['login']}/{it['name']}"
            link = it.get("html_url", None)
            desc = it.get("description", None)
            
            repos.append(
                {
                    "name": name,
                    "link": link,
                    "stars": it.get("stargazers_count", None),
                    "language": it.get("language", None),
                    "description": desc,
                }
            )
        return maybe_augment_tool_result(
            tool_name="search_github_repos",
            tool_args={"query": query, "limit": limit},
            result=repos,
        )
    except Exception as e:
        logger.error(f"search_github_repos failed: {e}")
        result = [{"error": str(e)}]
        return maybe_augment_tool_result(
            tool_name="search_github_repos",
            tool_args={"query": query, "limit": limit},
            result=result,
        )
