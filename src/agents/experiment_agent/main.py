"""
Main entry point for Experiment Agent System.

This script:
1. Validates configuration
2. Checks Docker environment
3. Sets up OpenAI API
4. Runs the complete experiment workflow
"""

import asyncio
import os
import sys
import argparse
from datetime import datetime
from typing import Optional
from pathlib import Path


from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncAzureOpenAI, AsyncOpenAI

# Import configuration
from src.agents.experiment_agent.config import (
    get_openai_config,
    get_docker_config,
    get_path_config,
    get_model_config,
    validate_config,
    print_config,
    ENABLE_TRACING,
    MODEL_NAME,
)

# Import experiment agent
from src.agents.experiment_agent.sub_agents.experiment_master import (
    create_experiment_master_agent,
)

# Import input processing
from src.agents.experiment_agent.input_processing import (
    load_research_input,
    ResearchInput,
)

# Import tools for Docker setup
from src.agents.experiment_agent.tools.execution_tools import set_docker_client
from src.agents.experiment_agent.environment import create_docker_client

# Import research tools for paper and code preparation
from src.agents.experiment_agent.tools.research_tools import (
    download_papers_by_titles,
    search_and_clone_repos_for_papers,
)

# Import custom logger
from src.agents.experiment_agent.logger import setup_agent_logging


# =============================================================================
# Colored Print Utilities
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_step(message: str):
    """Print a step message."""
    print(f"\n{Colors.OKCYAN}[STEP]{Colors.ENDC} {message}")


def print_success(message: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✓{Colors.ENDC} {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗{Colors.ENDC} {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠{Colors.ENDC} {message}")


def print_info(message: str):
    """Print an info message."""
    print(f"{Colors.OKBLUE}ℹ{Colors.ENDC} {message}")


# =============================================================================
# Configuration Setup
# =============================================================================


def setup_openai_api() -> bool:
    """
    Set up OpenAI API client.

    Returns:
        True if setup successful, False otherwise
    """
    print_step("Setting up OpenAI API...")

    try:
        config = get_openai_config()

        if config["use_azure"]:
            # Azure OpenAI setup
            from httpx import Timeout

            client = AsyncAzureOpenAI(
                api_version=config["api_version"],
                azure_endpoint=config["endpoint"],
                api_key=config["api_key"],
                timeout=Timeout(
                    connect=10.0,  # 连接超时10秒
                    read=300.0,  # 读取超时300秒（5分钟）
                    write=30.0,  # 写入超时30秒
                    pool=10.0,  # 连接池超时10秒
                ),
                max_retries=3,  # 最多重试3次
            )
            print_success(
                f"Azure OpenAI client initialized (endpoint: {config['endpoint']}, timeout: 300s)"
            )
        else:
            # OpenAI setup
            from httpx import Timeout

            client_kwargs = {
                "api_key": config["api_key"],
                "timeout": Timeout(
                    connect=10.0,  # 连接超时10秒
                    read=300.0,  # 读取超时300秒（5分钟）
                    write=30.0,  # 写入超时30秒
                    pool=10.0,  # 连接池超时10秒
                ),
                "max_retries": 3,  # 最多重试3次
            }
            if "base_url" in config and config["base_url"]:
                client_kwargs["base_url"] = config["base_url"]
                print_info(f"Using custom API base: {config['base_url']}")
            else:
                print_info("Using default OpenAI API base: https://api.openai.com/v1")

            client = AsyncOpenAI(**client_kwargs)
            print_success("OpenAI client initialized (timeout: 300s, retries: 3)")

        # Set as default client
        set_default_openai_client(client)
        set_default_openai_api("chat_completions")

        # Set tracing
        set_tracing_disabled(not ENABLE_TRACING)
        if ENABLE_TRACING:
            print_info("Tracing enabled")
        else:
            print_info("Tracing disabled")

        print_success("OpenAI API setup completed")
        return True

    except Exception as e:
        print_error(f"Failed to setup OpenAI API: {str(e)}")
        return False


def check_docker_environment() -> bool:
    """
    Check if Docker environment is properly configured.

    Returns:
        True if Docker is ready, False otherwise
    """
    print_step("Checking Docker environment...")

    try:
        docker_config = get_docker_config()

        # Create Docker client
        client = create_docker_client(
            host=docker_config["host"],
            port=docker_config["port"],
            timeout=docker_config["timeout"],
        )

        # Test connection
        result = client.test_connection()

        if result["success"]:
            print_success(
                f"Docker connection successful ({docker_config['host']}:{docker_config['port']})"
            )
            print_info(f"Container response: {result['message']}")

            # Set as global Docker client
            set_docker_client(client)

            # Get environment info
            env_info = client.run_command("python --version", retry=False)
            if env_info["status"] == 0:
                print_info(f"Python version: {env_info['result'].strip()}")

            return True
        else:
            print_error(f"Docker connection failed: {result['message']}")
            print_warning(
                f"Make sure Docker container is running on {docker_config['host']}:{docker_config['port']}"
            )
            print_warning("You can start the container with: docker-compose up -d")
            return False

    except Exception as e:
        print_error(f"Docker environment check failed: {str(e)}")
        return False


def validate_input_paths(
    input_type: str, input_path: Optional[str] = None
) -> tuple[bool, str]:
    """
    Validate input file exists.

    Args:
        input_type: Type of input ('paper' or 'idea')
        input_path: Optional custom input path

    Returns:
        Tuple of (is_valid, actual_path)
    """
    print_step(f"Validating {input_type} input...")

    path_config = get_path_config()

    if input_path:
        actual_path = input_path
    else:
        actual_path = (
            path_config["paper_input"]
            if input_type == "paper"
            else path_config["idea_input"]
        )

    if not os.path.exists(actual_path):
        print_error(f"Input file not found: {actual_path}")
        return False, actual_path

    file_size = os.path.getsize(actual_path)
    print_success(f"Input file found: {actual_path} ({file_size} bytes)")

    return True, actual_path


def run_prepare_workflow(input_path: str) -> bool:
    """
    Run preparation workflow to download papers and clone repositories.

    This workflow:
    1. Loads reference papers from idea.json
    2. Downloads papers from arXiv (sequential)
    3. Searches and clones related GitHub repositories (sequential)

    Args:
        input_path: Path to idea.json file

    Returns:
        True if preparation completed successfully, False otherwise
    """
    print_step("Running preparation workflow...")

    try:
        # Load idea.json to get reference papers
        import json

        print_info(f"Loading reference papers from: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            idea_data = json.load(f)

        # Extract reference papers
        reference_papers = idea_data.get("reference_papers", [])

        if not reference_papers:
            print_warning("No reference papers found in idea.json")
            return True  # Not an error, just nothing to prepare

        print_success(f"Found {len(reference_papers)} reference papers")
        for i, paper in enumerate(reference_papers, 1):
            print(f"  {i}. {paper}")

        # Get path configuration
        path_config = get_path_config()
        workspace_dir = path_config["local_workspace"]

        # Create output directories
        papers_dir = os.path.join(workspace_dir, "papers")
        repos_dir = os.path.join(workspace_dir, "repos")

        os.makedirs(papers_dir, exist_ok=True)
        os.makedirs(repos_dir, exist_ok=True)

        print_info(f"Papers will be saved to: {papers_dir}")
        print_info(f"Repositories will be cloned to: {repos_dir}")

        # Download papers from arXiv sequentially (single-threaded)
        print_step(f"Downloading papers from arXiv (sequential)...")

        paper_results = []
        for title in reference_papers:
            print(f"\n{'='*60}")
            print(f"Searching for: {title}")
            print("=" * 60)

            from src.agents.experiment_agent.tools.research_tools import (
                search_arxiv,
                download_arxiv_source,
            )

            # Search for paper with more results to increase chance of finding exact match
            papers = search_arxiv(title, max_results=50)

            if len(papers) == 0:
                msg = f"Cannot find the paper '{title}' in arxiv"
                print(f"❌ {msg}")
                paper_results.append(
                    {"status": -1, "message": msg, "path": None, "title": title}
                )
                continue

            # Display search results
            print(f"Found {len(papers)} results for '{title}'")
            if len(papers) > 0:
                print("  Top 5 results:")
                for i, paper in enumerate(papers[:5]):
                    print(f"    {i+1}. {paper['title']}")

            # Check for exact match - only accept completely identical titles
            title_lower = title.lower().strip()
            title_normalized = " ".join(title_lower.split())

            exact_match = None
            for paper in papers:
                paper_title_lower = paper["title"].lower().strip()
                paper_title_normalized = " ".join(paper_title_lower.split())

                # Strategy 1: Check exact match (normalized whitespace only)
                if paper_title_normalized == title_normalized:
                    exact_match = paper
                    print(f"  ✓ Found exact match (normalized): {paper['title']}")
                    break

            # Strategy 2: If no exact match, try ignoring punctuation (still strict)
            if not exact_match:
                import string

                title_no_punct = title_normalized.translate(
                    str.maketrans("", "", string.punctuation)
                ).strip()
                title_no_punct = " ".join(title_no_punct.split())

                for paper in papers:
                    paper_title_lower = paper["title"].lower().strip()
                    paper_title_normalized = " ".join(paper_title_lower.split())
                    paper_title_no_punct = paper_title_normalized.translate(
                        str.maketrans("", "", string.punctuation)
                    ).strip()
                    paper_title_no_punct = " ".join(paper_title_no_punct.split())

                    # Only accept if words match exactly (ignoring punctuation)
                    if title_no_punct == paper_title_no_punct:
                        exact_match = paper
                        print(
                            f"  ✓ Found exact match (ignoring punctuation): {paper['title']}"
                        )
                        break

            if not exact_match:
                msg = f"No exact match found for '{title}'. Skipping download."
                print(f"⚠ {msg}")
                if len(papers) > 0:
                    print(f"  Best match was: {papers[0]['title']}")
                    print(f"  Searched through {len(papers)} results")
                paper_results.append(
                    {"status": -1, "message": msg, "path": None, "title": title}
                )
                continue

            # Use exact match
            best_paper = exact_match
            print(f"  ✓ Found exact match: {best_paper['title']}")
            print(f"  ArXiv URL: {best_paper.get('url', 'N/A')}")

            # Download paper
            try:
                download_info = download_arxiv_source(
                    best_paper["url"], workspace_dir, title
                )

                if download_info["status"] == 0:
                    print(f"  ✓ Successfully downloaded: {title}")
                    print(f"    Path: {download_info['path']}")

                paper_results.append({**download_info, "title": title})
            except Exception as e:
                print(f"❌ Error downloading '{title}': {str(e)}")
                paper_results.append(
                    {"status": -1, "message": str(e), "path": None, "title": title}
                )

        # Print paper download results
        successful_papers = sum(1 for r in paper_results if r["status"] == 0)
        print_success(
            f"Downloaded {successful_papers}/{len(reference_papers)} papers successfully"
        )

        for result in paper_results:
            if result["status"] == 0:
                print_success(f"  ✓ {result['title']}")
            else:
                print_warning(f"  ✗ {result['title']}: {result['message']}")

        # Search and clone GitHub repositories sequentially (single-threaded)
        print_step(f"Searching and cloning GitHub repositories (sequential)...")

        repo_results = []
        for title in reference_papers:
            print(f"\n{'='*60}")
            print(f"Searching for code repositories for: {title}")
            print("=" * 60)

            from src.agents.experiment_agent.tools.research_tools import (
                search_github_repos,
                clone_github_repo,
                extract_github_links_from_text,
            )
            import time

            max_repos_per_paper = 1
            results = []

            try:
                # Step 1: Try to find GitHub links in downloaded paper content
                github_links = []
                paper_path = None

                # Find the downloaded paper file
                for result in paper_results:
                    if result.get("title") == title and result.get("status") == 0:
                        paper_path = result.get("path")
                        break

                if paper_path and os.path.exists(paper_path):
                    print("Searching for GitHub links in paper content...")

                    try:
                        with open(paper_path, "r", encoding="utf-8") as f:
                            paper_content = f.read()

                        github_links = extract_github_links_from_text(paper_content)

                        if github_links:
                            print(
                                f"  ✓ Found {len(github_links)} GitHub link(s) in paper:"
                            )
                            for link in github_links[:max_repos_per_paper]:
                                print(f"    - {link}")
                        else:
                            print(f"  ⚠ No GitHub links found in paper content")
                    except Exception as e:
                        print(f"  ⚠ Failed to read paper content: {str(e)}")
                else:
                    print(f"  ⚠ Paper not downloaded, skipping content search")

                # Step 2: Clone repositories from paper links if found
                cloned_count = 0
                if github_links:
                    for link in github_links[:max_repos_per_paper]:
                        if cloned_count >= max_repos_per_paper:
                            break

                        # Convert GitHub URL to clone URL
                        if "github.com" in link:
                            # Extract owner/repo from URL
                            parts = (
                                link.replace("https://", "")
                                .replace("http://", "")
                                .split("/")
                            )
                            if len(parts) >= 3 and parts[0] == "github.com":
                                owner = parts[1]
                                repo = parts[2].split(".")[0]  # Remove .git if present
                                clone_url = f"https://github.com/{owner}/{repo}.git"
                                repo_name = f"{owner}_{repo}"

                                print(f"  Cloning {owner}/{repo} from paper link...")

                                clone_result = clone_github_repo(
                                    clone_url,
                                    repos_dir,
                                    repo_name,
                                )

                                results.append(
                                    {
                                        **clone_result,
                                        "paper_title": title,
                                        "repo_name": f"{owner}/{repo}",
                                        "repo_url": link,
                                        "source": "paper_content",
                                    }
                                )

                                if clone_result["status"] == 0:
                                    print(f"    ✓ {clone_result['message']}")
                                    cloned_count += 1
                                else:
                                    print(f"    ✗ {clone_result['message']}")

                                time.sleep(1)  # Rate limiting

                # Step 3: If not enough repos from paper, search GitHub
                if cloned_count < max_repos_per_paper:
                    print("Searching GitHub for additional repositories...")

                    query = f"{title} -user:lucidrains"
                    repos = search_github_repos(
                        query, limit=(max_repos_per_paper - cloned_count) * 2
                    )

                    if len(repos) > 0:
                        print(f"  Found {len(repos)} repositories on GitHub")
                        for i, repo in enumerate(
                            repos[: max_repos_per_paper - cloned_count]
                        ):
                            print(f"    {i+1}. {repo['name']} - {repo['stars']} stars")

                        # Clone additional repositories
                        for repo in repos[: max_repos_per_paper - cloned_count]:
                            if cloned_count >= max_repos_per_paper:
                                break

                            print(f"  Cloning {repo['name']}...")

                            clone_result = clone_github_repo(
                                repo["clone_url"],
                                repos_dir,
                                repo["name"].replace("/", "_"),
                            )

                            results.append(
                                {
                                    **clone_result,
                                    "paper_title": title,
                                    "repo_name": repo["name"],
                                    "repo_url": repo["link"],
                                    "source": "github_search",
                                }
                            )

                            if clone_result["status"] == 0:
                                print(f"    ✓ {clone_result['message']}")
                                cloned_count += 1
                            else:
                                print(f"    ✗ {clone_result['message']}")

                            time.sleep(1)  # Rate limiting
                    else:
                        if cloned_count == 0:
                            msg = f"No GitHub repositories found for '{title}'"
                            print(f"  ⚠ {msg}")
                            results.append(
                                {
                                    "status": -1,
                                    "message": msg,
                                    "path": None,
                                    "paper_title": title,
                                }
                            )

            except Exception as e:
                msg = f"Failed to search/clone repositories for '{title}': {str(e)}"
                print(f"❌ {msg}")
                results.append(
                    {"status": -1, "message": msg, "path": None, "paper_title": title}
                )

            repo_results.extend(results)

        # Print repository clone results
        successful_repos = sum(1 for r in repo_results if r["status"] == 0)
        print_success(f"Cloned {successful_repos} repositories successfully")

        for result in repo_results:
            if result["status"] == 0:
                print_success(
                    f"  ✓ {result.get('repo_name', 'Unknown')} for '{result['paper_title']}'"
                )
            else:
                print_warning(
                    f"  ✗ {result.get('repo_name', 'Unknown')}: {result['message']}"
                )

        # Summary
        print(f"\n{Colors.BOLD}Preparation Summary:{Colors.ENDC}")
        print(f"  Papers downloaded: {successful_papers}/{len(reference_papers)}")
        print(f"  Repositories cloned: {successful_repos}")
        print(f"  Papers directory: {papers_dir}")
        print(f"  Repos directory: {repos_dir}")

        print_success("Preparation workflow completed!")
        return True

    except KeyboardInterrupt:
        print_warning("\nPreparation workflow interrupted by user")
        return False
    except Exception as e:
        print_error(f"Preparation workflow failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Main Workflow
# =============================================================================


async def run_experiment_workflow(
    input_type: str,
    input_path: str,
    max_iterations: int = 5,
    verbose: bool = False,
    experiment_id: Optional[str] = None,
) -> bool:
    """
    Run the complete experiment workflow.

    Args:
        input_type: Type of input ('paper' or 'idea')
        input_path: Path to input file
        max_iterations: Maximum workflow iterations
        verbose: Enable verbose logging
        experiment_id: Experiment ID for caching

    Returns:
        True if workflow completed successfully, False otherwise
    """
    print_step("Starting experiment workflow...")

    try:
        # Load input file
        print_info(f"Loading {input_type} from: {input_path}")

        if input_type == "idea":
            # For ideas, load JSON directly without conversion
            import json

            with open(input_path, "r", encoding="utf-8") as f:
                idea_data = json.load(f)

            # Extract title for display
            title = idea_data.get("idea", {}).get("title", "Untitled Idea")
            print_success(f"Idea loaded successfully")
            print_info(f"Title: {title}")

            # Pass the raw JSON data directly
            research_input = json.dumps(idea_data, indent=2, ensure_ascii=False)
            actual_input_type = "idea"

        else:
            # For papers, use the conversion pipeline
            research_input_obj = load_research_input(input_path, encoding="utf-8")

            print_success(f"Input loaded and converted to standardized format")
            print_info(f"Input type: {research_input_obj.input_type}")
            if research_input_obj.title:
                print_info(f"Title: {research_input_obj.title}")
            print_info(f"Content length: {len(research_input_obj.content)} characters")

            # Convert to text format for agent processing
            research_input = research_input_obj.to_text()
            actual_input_type = research_input_obj.input_type

        # Create experiment master agent
        print_step("Initializing Experiment Master Agent...")

        path_config = get_path_config()
        model_config = get_model_config()

        master_agent = create_experiment_master_agent(
            model=MODEL_NAME,
            expensive_model=model_config["expensive_model"],
            cheap_model=model_config["cheap_model"],
            max_iterations=max_iterations,
            working_dir=path_config["project_dir"],
            log_dir=path_config["logs_dir"],
            cache_dir=path_config["cache_dir"],
            verbose=verbose,
        )

        print_success("Experiment Master Agent initialized")
        print_info(f"Max iterations: {max_iterations}")
        print_info(
            f"Expensive model (code plan/implement/judge): {model_config['expensive_model']}"
        )
        print_info(
            f"Cheap model (pre-analysis/execute/analysis): {model_config['cheap_model']}"
        )
        print_info(f"Working directory: {path_config['project_dir']}")
        print_info(f"Log directory: {path_config['logs_dir']}")
        print_info(f"Cache directory: {path_config['cache_dir']}")

        # Run workflow
        print_step("Running experiment workflow...")
        print_info("This may take a while depending on the complexity...")
        print_info("Streaming mode enabled - you will see real-time output")

        start_time = datetime.now()

        result = await master_agent.run_workflow(
            research_input=research_input,
            input_type=actual_input_type,
            experiment_id=experiment_id,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Print results
        print_step("Workflow Results")
        print(f"\n{'=' * 80}")
        print(f"Status: {result.final_status}")
        print(f"Completed: {result.workflow_completed}")
        print(f"Total Iterations: {result.total_iterations}")
        print(f"Execution Time: {duration:.2f} seconds")
        print(f"{'=' * 80}")

        print(f"\n{Colors.BOLD}Workflow History:{Colors.ENDC}")
        for i, step in enumerate(result.workflow_history, 1):
            print(f"  {i}. {step.agent_name} - {step.status}")

        print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
        print(result.overall_summary)

        if result.key_findings:
            print(f"\n{Colors.BOLD}Key Findings:{Colors.ENDC}")
            for finding in result.key_findings:
                print(f"  • {finding}")

        if result.final_recommendations:
            print(f"\n{Colors.BOLD}Recommendations:{Colors.ENDC}")
            print(result.final_recommendations)

        # Save results
        results_file = os.path.join(
            path_config["results_dir"],
            f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"Experiment Results\n")
            f.write(f"{'=' * 80}\n\n")
            f.write(f"Input Type: {actual_input_type}\n")
            f.write(f"Input Path: {input_path}\n")
            if input_type == "idea":
                f.write(f"Title: {title}\n")
            elif (
                hasattr(locals().get("research_input_obj"), "title")
                and locals().get("research_input_obj").title
            ):
                f.write(f"Title: {locals().get('research_input_obj').title}\n")
            f.write(f"Status: {result.final_status}\n")
            f.write(f"Completed: {result.workflow_completed}\n")
            f.write(f"Total Iterations: {result.total_iterations}\n")
            f.write(f"Execution Time: {duration:.2f} seconds\n\n")
            f.write(f"Workflow History:\n")
            for i, step in enumerate(result.workflow_history, 1):
                f.write(f"  {i}. {step.agent_name} - {step.status}\n")
            f.write(f"\nSummary:\n{result.overall_summary}\n\n")
            if result.key_findings:
                f.write(f"Key Findings:\n")
                for finding in result.key_findings:
                    f.write(f"  • {finding}\n")
            if result.final_recommendations:
                f.write(f"\nRecommendations:\n{result.final_recommendations}\n")

        print_success(f"Results saved to: {results_file}")

        if result.workflow_completed:
            print_success("Workflow completed successfully!")
            return True
        else:
            print_warning("Workflow completed with issues")
            return False

    except KeyboardInterrupt:
        print_warning("\nWorkflow interrupted by user")
        return False
    except Exception as e:
        print_error(f"Workflow failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Main Function
# =============================================================================


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Research Experiment Agent Workflow"
    )

    parser.add_argument(
        "--input-type",
        type=str,
        choices=["paper", "idea"],
        default="idea",
        help="Type of research input (default: idea)",
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to input file (overrides config)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum workflow iterations (default: 5)",
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip environment checks (not recommended)",
    )

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print configuration and exit",
    )

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run preparation workflow to download papers and clone repositories",
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only run preparation workflow, skip experiment workflow",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose agent logging (DEBUG level)",
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID for caching (if not provided, auto-generates timestamp-based ID)",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = get_args()

    # Print header
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("=" * 80)
    print("Research Experiment Agent System".center(80))
    print("=" * 80)
    print(f"{Colors.ENDC}\n")

    # Print configuration if requested
    if args.print_config:
        print_config()
        return

    # Step 1: Validate configuration
    print_step("Validating configuration...")
    is_valid, errors = validate_config()

    if not is_valid:
        print_error("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print_success("Configuration is valid")

    # Setup agent logging
    setup_agent_logging(verbose=args.verbose if hasattr(args, "verbose") else False)

    # Step 2: Check Docker environment
    if not args.skip_checks:
        if not check_docker_environment():
            print_error("Docker environment check failed")
            print_info(
                "You can skip this check with --skip-checks flag (not recommended)"
            )
            sys.exit(1)
    else:
        print_warning("Skipping Docker environment check")

    # Step 3: Setup OpenAI API
    if not setup_openai_api():
        print_error("OpenAI API setup failed")
        sys.exit(1)

    # Step 4: Validate input
    is_valid, input_path = validate_input_paths(args.input_type, args.input_path)
    if not is_valid:
        sys.exit(1)

    # Step 5: Run preparation workflow if requested
    if args.prepare or args.prepare_only:
        if args.input_type == "idea":
            prepare_success = run_prepare_workflow(input_path)
            if not prepare_success:
                print_error("Preparation workflow failed")
                sys.exit(1)
        else:
            print_warning(
                "Preparation workflow is only available for idea input type, skipping..."
            )

        # If prepare-only mode, exit after preparation
        if args.prepare_only:
            print(f"\n{Colors.HEADER}{Colors.BOLD}")
            print("=" * 80)
            print(f"{Colors.OKGREEN}Preparation completed successfully!{Colors.ENDC}")
            print("=" * 80)
            print(f"{Colors.ENDC}\n")
            sys.exit(0)

    # Step 6: Run experiment workflow
    success = await run_experiment_workflow(
        input_type=args.input_type,
        input_path=input_path,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        experiment_id=args.experiment_id,
    )

    # Final status
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("=" * 80)
    if success:
        print(f"{Colors.OKGREEN}Experiment completed successfully!{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}Experiment completed with errors{Colors.ENDC}")
    print("=" * 80)
    print(f"{Colors.ENDC}\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
