"""
Test script for new BaseAgent (OpenHands SDK) - testing skills + tools.
"""

import asyncio
import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime

# Add src/agents to path
current_dir = os.path.dirname(os.path.abspath(__file__))
agents_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
repo_root = os.path.abspath(os.path.join(agents_root, "..", ".."))

if agents_root not in sys.path:
    sys.path.insert(0, agents_root)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from openhands.sdk import Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.glob import GlobTool
from openhands.tools.grep import GrepTool


from blog_agent.agent.new_base_agent import BaseAgent
from blog_agent.config.loader import load_config
from blog_agent.tools.illustrate import illustrate
import blog_agent.tools.search_core_tool
import blog_agent.tools.search_paper_abstract_tool
import blog_agent.tools.download_paper_pdf_tool
import blog_agent.tools.count_words_tool


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL= "MiniMax-M2.5"

# Define a custom agent
class IDEAAgent(BaseAgent):
    def _build_system_prompt(self, **kwargs) -> str:
        return """You are a helpful assistant with access to terminal commands."""


class WRITEAgent(BaseAgent):
    def _build_system_prompt(self, **kwargs) -> str:
        return """You are a helpful assistant with access to terminal commands."""


class ANALYZEAgent(BaseAgent):
    def _build_system_prompt(self, **kwargs) -> str:
        return """You are a helpful assistant with access to terminal commands."""


class REFINEAgent(BaseAgent):
    def _build_system_prompt(self, **kwargs) -> str:
        return """You are a helpful assistant with access to terminal commands."""


def get_status_file(experiment: str) -> str:
    """Get the path to the status file for this experiment."""
    return os.path.join(agents_root, "blog_agent", "workspaces", experiment, "workflow_status.json")


def load_status(experiment: str) -> dict:
    """Load workflow status from file."""
    status_file = get_status_file(experiment)
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load status file: {e}")
    return {
        "experiment": experiment,
        "current_step": 0,
        "current_loop": 1,
        "steps_completed": [],
        "last_updated": None
    }


def save_status(experiment: str, status: dict):
    """Save workflow status to file."""
    status_file = get_status_file(experiment)
    status["last_updated"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, 'w', encoding='utf-8') as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    logger.info(f"Status saved: step {status['current_step']}, completed: {status['steps_completed']}")


def parse_blog_score(file_path: str) -> tuple[int, str]:
    """
    Parse blog_analysis.md and extract total score and rating.

    Args:
        file_path: Path to blog_analysis.md

    Returns:
        Tuple of (score, rating) or (0, "") if parsing fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return 0, ""
    except Exception:
        return 0, ""

    # Match: **Score: 68/100** -- Below Standard
    match = re.search(r'\*\*Score:\s*(\d+)/(\d+)\*\*\s*--\s*(.+)', content)
    if not match:
        return 0, ""

    score = int(match.group(1))
    rating = match.group(3).strip()
    return score, rating


async def main(experiment: str, resume: bool = False):
    workspace = repo_root
    skills_dir = os.path.join(agents_root, "blog_agent", "skills")
    maxturn = 3
    config = load_config()

    # Load or initialize status
    status = load_status(experiment)
    start_step = 0
    start_loop = 1

    if resume:
        start_step = status.get("current_step", 0)
        start_loop = status.get("current_loop", 1)
        logger.info(f"Resuming from step {start_step}, loop {start_loop}, completed steps: {status.get('steps_completed', [])}")

    # Check and create workspace directories
    blog_workspace = os.path.join(agents_root, "blog_agent", "workspaces", experiment)
    default_source_workspace_root = config.get(
        "source_workspace_root",
        os.path.join(repo_root, "workspace"),
    )
    source_workspace = os.environ.get("BLOG_AGENT_SOURCE_WORKSPACE")
    if not source_workspace:
        source_workspace = os.path.join(default_source_workspace_root, experiment)
    source_workspace = os.path.abspath(os.path.expanduser(source_workspace))

    if not os.path.exists(source_workspace):
        logger.error(f"Source workspace not found: {source_workspace}")
        logger.error(f"Please ensure the experiment '{experiment}' exists before running blog workflow.")
        sys.exit(1)

    if not os.path.exists(blog_workspace):
        logger.info(f"Creating blog workspace: {blog_workspace}")
        os.makedirs(blog_workspace, exist_ok=True)

    # Create test_output directory for images
    test_output_dir = os.path.join(blog_workspace, "test_output")
    if not os.path.exists(test_output_dir):
        logger.info(f"Creating test_output directory: {test_output_dir}")
        os.makedirs(test_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running blog workflow for experiment: {experiment}")
    print(f"Source workspace: {source_workspace}")
    print(f"Blog workspace: {blog_workspace}")
    print(f"{'='*60}\n")

    # Create agents
    ideaagent = IDEAAgent(
        agent_type="IDEAAgent",
        model=MODEL,
        workspace=workspace,
    )
    ideaagent.add_tool(Tool(name=TerminalTool.name))
    ideaagent.add_tool(Tool(name=FileEditorTool.name))
    ideaagent.add_tool(Tool(name=GlobTool.name))
    ideaagent.add_tool(Tool(name=GrepTool.name))
    ideaagent.add_tool(Tool(name="SearchCoreNodesTool"))
    ideaagent.add_tool(Tool(name="SearchPaperAbstractTool"))
    ideaagent.add_tool(Tool(name="DownloadPaperPdfTool"))
    ideaagent.load_skills_from_dir(skills_dir)

    writeagent = WRITEAgent(
        agent_type="WRITEAgent",
        model=MODEL,
        workspace=workspace,
    )
    writeagent.add_tool(Tool(name=TerminalTool.name))
    writeagent.add_tool(Tool(name=FileEditorTool.name))
    writeagent.add_tool(Tool(name=GlobTool.name))
    writeagent.add_tool(Tool(name=GrepTool.name))
    writeagent.load_skills_from_dir(skills_dir)

    analyzeagent = ANALYZEAgent(
        agent_type="ANALYZEAgent",
        model=MODEL,
        workspace=workspace,
    )
    analyzeagent.add_tool(Tool(name=TerminalTool.name))
    analyzeagent.add_tool(Tool(name=FileEditorTool.name))
    analyzeagent.add_tool(Tool(name=GlobTool.name))
    analyzeagent.add_tool(Tool(name=GrepTool.name))
    analyzeagent.add_tool(Tool(name="CountWords"))
    analyzeagent.load_skills_from_dir(skills_dir)

    refineagent = REFINEAgent(
        agent_type="REFINEAgent",
        model=MODEL,
        workspace=workspace,
    )
    refineagent.add_tool(Tool(name=TerminalTool.name))
    refineagent.add_tool(Tool(name=FileEditorTool.name))
    refineagent.add_tool(Tool(name=GlobTool.name))
    refineagent.add_tool(Tool(name=GrepTool.name))
    refineagent.load_skills_from_dir(skills_dir)

    # Step 1: IDEAAgent
    if start_step <= 0:
        logger.info("[Step 1/4] Running IDEAAgent (workspace-navigator)...")
        result = await ideaagent.run_async(
            user_prompt=f"Explore {experiment} project from source workspace {source_workspace}. Write blog_idea.md in blog workspace {blog_workspace}, then search papers, get details, download PDFs to {blog_workspace}, verify content, and add Papers for Citation table to blog_idea.md."
        )
        logger.info(f"[RESULT] {result[:500]}..." if len(result) > 500 else f"[RESULT] {result}")
        status["current_step"] = 1
        status["steps_completed"].append("IDEAAgent")
        save_status(experiment, status)

    # Step 2: WRITEAgent
    if start_step <= 1:
        logger.info("[Step 2/4] Running WRITEAgent (blog-writer)...")
        result = await writeagent.run_async(
            user_prompt=f"Write blog article for {experiment}. Read blog_idea.md from {blog_workspace}, check Papers for Citation, read relevant PDFs from {blog_workspace}, compare against source workspace {source_workspace}, write section by section with citations<sup>[1]</sup>, create graph method files in {test_output_dir}, and insert <graph1> placeholders."
        )
        logger.info(f"[RESULT] {result[:500]}..." if len(result) > 500 else f"[RESULT] {result}")
        status["current_step"] = 2
        status["steps_completed"].append("WRITEAgent")
        save_status(experiment, status)

    # Step 3: ANALYZEAgent
    if start_step <= 2:
        logger.info("[Step 3/4] Running ANALYZEAgent (blog-analyze)...")
        result = await analyzeagent.run_async(
            user_prompt=f"Analyze blog article for {experiment}. Read blog_article.md from {blog_workspace}, verify code accuracy against source workspace {source_workspace}, verify citations against PDFs in {blog_workspace}, check images in {test_output_dir}, and generate quality report with research integrity check."
        )
        logger.info(f"[RESULT] {result[:500]}..." if len(result) > 500 else f"[RESULT] {result}")
        status["current_step"] = 3
        status["steps_completed"].append("ANALYZEAgent")
        save_status(experiment, status)

    # Step 4: REFINEAgent (loop)
    blog_analysis_path = os.path.join(agents_root, "blog_agent", "workspaces", experiment, "blog_analysis.md")
    score, rating = parse_blog_score(blog_analysis_path)
    logger.info(f"Score: {score}, Rating: {rating}")
    currentloop = start_loop

    while score <= 90 and currentloop <= maxturn:
        # Save current loop progress before starting
        status["current_loop"] = currentloop
        status["current_step"] = 3  # Step 3 = ANALYZE/REFINE phase
        save_status(experiment, status)

        logger.info(f"[Refine Loop {currentloop}/{maxturn}] Running REFINEAgent...")

        result = await refineagent.run_async(
            user_prompt=f"Refine blog article for {experiment}. Read blog_article.md and blog_analysis.md from {blog_workspace}, improve based on recommendations, fix image issues, and fix citation issues if any."
        )
        logger.info(f"[RESULT] {result[:500]}..." if len(result) > 500 else f"[RESULT] {result}")

        # Remove old analysis and re-analyze
        if os.path.exists(blog_analysis_path):
            os.remove(blog_analysis_path)

        logger.info(f"[Refine Loop {currentloop}/{maxturn}] Running ANALYZEAgent...")
        result = await analyzeagent.run_async(
            user_prompt=f"Analyze blog article for {experiment}. Read blog_article.md from {blog_workspace}, verify code accuracy against source workspace {source_workspace}, verify citations against PDFs in {blog_workspace}, check images in {test_output_dir}, and generate quality report with research integrity check."
        )
        logger.info(f"[RESULT] {result[:500]}..." if len(result) > 500 else f"[RESULT] {result}")

        score, rating = parse_blog_score(blog_analysis_path)
        logger.info(f"Score: {score}, Rating: {rating}")
        status["current_loop"] = currentloop
        save_status(experiment, status)
        currentloop += 1

    # Mark complete
    status["current_step"] = 4
    status["current_loop"] = 1  # Reset loop counter for next run
    status["steps_completed"].append("REFINEAgent")
    save_status(experiment, status)

    # Step 5: Generate images and replace placeholders
    if start_step <= 4:
        logger.info("[Step 5/5] Generating images and replacing placeholders...")

        workspace = os.path.join(agents_root, "blog_agent", "workspaces", experiment)
        output_dir = os.path.join(workspace, "test_output")
        blog_article_path = os.path.join(workspace, "blog_article.md")

        only_gen_img = config.get("illustrate", {}).get("only_gen_img", True)

        # Find graph files
        graph_files = []
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                match = re.match(r'graph(\d+)\.md', filename)
                if match:
                    graph_num = int(match.group(1))
                    graph_files.append((graph_num, os.path.join(output_dir, filename)))

        graph_files = sorted(graph_files, key=lambda x: x[0])

        if not graph_files:
            logger.warning("No graph method files found, skipping image generation")
        else:
            logger.info(f"Found {len(graph_files)} graph files, generating images...")

            # Generate images
            for graph_num, method_file in graph_files:
                output_filename = f"figure{graph_num}.png"
                logger.info(f"  Generating {output_filename}...")

                result = illustrate(
                    method_file=method_file,
                    output_dir=output_dir,
                    output_filename=output_filename,
                    only_gen_img=only_gen_img,
                )

                if result.get("success"):
                    logger.info(f"  ✓ Success: {result.get('figure_path', output_filename)}")
                else:
                    logger.error(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

            # Replace <graphN> placeholders with markdown images
            logger.info("Replacing placeholders with images...")

            if os.path.exists(blog_article_path):
                with open(blog_article_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                for graph_num, _ in graph_files:
                    graph_md_path = os.path.join(output_dir, f"graph{graph_num}.md")

                    # Extract description from graphN.md
                    desc = "Graph image"
                    if os.path.exists(graph_md_path):
                        with open(graph_md_path, 'r', encoding='utf-8') as f:
                            graph_content = f.read()
                        lines = graph_content.strip().split('\n')
                        for line in lines:
                            if line.strip() and not line.startswith('#'):
                                desc = line.strip()[:50]
                                break

                    rel_path = f"test_output/figure{graph_num}.png"
                    old = f"<graph{graph_num}>"
                    new = f"![{desc}]({rel_path})"
                    content = content.replace(old, new)
                    logger.info(f"  Replaced {old} -> {new}")

                with open(blog_article_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                logger.info(f"Updated: {blog_article_path}")
            else:
                logger.warning(f"Blog article not found: {blog_article_path}")

        status["current_step"] = 5
        status["steps_completed"].append("ImageGeneration")
        save_status(experiment, status)

    print(f"\n{'='*60}")
    print(f"Blog workflow completed for: {experiment}")
    print(f"Final Score: {score}, Rating: {rating}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run blog agent workflow")
    parser.add_argument("--experiment", required=True, help="Experiment/project name")
    parser.add_argument("--resume", action="store_true", help="Resume from the last completed step")
    args = parser.parse_args()

    asyncio.run(main(args.experiment, args.resume))
