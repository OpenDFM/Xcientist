"""
Pre-Analysis Agent - Main orchestrator for research analysis.

This agent directly routes inputs (papers or ideas) to appropriate analyzers
and produces unified analysis output.

Architecture:
- Paper Analyzers: Extract concepts and algorithms from papers
- Idea Analyzers: Elaborate concepts and generate algorithms from ideas
- Output Unifier: Synthesizes results into unified format
"""

import asyncio
import os
import time
from typing import Dict, Optional

from agents import Agent, Runner

# Import hooks
from src.agents.experiment_agent.logger import create_verbose_hooks


# =============================================================================
# Pretty Print Utilities
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


def print_section(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{char * 80}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{title.center(80)}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{char * 80}{Colors.ENDC}\n")


def print_subsection(title: str):
    """Print a subsection header."""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{'─' * 80}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}📋 {title}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{Colors.BOLD}{'─' * 80}{Colors.ENDC}\n")


def print_success(message: str, indent: int = 0):
    """Print a success message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.OKGREEN}✓{Colors.ENDC} {message}")


def print_error(message: str, indent: int = 0):
    """Print an error message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.FAIL}✗{Colors.ENDC} {message}")


def print_info(message: str, indent: int = 0):
    """Print an info message."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.OKBLUE}ℹ{Colors.ENDC} {message}")


def print_result_box(title: str, content: str, max_length: int = 500):
    """Print a boxed result with simple header."""
    # Print header
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}  {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}\n")

    # Truncate content if too long
    if len(content) > max_length:
        display_content = content[:max_length] + "\n... (truncated for display)"
    else:
        display_content = content

    # Print content without line truncation
    print(f"{Colors.OKGREEN}{display_content}{Colors.ENDC}\n")
    print(f"{Colors.BOLD}{Colors.OKCYAN}{'═' * 80}{Colors.ENDC}\n")


from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    PreAnalysisOutput,
)
from src.agents.experiment_agent.sub_agents.pre_analysis.paper_concept_analyzer import (
    create_paper_concept_analyzer,
)
from src.agents.experiment_agent.sub_agents.pre_analysis.paper_algorithm_analyzer import (
    create_paper_algorithm_analyzer,
)
from src.agents.experiment_agent.sub_agents.pre_analysis.idea_concept_analyzer import (
    create_idea_concept_analyzer,
)
from src.agents.experiment_agent.sub_agents.pre_analysis.idea_algorithm_analyzer import (
    create_idea_algorithm_analyzer,
)
from src.agents.experiment_agent.sub_agents.pre_analysis.output_unifier import (
    create_output_unifier,
)


class PreAnalysisAgent:
    """
    Main pre-analysis agent that orchestrates the entire analysis workflow.

    This agent:
    1. Detects input type (paper or idea)
    2. Directly calls appropriate concept and algorithm analyzers
    3. Unifies the outputs into a consistent format
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        tools: Optional[Dict[str, list]] = None,
        workspace_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the pre-analysis agent system.

        Args:
            model: Model to use for all agents
            tools: Optional dictionary mapping agent types to their tools.
                   If None, automatically loads recommended tools.
                   e.g., {"paper": [...], "idea": [...]}
            workspace_dir: Workspace directory path
            verbose: If True, enable verbose hooks to show full LLM responses and tool calls
        """
        self.model = model
        self.verbose = verbose

        # Create hooks for verbose output
        self.hooks = (
            create_verbose_hooks(
                show_llm_responses=verbose,
                show_tools=verbose,
            )
            if verbose
            else None
        )

        # Get workspace directory from config if not provided
        if workspace_dir is None:
            from src.agents.experiment_agent.config import LOCAL_WORKSPACE_DIR

            workspace_dir = LOCAL_WORKSPACE_DIR
        self.workspace_dir = workspace_dir

        # Auto-load recommended tools if not provided
        if tools is None:
            # Import here to avoid circular import
            from src.agents.experiment_agent.tools import (
                DOCUMENT_TOOLS,
                FILE_TOOLS,
                REPOSITORY_TOOLS,
            )

            # Minimal tools to avoid token limit issues
            self.tools = {
                "paper": FILE_TOOLS[:3]
                + REPOSITORY_TOOLS[
                    :2
                ],  # read, write, list + list_papers, generate_tree
                "idea": FILE_TOOLS[:3] + REPOSITORY_TOOLS[:2],
            }
        else:
            self.tools = tools

        # Initialize analyzers WITHOUT tools (they should analyze based on provided content only)
        self.paper_concept_analyzer = create_paper_concept_analyzer(
            model=model, tools=[], workspace_dir=workspace_dir
        )

        self.paper_algorithm_analyzer = create_paper_algorithm_analyzer(
            model=model, tools=[], workspace_dir=workspace_dir
        )

        self.idea_concept_analyzer = create_idea_concept_analyzer(
            model=model, tools=[], workspace_dir=workspace_dir
        )

        self.idea_algorithm_analyzer = create_idea_algorithm_analyzer(
            model=model, tools=[], workspace_dir=workspace_dir
        )

        # Initialize output unifier
        self.output_unifier = create_output_unifier(model=model)

        # Create a main agent for handoff compatibility
        # This agent serves as the entry point for the orchestrator
        self.agent = Agent(
            name="Pre-Analysis Agent",
            instructions="""You are the Pre-Analysis Agent responsible for analyzing research papers and ideas.

YOUR TASK:
Analyze the research input (paper or idea) and provide a comprehensive structured analysis covering:

1. **System Architecture and Conceptual Framework**
   - Overall system design and components
   - How different parts interact
   - Core architectural decisions

2. **Key Innovations and Theoretical Basis**
   - Novel contributions and unique approaches
   - Theoretical foundations and principles
   - Design philosophy and rationale

3. **Algorithms and Mathematical Formulations**
   - Core algorithms and their mathematical representation
   - Key equations and formulations
   - Algorithm flow and pseudo-code if applicable

4. **Technical Details and Computational Methods**
   - Implementation considerations
   - Computational complexity
   - Required dependencies and frameworks
   - Performance considerations

IMPORTANT WORKFLOW INSTRUCTIONS:
1. Read and understand the research input carefully
2. Extract and analyze all key technical information
3. Provide a detailed, well-structured analysis in markdown format
4. After completing your analysis, you MUST call the handoff function to return control to the Experiment Master Orchestrator
5. The orchestrator will use your analysis to proceed to the next stage (Code Planning)

OUTPUT FORMAT:
Provide your analysis in clear markdown format with proper sections and subsections.
Then explicitly handoff back to the orchestrator.

Example ending:
"...This analysis reveals a highly innovative framework with potential to address critical issues.
→ Analysis complete. Returning control to Experiment Master Orchestrator for next steps."

Then call: transfer_to_experiment_master_orchestrator()""",
            model=model,
        )

    def _detect_input_type(self, input_data: str) -> str:
        """
        Detect whether the input is a paper or an idea.

        Args:
            input_data: The input content

        Returns:
            "paper" or "idea"
        """
        # First check for standardized input format marker
        if "INPUT_TYPE: PAPER" in input_data:
            return "paper"
        elif "INPUT_TYPE: IDEA" in input_data:
            return "idea"

        # Check for explicit "research idea" markers
        if "Analyze the following research idea:" in input_data:
            return "idea"
        if "=== IDEA INFORMATION ===" in input_data:
            return "idea"

        # Fallback: Simple heuristic check for LaTeX markers or JSON structure
        if "\\documentclass" in input_data or "\\begin{document}" in input_data:
            return "paper"
        elif '"messages"' in input_data or '"idea_evaluation"' in input_data:
            return "idea"

        # Default to paper if uncertain
        return "paper"

    async def _iterative_paper_analysis(
        self, input_data: str, papers_dir: str
    ) -> tuple:
        """
        Iteratively analyze papers in the papers directory.

        Args:
            input_data: The research idea or paper content
            papers_dir: Directory containing papers to analyze

        Returns:
            Tuple of (concept_analysis, algorithm_analysis)
        """
        import os
        from pathlib import Path

        papers_path = Path(papers_dir)
        if not papers_path.exists():
            print_info(
                f"Papers directory not found: {papers_dir}, skipping paper analysis"
            )
            return None, None

        # Get list of paper files (prefer markdown files generated from PDF)
        paper_files = [
            f
            for f in papers_path.iterdir()
            if f.is_file() and f.suffix in [".md", ".txt"]
        ]

        if not paper_files:
            print_info(f"No markdown/text papers found in {papers_dir}")
            return None, None

        print_subsection(f"Iterative Paper Analysis ({len(paper_files)} papers found)")
        for idx, pf in enumerate(paper_files, 1):
            print_info(f"{idx}. {pf.name}", indent=0)

        # Iteratively analyze each paper
        accumulated_concept = ""
        accumulated_algorithm = ""

        for i, paper_file in enumerate(paper_files[:3]):  # Limit to first 3 papers
            print(f"\n{Colors.BOLD}{'─' * 80}{Colors.ENDC}")
            print(
                f"{Colors.BOLD}📄 Paper {i+1}/{min(len(paper_files), 3)}: {Colors.OKGREEN}{paper_file.name}{Colors.ENDC}"
            )
            print(f"{Colors.BOLD}{'─' * 80}{Colors.ENDC}")

            # Read paper content (full content, no truncation)
            try:
                with open(paper_file, "r", encoding="utf-8") as f:
                    paper_content = f.read()  # Read full content
                print_info(f"Loaded {len(paper_content)} characters", indent=1)
            except Exception as e:
                print_error(f"Failed to read: {e}", indent=1)
                continue

            # Concept analysis iteration
            concept_prompt = f"""Analyze this paper in the context of the research idea.

Research Idea Summary:
{input_data}

Paper to analyze:
{paper_content}

Previous accumulated concepts:
{accumulated_concept if accumulated_concept else 'None yet'}

Extract and ADD new conceptual insights from this paper. Build upon previous concepts."""

            try:
                print_info("Analyzing conceptual framework...", indent=1)
                # Use streamed version for real-time output
                concept_stream = Runner.run_streamed(
                    self.paper_concept_analyzer,
                    concept_prompt,
                    hooks=self.hooks,
                    max_turns=100,
                )
                # Process stream and print events for debugging
                async for event in concept_stream.stream_events():
                    # Only print text content
                    if hasattr(event, "data"):
                        event_type = type(event.data).__name__
                        if "FunctionCallArguments" not in event_type and hasattr(
                            event.data, "delta"
                        ):
                            delta = event.data.delta
                            if hasattr(delta, "content") and delta.content:
                                print(delta.content, end="", flush=True)
                            elif hasattr(delta, "text") and delta.text:
                                print(delta.text, end="", flush=True)
                concept_result = concept_stream  # The stream object is the result
                concept_analysis = concept_result.final_output

                # Add to accumulated concepts
                concept_text = concept_analysis.conceptual_framework
                accumulated_concept += (
                    f"\n\n### From {paper_file.name}:\n{concept_text}"
                )

                print_success(
                    f"Concept analysis completed ({len(concept_text)} chars)", indent=1
                )
                print_result_box("Key Concepts Extracted", concept_text, max_length=300)

                # Rate limiting: wait between API calls
                if i < len(paper_files[:3]) - 1:  # Don't wait after last paper
                    print_info(
                        f"Waiting 10 seconds to avoid rate limiting...", indent=1
                    )
                    await asyncio.sleep(10)

            except Exception as e:
                print_error(f"Concept analysis failed: {e}", indent=1)

        return accumulated_concept, accumulated_algorithm

    async def _iterative_repo_analysis(self, input_data: str, repos_dir: str) -> str:
        """
        Iteratively analyze code repositories.

        Args:
            input_data: The research idea
            repos_dir: Directory containing repositories

        Returns:
            Accumulated algorithm analysis
        """
        import os
        from pathlib import Path

        repos_path = Path(repos_dir)
        if not repos_path.exists():
            print_info(
                f"Repos directory not found: {repos_dir}, skipping repo analysis"
            )
            return ""

        # Get list of repositories
        repo_dirs = [d for d in repos_path.iterdir() if d.is_dir()]

        if not repo_dirs:
            print_info(f"No repositories found in {repos_dir}")
            return ""

        print_subsection(
            f"Iterative Repository Analysis ({len(repo_dirs)} repos found)"
        )
        for idx, rd in enumerate(repo_dirs, 1):
            print_info(f"{idx}. {rd.name}", indent=0)

        accumulated_algorithm = ""

        for i, repo_dir in enumerate(repo_dirs[:2]):  # Limit to first 2 repos
            print(f"\n{Colors.BOLD}{'─' * 80}{Colors.ENDC}")
            print(
                f"{Colors.BOLD}💾 Repository {i+1}/{min(len(repo_dirs), 2)}: {Colors.OKGREEN}{repo_dir.name}{Colors.ENDC}"
            )
            print(f"{Colors.BOLD}{'─' * 80}{Colors.ENDC}")

            # Generate code tree for the repo
            from src.agents.experiment_agent.tools.repository_tools import (
                generate_code_tree,
                get_repository_overview,
            )

            print_info("Generating code tree...", indent=1)
            tree_result = generate_code_tree(str(repo_dir), max_depth=3)
            if not tree_result["success"]:
                print_error(
                    f"Failed to generate tree: {tree_result.get('error')}", indent=1
                )
                continue

            print_success(
                f"Code tree generated ({tree_result['statistics']['files']} files, {tree_result['statistics']['directories']} dirs)",
                indent=1,
            )

            print_info("Analyzing repository structure...", indent=1)
            overview_result = get_repository_overview(str(repo_dir))
            if not overview_result["success"]:
                print_error(
                    f"Failed to get overview: {overview_result.get('error')}", indent=1
                )
                continue

            print_success(
                f"Found {len(overview_result['important_files'])} important files",
                indent=1,
            )

            # Algorithm analysis iteration
            algo_prompt = f"""Analyze this code repository in the context of the research idea.

Research Idea Summary:
{input_data[:2000]}

Repository: {repo_dir.name}

Code Tree:
{tree_result['tree_text']}

Important Files:
{str(overview_result['important_files'][:10])}

Previous accumulated algorithms:
{accumulated_algorithm if accumulated_algorithm else 'None yet'}

Extract algorithm patterns and implementations. Build upon previous analysis."""

            try:
                print_info("Analyzing algorithm implementations...", indent=1)
                # Use streamed version for real-time output
                algo_stream = Runner.run_streamed(
                    self.paper_algorithm_analyzer,
                    algo_prompt,
                    hooks=self.hooks,
                    max_turns=100,
                )
                async for event in algo_stream.stream_events():
                    if hasattr(event, "data"):
                        event_type = type(event.data).__name__
                        if "FunctionCallArguments" not in event_type and hasattr(
                            event.data, "delta"
                        ):
                            delta = event.data.delta
                            if hasattr(delta, "content") and delta.content:
                                print(delta.content, end="", flush=True)
                            elif hasattr(delta, "text") and delta.text:
                                print(delta.text, end="", flush=True)
                algo_result = algo_stream  # The stream object is the result
                algorithm_analysis = algo_result.final_output

                # Add to accumulated algorithms
                algo_text = algorithm_analysis.algorithms
                accumulated_algorithm += f"\n\n### From {repo_dir.name}:\n{algo_text}"

                print_success(
                    f"Algorithm analysis completed ({len(algo_text)} chars)", indent=1
                )
                print_result_box("Algorithm Patterns Found", algo_text, max_length=300)

                # Rate limiting: wait between API calls
                if i < len(repo_dirs[:2]) - 1:  # Don't wait after last repo
                    print_info(
                        f"Waiting 10 seconds to avoid rate limiting...", indent=1
                    )
                    await asyncio.sleep(10)

            except Exception as e:
                print_error(f"Algorithm analysis failed: {e}", indent=1)

        return accumulated_algorithm

    async def analyze(
        self, input_data: str, input_path: Optional[str] = None
    ) -> PreAnalysisOutput:
        """
        Analyze research input (paper or idea) and produce unified output.

        Args:
            input_data: The input content (file path or direct content)
            input_path: Optional file path for context

        Returns:
            PreAnalysisOutput with comprehensive analysis
        """
        # Step 1: Detect input type
        input_type = self._detect_input_type(input_data)

        print_section("PRE-ANALYSIS WORKFLOW", "=")
        print_info(
            f"Input type detected: {Colors.BOLD}{input_type.upper()}{Colors.ENDC}"
        )
        print_info(f"Input length: {len(input_data)} characters")

        # Skip iterative analysis - analyzers will work with provided input only
        print_info("Skipping paper and repository analysis (tools disabled)")

        # Step 2: Run main analyzers directly with input data
        if input_type == "paper":
            concept_prompt = (
                f"Extract conceptual framework from this paper:\n{input_data}"
            )
        else:  # idea
            concept_prompt = (
                f"Elaborate conceptual framework for this research idea:\n{input_data}"
            )

        print_subsection("Main Concept Analysis")
        print_info("Running concept analysis...")

        # Use streamed version for real-time output
        concept_stream = Runner.run_streamed(
            (
                self.idea_concept_analyzer
                if input_type == "idea"
                else self.paper_concept_analyzer
            ),
            concept_prompt,
            hooks=self.hooks,
            max_turns=100,
        )
        async for event in concept_stream.stream_events():
            # Only print text content, skip tool call arguments
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type:
                    if hasattr(event.data, "delta"):
                        delta = event.data.delta
                        if hasattr(delta, "content") and delta.content:
                            print(delta.content, end="", flush=True)
                        elif hasattr(delta, "text") and delta.text:
                            print(delta.text, end="", flush=True)
        concept_result = concept_stream  # The stream object is the result
        concept_analysis = concept_result.final_output

        print_success(f"Concept analysis completed")
        print_result_box(
            "System Architecture", concept_analysis.system_architecture, max_length=2000
        )

        # Rate limiting: wait before next major analysis
        print_info(f"Waiting 10 seconds before algorithm analysis...", indent=0)
        await asyncio.sleep(10)

        # Step 3: Algorithm analysis
        if input_type == "paper":
            algo_prompt = f"Extract algorithms from this paper:\n{input_data}"
        else:  # idea
            algo_prompt = (
                f"Generate algorithmic specifications for this idea:\n{input_data}"
            )

        print_subsection("Main Algorithm Analysis")
        print_info("Running algorithm analysis...")

        # Use streamed version for real-time output
        algorithm_stream = Runner.run_streamed(
            (
                self.idea_algorithm_analyzer
                if input_type == "idea"
                else self.paper_algorithm_analyzer
            ),
            algo_prompt,
            hooks=self.hooks,
            max_turns=100,
        )
        async for event in algorithm_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type and hasattr(
                    event.data, "delta"
                ):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
        algorithm_result = algorithm_stream  # The stream object is the result
        algorithm_analysis = algorithm_result.final_output

        print_success(f"Algorithm analysis completed")
        print_result_box(
            "Core Algorithms", algorithm_analysis.algorithms, max_length=2000
        )

        # Rate limiting: wait before output unification
        print_info(f"Waiting 10 seconds before output unification...", indent=0)
        await asyncio.sleep(10)

        # Step 5: Unify outputs
        unified_input = f"""
Input Type: {input_type}

=== CONCEPT ANALYSIS ===
System Architecture: {concept_analysis.system_architecture}

Conceptual Framework: {concept_analysis.conceptual_framework}

Design Philosophy: {concept_analysis.design_philosophy}

Key Innovations: {concept_analysis.key_innovations}

Theoretical Basis: {concept_analysis.theoretical_basis}

=== ALGORITHM ANALYSIS ===
Algorithms: {algorithm_analysis.algorithms}

Mathematical Formulations: {algorithm_analysis.mathematical_formulations}

Technical Details: {algorithm_analysis.technical_details}

Computational Methods: {algorithm_analysis.computational_methods}

Algorithm Flow: {algorithm_analysis.algorithm_flow}
"""

        print_subsection("Unifying Analysis Results")
        print_info("Synthesizing concept and algorithm analysis...")

        # Use streamed version for real-time output
        unified_stream = Runner.run_streamed(
            self.output_unifier, unified_input, hooks=self.hooks, max_turns=100
        )
        async for event in unified_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type and hasattr(
                    event.data, "delta"
                ):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
        unified_result = unified_stream  # The stream object is the result

        print_success("Output unification completed")

        # Display unified result
        unified_output = unified_result.final_output
        print_result_box(
            "Unified Analysis Summary", unified_output.summary, max_length=3000
        )

        print_success("Pre-analysis completed successfully!")
        print_section("PRE-ANALYSIS COMPLETE", "=")

        return unified_result.final_output

    def analyze_sync(
        self, input_data: str, input_path: Optional[str] = None
    ) -> PreAnalysisOutput:
        """
        Synchronous version of analyze method.

        Args:
            input_data: The input content (file path or direct content)
            input_path: Optional file path for context

        Returns:
            PreAnalysisOutput with comprehensive analysis
        """
        import asyncio

        return asyncio.run(self.analyze(input_data, input_path))


def create_pre_analysis_agent(
    model: str = "gpt-4o",
    tools: Optional[Dict[str, list]] = None,
    workspace_dir: Optional[str] = None,
    verbose: bool = False,
) -> PreAnalysisAgent:
    """
    Factory function to create a pre-analysis agent system.

    Args:
        model: Model to use for all agents
        tools: Dictionary mapping agent types to their tools
        workspace_dir: Workspace directory path
        verbose: If True, enable verbose hooks to show full LLM responses and tool calls

    Returns:
        PreAnalysisAgent instance
    """
    return PreAnalysisAgent(
        model=model, tools=tools, workspace_dir=workspace_dir, verbose=verbose
    )


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create the pre-analysis agent
        agent = create_pre_analysis_agent(model="gpt-4o")

        # Example with a paper
        paper_content = "\\documentclass{article}..."
        result = await agent.analyze(paper_content)

        print("Analysis Result:")
        print(f"Input Type: {result.input_type}")
        print(f"Summary: {result.summary}")
        print(f"Implementation Guidance: {result.implementation_guidance}")

    asyncio.run(main())
