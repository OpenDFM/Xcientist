"""
Code Architect Agent - System Design

Based on the paper "Towards a Science of Scaling Agent Systems":
- Handles the "high sequential dependency" phase
- Uses the strongest model available for global consistency
- Outputs a detailed Blueprint (not implementation code)
"""

import os
import logging
from typing import Optional, List

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint
from src.agents.experiment_agent.layers.code.schemas.proposal import Proposal
from src.agents.experiment_agent.shared.exceptions import exit_on_rate_limit
from src.agents.experiment_agent.shared.tools.core import get_architect_tools
from src.agents.experiment_agent.shared.tools.parsing import extract_json_from_llm_output
from src.agents.experiment_agent.shared.utils.config import CODE_ARCHITECT_MODEL
from src.agents.experiment_agent.shared.utils.prompts import load_and_render_prompt


logger = logging.getLogger(__name__)


class CodeArchitectAgent(BaseAgent):
    """
    Senior System Architect Agent.

    Designs complete system blueprints from research proposals.
    Uses tools to explore reference codebases for patterns.
    """

    def __init__(
        self,
        model: str = CODE_ARCHITECT_MODEL,
        verbose: bool = True,
    ):
        super().__init__(
            agent_type="CodeArchitect",
            model=model,
            max_turns=10000,
            verbose=verbose,
        )
        self.reference_repos: List[str] = []

    def add_reference_repo(self, repo_path: str):
        """Add a reference repository for the architect to learn from."""
        self.reference_repos.append(repo_path)

    def _get_tools(self) -> List:
        return get_architect_tools()

    async def create_blueprint(
        self,
        proposal: Proposal,
        reference_repos: Optional[List[str]] = None,
        experiment_id: Optional[str] = None,
        dataset_dir: Optional[str] = None,
        max_retries: int = 3,
    ) -> Blueprint:
        """
        Create a comprehensive system blueprint from a proposal.

        Args:
            proposal: The research proposal
            reference_repos: Optional list of reference repository paths
            max_retries: Maximum number of retry attempts on validation failure

        Returns:
            Blueprint object with complete system design
        """
        repos_to_use = reference_repos or self.reference_repos

        self._log_info("Analyzing proposal and designing system blueprint...")
        self._log_info(f"Title: {proposal.idea.title}")
        self._log_info(f"Key Innovations: {len(proposal.idea.key_innovations)}")
        if repos_to_use:
            self._log_info(f"Reference Repos: {len(repos_to_use)}")

        # Build prompts
        system_prompt = self._build_system_prompt(
            reference_repos=repos_to_use,
            experiment_id=experiment_id,
            dataset_dir=dataset_dir,
        )
        user_prompt = self._build_user_prompt(
            proposal=proposal,
            experiment_id=experiment_id,
            dataset_dir=dataset_dir,
        )

        last_error = None

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                self._log_warning(f"Retry attempt {attempt}/{max_retries}...")
                # Add error feedback to prompt
                retry_prompt = f"""{user_prompt}

---
⚠️ **PREVIOUS ATTEMPT FAILED - PLEASE FIX THE FOLLOWING ERRORS:**

{last_error}

**CRITICAL:** You MUST output a complete, valid JSON Blueprint with ALL required fields:
- Every `FunctionSignature` must have: `name`, `args`, `return_type`, `docstring`
- Every `ClassSignature` must have: `name`, `methods` (list of FunctionSignature), `docstring`
- Every `FileSpec` must have: `file_path`, `description`, `dependencies`

Please regenerate the complete Blueprint JSON with all required fields filled in.
"""
                user_prompt = retry_prompt

            # Run agent
            result = await self._run_agent(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                tools=self._get_tools(),
            )

            # Try to extract blueprint
            try:
                blueprint = self._extract_blueprint(result)

                # Validate DAG structure to catch circular dependencies immediately
                blueprint.validate_dag()

                self._log_success(
                    f"Blueprint created with {len(blueprint.files)} files"
                )
                return blueprint
            except Exception as e:
                last_error = str(e)

                exit_on_rate_limit(last_error)

                self._log_error(f"Blueprint validation failed: {last_error[:500]}")
                if attempt == max_retries:
                    raise ValueError(
                        f"Failed to generate valid Blueprint after {max_retries} attempts. Last error: {last_error}"
                    )

        raise ValueError("Failed to generate Blueprint")

    def _build_system_prompt(
        self, reference_repos: Optional[List[str]] = None, **kwargs
    ) -> str:
        """Build the system prompt for the architect agent."""
        dataset_dir = kwargs.get("dataset_dir")
        experiment_id = kwargs.get("experiment_id")

        reference_repos_str = ""
        if reference_repos:
            example_repo = reference_repos[0]
            repos_list = "\n".join(f"- {repo}" for repo in reference_repos)
            reference_repos_str = f"""
**REFERENCE REPOSITORIES (use FULL paths):**
{repos_list}

**TOOLS:**
- `bash(command)`: Execute shell commands (grep, tree, find, etc.)
- `file_viewer(file_path, start_line, end_line)`: View file with line numbers

**WORKFLOW: grep -> file_viewer**
IMPORTANT: Use the FULL absolute paths listed above!
Example using first repo:
1. `bash("tree -L 2 {example_repo}")` - understand overall layout
2. `bash("grep -rn 'class ' {example_repo} --include='*.py' | head -30")` - find classes
3. `file_viewer("{example_repo}/path/to/file.py", 1, 50)` - view code
"""

        dataset_context_str = ""
        if dataset_dir:
            dataset_context_str = f"""
**DATASET CONTEXT (derived from runtime `--experiment`; do NOT hard-code paths):**
- experiment_id: {experiment_id or ""}
- dataset_dir: {dataset_dir}

**REQUIRED:** Before designing the Blueprint, spend 1-2 tool calls to inspect `dataset_dir`:
- `bash("ls -la {dataset_dir}")`
- `bash("find {dataset_dir} -maxdepth 2 -type f | head -50")`
- (Optional) `bash("python -c \\"import glob,os; p=glob.glob(os.path.join(r'{dataset_dir}','**','*'), recursive=True); print('files', sum(os.path.isfile(x) for x in p));\\"")`

Use what you learn (file formats, naming conventions, splits, metadata) to design data loading, preprocessing, and evaluation interfaces in the Blueprint.
"""
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "code_architect",
            "system.txt",
        )
        return load_and_render_prompt(
            prompt_path=prompt_path,
            variables={
                "dataset_context_str": (
                    dataset_context_str.strip("\n") if dataset_context_str else ""
                ),
                "reference_repos_str": (
                    reference_repos_str.strip("\n") if reference_repos_str else ""
                ),
            },
        )

    def _build_user_prompt(self, proposal: Proposal = None, **kwargs) -> str:
        """Build the user prompt with the proposal."""
        builder = PromptBuilder()
        experiment_id = kwargs.get("experiment_id")
        dataset_dir = kwargs.get("dataset_dir")

        if experiment_id or dataset_dir:
            builder.add_header("Runtime Context", level=2)
            if experiment_id:
                builder.add_section("Experiment ID", str(experiment_id))
            if dataset_dir:
                builder.add_section(
                    "Dataset Directory (do NOT hard-code)", str(dataset_dir)
                )

        builder.add_header("Research Proposal")
        builder.add_section("Title", proposal.idea.title)
        builder.add_section("Description", proposal.idea.description)
        builder.add_header("Key Innovations", level=2)
        builder.add_list(proposal.idea.key_innovations)

        if proposal.idea.methodology:
            builder.add_header("Methodology", level=2)
            for section, content in proposal.idea.methodology.items():
                title = section.replace("_", " ").title()
                builder.add_section(title, content)

        builder.add_header("Expected Outcomes", level=2)
        builder.add_list(proposal.idea.expected_outcomes)

        if proposal.reference_papers:
            builder.add_header("Reference Papers", level=2)
            builder.add_list(proposal.reference_papers)

        builder.add_separator()
        builder.add_header("Your Task", level=2)
        builder.add_list(
            [
                "If `Dataset Directory` is provided, inspect it first with 1-2 tool calls and let the observed data format drive the Blueprint design",
                "If reference repositories are available, explore them first using the tools",
                "Design a complete system blueprint for this research proposal",
                "IMPORTANT: Your Blueprint MUST include at least one runnable end-to-end test that validates the primary entry point flow; it MUST run fast by using a tiny dataset subset or a dedicated smoke-test mode",
                "Output the Blueprint as a JSON object wrapped in ```json ... ```",
            ],
            ordered=True,
        )

        return builder.build()

    def _extract_blueprint(self, result) -> Blueprint:
        """Extract Blueprint from agent result."""
        json_data = self._extract_json(result)

        if json_data is None:
            output = self._extract_output(result)
            raise ValueError(
                f"Could not extract Blueprint JSON from output:\n{output[:1000]}"
            )

        try:
            blueprint = Blueprint(**json_data)
            return blueprint
        except Exception as e:
            logger.error(f"Error validating Blueprint: {e}")
            raise
