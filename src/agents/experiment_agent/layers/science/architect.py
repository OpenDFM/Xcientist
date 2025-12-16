"""
Experiment Architect Agent - Experiment Design

Based on the paper "Towards a Science of Scaling Agent Systems":
- Handles the "high sequential dependency" phase of experiment design
- Uses the strongest model available for global consistency
- Outputs a detailed ExperimentPlan
"""

import logging
import os
from typing import Optional, List, Any

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.layers.science.schemas.experiment import ExperimentPlan, ExperimentTask
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint
from src.agents.experiment_agent.layers.code.schemas.manifest import CodeManifest
from src.agents.experiment_agent.layers.code.schemas.proposal import Proposal
from src.agents.experiment_agent.shared.tools.core import get_architect_tools
from src.agents.experiment_agent.shared.tools.parsing import extract_json_from_llm_output
from src.agents.experiment_agent.shared.utils.cache import Cache
from src.agents.experiment_agent.shared.utils.config import SCIENCE_ARCHITECT_MODEL
from src.agents.experiment_agent.shared.utils.prompts import load_and_render_prompt


logger = logging.getLogger(__name__)


class ExpArchitectAgent(BaseAgent):
    """
    Senior Experimental Architect Agent.

    Designs experiment plans based on research proposals and code manifests.
    """

    def __init__(
        self,
        model: str = SCIENCE_ARCHITECT_MODEL,
        verbose: bool = True,
    ):
        super().__init__(
            agent_type="ExpArchitect",
            model=model,
            max_turns=10000,
            verbose=verbose,
        )

    def _get_tools(self) -> List:
        return get_architect_tools()

    async def design_experiments(
        self,
        manifest: CodeManifest,
        proposal: Proposal,
        code_blueprint: Optional[Blueprint] = None,
        project_root: Optional[str] = None,
        previous_results: Optional[List[Any]] = None,
        experiment_id: Optional[str] = None,
        dataset_dir: Optional[str] = None,
    ) -> ExperimentPlan:
        """
        Design a comprehensive experiment plan.

        Args:
            manifest: CodeManifest from CodeLayer
            proposal: The research proposal
            blueprint_id: Hash/ID to load Code Layer Blueprint from Cache (preferred)
            project_root: Project root path (preferred; falls back to manifest.project_root)
            previous_results: Optional results from previous experiments

        Returns:
            ExperimentPlan object with tasks to execute
        """
        self._log_info("Designing experiment plan...")
        effective_project_root = project_root or manifest.project_root
        self._log_info(f"Project: {effective_project_root}")
        self._log_info(f"Entry Point: {manifest.entry_point}")
        self._log_info(f"Goal: {proposal.idea.title}")

        blueprint = code_blueprint
        project_skeleton = self._build_project_skeleton(blueprint) if blueprint else ""

        # Build prompts
        system_prompt = self._build_system_prompt(
            manifest=manifest,
            project_root=effective_project_root,
            project_skeleton=project_skeleton,
            experiment_id=experiment_id,
            dataset_dir=dataset_dir,
        )
        user_prompt = self._build_user_prompt(
            manifest=manifest,
            proposal=proposal,
            project_skeleton=project_skeleton,
            previous_results=previous_results,
            experiment_id=experiment_id,
            dataset_dir=dataset_dir,
        )

        # Run agent
        result = await self._run_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=self._get_tools(),
        )

        # Extract plan
        plan = self._extract_plan(result)

        self._log_success(f"Plan created with {len(plan.tasks)} tasks")
        self._log_info(f"Analysis Goal: {plan.analysis_goal}")

        return plan

    def _build_project_skeleton(self, blueprint: Blueprint) -> str:
        """
        Build a compact project skeleton summary from the Code Blueprint.

        This is a guide for what to inspect with tools; parameter values MUST be
        discovered by reading code / running --help, not guessed.
        """
        builder = PromptBuilder()
        builder.add_header("Code Blueprint (Project Skeleton)")
        builder.add_key_value("Entry Point", blueprint.entry_point)
        builder.add_key_value("Total Files", str(len(blueprint.file_tree)))
        builder.add_text("")

        max_tree = 60
        if blueprint.file_tree:
            builder.add_header("File Tree (truncated)", level=2)
            builder.add_list(blueprint.file_tree[:max_tree])
            if len(blueprint.file_tree) > max_tree:
                builder.add_text(
                    f"... ({len(blueprint.file_tree) - max_tree} more files)"
                )

        max_specs = 30
        if blueprint.files:
            builder.add_header("Key File Specs (truncated)", level=2)
            for fs in blueprint.files[:max_specs]:
                builder.add_header(fs.file_path, level=3)
                builder.add_text(fs.description)
                if fs.dependencies:
                    deps = ", ".join(fs.dependencies[:8]) + (
                        " ..." if len(fs.dependencies) > 8 else ""
                    )
                    builder.add_text(f"Dependencies: {deps}")
                if fs.classes:
                    classes = ", ".join([c.name for c in fs.classes[:8]]) + (
                        " ..." if len(fs.classes) > 8 else ""
                    )
                    builder.add_text(f"Classes: {classes}")
                if fs.functions:
                    funcs = ", ".join([f.name for f in fs.functions[:10]]) + (
                        " ..." if len(fs.functions) > 10 else ""
                    )
                    builder.add_text(f"Functions: {funcs}")
                builder.add_text("")
            if len(blueprint.files) > max_specs:
                builder.add_text(
                    f"... ({len(blueprint.files) - max_specs} more file specs)"
                )

        return builder.build()

    def _build_system_prompt(
        self,
        manifest: CodeManifest = None,
        project_root: Optional[str] = None,
        project_skeleton: str = "",
        **kwargs,
    ) -> str:
        """Build the system prompt for the experiment architect agent."""
        dataset_dir = kwargs.get("dataset_dir")
        experiment_id = kwargs.get("experiment_id")

        scripts_info = ""
        if manifest and manifest.scripts:
            scripts_list = "\n".join(
                f"  - {name}: `{cmd}`" for name, cmd in manifest.scripts.items()
            )
            scripts_info = f"\n**AVAILABLE SCRIPTS:**\n{scripts_list}\n"

        project_root = project_root or (manifest.project_root if manifest else ".")
        entry_point = manifest.entry_point if manifest else "main.py"
        config_file = manifest.config_file if manifest else "Not specified"
        skeleton_block = f"\n{project_skeleton}\n" if project_skeleton else ""

        dataset_context_str = ""
        if dataset_dir:
            dataset_context_str = f"""
**RUNTIME DATASET CONTEXT (do NOT hard-code paths):**
- experiment_id: {experiment_id or ""}
- dataset_dir: {dataset_dir}

**REQUIRED (before you finalize the plan):** Inspect `dataset_dir` with 1-2 tool calls:
- `bash("ls -la {dataset_dir}")`
- `bash("find {dataset_dir} -maxdepth 2 -type f | head -50")`

Let the observed formats/splits drive the data loading commands and evaluation checks.
"""
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "exp_architect",
            "system.txt",
        )
        return load_and_render_prompt(
            prompt_path=prompt_path,
            variables={
                "dataset_context_str": (
                    dataset_context_str.strip("\n") if dataset_context_str else ""
                ),
                "project_root": str(project_root),
                "entry_point": str(entry_point),
                "config_file": str(config_file),
                "scripts_info": scripts_info.strip("\n") if scripts_info else "",
                "skeleton_block": skeleton_block.strip("\n") if skeleton_block else "",
            },
        )

    def _build_user_prompt(
        self,
        manifest: CodeManifest = None,
        proposal: Proposal = None,
        project_skeleton: str = "",
        previous_results: Optional[List[Any]] = None,
        **kwargs,
    ) -> str:
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
            builder.add_text("")

        builder.add_header("Research Proposal")
        builder.add_section("Title", proposal.idea.title)
        builder.add_section("Description", proposal.idea.description)

        builder.add_header("Key Innovations to Validate", level=2)
        builder.add_list(proposal.idea.key_innovations, ordered=True)

        builder.add_header("Expected Outcomes", level=2)
        builder.add_list(proposal.idea.expected_outcomes)

        if project_skeleton:
            builder.add_separator()
            builder.add_header("Code Blueprint (Injected)", level=2)
            builder.add_text(
                "The Code Blueprint content is provided below. Use tools to read real code before deciding parameters."
            )
            builder.add_text("")
            builder.add_text(project_skeleton.strip())

        # Previous results context
        if previous_results:
            builder.add_separator()
            builder.add_header("Previous Experiment Results")
            builder.add_text("The following experiments have been run:")
            builder.add_text("")

            for result in previous_results:
                if hasattr(result, "task_id"):
                    status = "✓ Success" if result.success else "✗ Failed"
                    builder.add_text(f"- **{result.task_id}**: {status}")
                    if hasattr(result, "metrics") and result.metrics:
                        metrics_str = ", ".join(
                            f"{k}={v:.4f}" for k, v in result.metrics.items()
                        )
                        builder.add_text(f"  Metrics: {metrics_str}")

            builder.add_text("")
            builder.add_text(
                "**Task:** Design FOLLOW-UP experiments based on these results."
            )
        else:
            builder.add_separator()
            builder.add_header("Your Task", level=2)
            builder.add_list(
                [
                    "If `Dataset Directory` is provided, inspect it first with 1-2 tool calls and let the observed data format drive the experiment design",
                    "Explore the codebase using the tools",
                    "Design 2-5 key experiments to validate the core claims",
                    "Include a baseline experiment for comparison",
                    "Output the ExperimentPlan as JSON",
                ],
                ordered=True,
            )

        return builder.build()

    def _extract_plan(self, result) -> ExperimentPlan:
        """Extract ExperimentPlan from agent result."""
        json_data = self._extract_json(result)

        if json_data is None:
            self._log_warning("Could not extract ExperimentPlan JSON from agent output")
            return ExperimentPlan(
                tasks=[], analysis_goal="Failed to parse experiment plan"
            )

        try:
            plan = ExperimentPlan(**json_data)
            return plan
        except Exception as e:
            logger.warning(f"Error validating ExperimentPlan: {e}")

            # Try to create with partial data
            tasks = []
            if "tasks" in json_data:
                for task_data in json_data["tasks"]:
                    try:
                        tasks.append(ExperimentTask(**task_data))
                    except Exception:
                        pass

            return ExperimentPlan(
                tasks=tasks,
                analysis_goal=json_data.get("analysis_goal", "Partially parsed plan"),
            )
