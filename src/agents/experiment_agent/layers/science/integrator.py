"""
Experiment Integrator Agent - Result Analysis & Verification

Based on the paper "Towards a Science of Scaling Agent Systems":
- Returns to single-agent mode for global result analysis
- Has full view of all experiment results
- Performs statistical analysis and draws conclusions
- Produces corrected experiment blueprints (ExperimentPlan) for the next iteration if needed
"""

import logging
import os
import json
import re
from typing import Dict, List, Optional, Any

from src.agents.experiment_agent.layers.base.agent import BaseAgent, PromptBuilder
from src.agents.experiment_agent.layers.science.schemas.experiment import (
    ExperimentResult,
    ExperimentPlan,
    ExperimentTask,
    ScienceAnalysis,
)
from src.agents.experiment_agent.layers.code.schemas.proposal import Proposal
from src.agents.experiment_agent.layers.code.schemas.blueprint import Blueprint
from src.agents.experiment_agent.shared.tools.core import get_integrator_tools
from src.agents.experiment_agent.shared.tools.parsing import (
    extract_json_from_llm_output,
)
from src.agents.experiment_agent.shared.utils.config import SCIENCE_INTEGRATOR_MODEL
from src.agents.experiment_agent.shared.utils.prompts import load_and_render_prompt


logger = logging.getLogger(__name__)


class ExpIntegratorAgent(BaseAgent):
    """
    Experimental Integrator (Senior Scientist) Agent.

    Analyzes experiment results and draws conclusions.
    """

    def __init__(
        self,
        model: str = SCIENCE_INTEGRATOR_MODEL,
        verbose: bool = True,
    ):
        super().__init__(
            agent_type="ExpScientist",
            model=model,
            max_turns=10000,
            verbose=verbose,
        )

    def _get_tools(self) -> List:
        return get_integrator_tools()

    def _normalize_analysis(
        self,
        analysis: ScienceAnalysis,
    ) -> ScienceAnalysis:
        # Ensure key_findings is a list[str]
        key_findings = getattr(analysis, "key_findings", None)
        if not isinstance(key_findings, list):
            key_findings = []
        key_findings = [str(x) for x in key_findings if x is not None]

        summary = str(getattr(analysis, "summary", "") or "").strip()
        if not summary:
            summary = "Analysis completed."

        next_experiments = getattr(analysis, "next_experiments", None)
        if next_experiments is not None and not isinstance(
            next_experiments, ExperimentPlan
        ):
            next_experiments = None

        verdict = getattr(analysis, "verdict", None)
        if verdict is not None:
            verdict = str(verdict).strip() or None

        report_md = str(getattr(analysis, "report_md", "") or "")
        feedback_md = str(getattr(analysis, "feedback_md", "") or "")

        return ScienceAnalysis(
            success=bool(getattr(analysis, "success", False)),
            verdict=verdict,
            summary=summary,
            key_findings=key_findings,
            report_md=report_md,
            feedback_md=feedback_md,
            optimization_tickets=list(
                getattr(analysis, "optimization_tickets", []) or []
            ),
            next_experiments=next_experiments,
        )

    async def analyze_results(
        self,
        results: List[ExperimentResult],
        goal: str,
        project_root: Optional[str] = None,
        proposal: Optional[Proposal] = None,
        code_blueprint: Optional[Blueprint] = None,
        plan: Optional[ExperimentPlan] = None,
        doc_paths: Optional[Dict[str, str]] = None,
    ) -> ScienceAnalysis:
        """
        Analyze experimental results and draw conclusions.

        Args:
            results: List of experiment results
            goal: The analysis goal
            project_root: Optional project root for examining artifacts
            proposal: Optional research proposal / idea input (upstream context)
            code_blueprint: Optional code blueprint from Code Architect
            plan: Optional experiment plan/blueprint (tasks + commands)

        Returns:
            ScienceAnalysis with conclusions and recommendations
        """
        self._log_info("Analyzing experiment results...")
        success_count = sum(1 for r in results if r.success)
        self._log_info(
            f"Total experiments: {len(results)}, Successful: {success_count}"
        )
        self._log_info(f"Goal: {goal}")

        # Prefer persisted results under result_dir for robustness (resume-friendly).
        if project_root:
            results = self._hydrate_results_from_result_dirs(results, project_root)

        # Build prompts
        system_prompt = self._build_system_prompt(project_root=project_root)
        user_prompt = self._build_user_prompt(
            results=results,
            goal=goal,
            proposal=proposal,
            code_blueprint=code_blueprint,
            plan=plan,
            doc_paths=doc_paths,
        )

        # Run agent
        result = await self._run_agent(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            tools=self._get_tools() if project_root else [],
        )

        # Extract analysis
        analysis = self._normalize_analysis(self._extract_analysis(result))

        self._log_success("Analysis complete")
        self._log_info(f"Success: {analysis.success}")
        self._log_info(f"Key findings: {len(analysis.key_findings)}")
        if analysis.next_experiments:
            self._log_info(
                f"Corrected experiment plan proposed: {len(analysis.next_experiments.tasks)} tasks"
            )

        return analysis

    async def analyze_iteration_dir(
        self,
        goal: str,
        project_root: str,
        iteration_result_dir: str,
        proposal: Optional[Proposal] = None,
        doc_paths: Optional[Dict[str, str]] = None,
    ) -> ScienceAnalysis:
        """
        Analyze a Markdown-driven iteration.

        Inputs are primarily:
        - iteration_result_dir (must contain result_summary.json and per-task run folders)
        - constraint docs (idea/spec/plan/tasks) via doc_paths
        """
        builder = PromptBuilder()
        builder.add_header("Science Iteration Analysis Request")
        builder.add_section("Analysis Goal", str(goal or "").strip())
        builder.add_text("")
        builder.add_key_value("Project Root", str(project_root))
        builder.add_key_value("Iteration Result Dir", str(iteration_result_dir))

        if doc_paths:
            builder.add_header(
                "Science Constraint Docs (absolute; read when needed)", level=2
            )
            for k in [
                "idea_path",
                "spec_path",
                "plan_path",
                "tasks_path",
                "prev_report_path",
                "prev_feedback_path",
            ]:
                if doc_paths.get(k):
                    builder.add_text(f"- **{k}**: `{doc_paths.get(k)}`")
            builder.add_text("")

        if proposal is not None:
            try:
                proposal_data = (
                    proposal.model_dump()
                    if hasattr(proposal, "model_dump")
                    else dict(proposal)
                )
            except Exception:
                proposal_data = {"proposal": str(proposal)}
            builder.add_header("Idea / Proposal (Input Context)", level=2)
            builder.add_code(
                json.dumps(proposal_data, ensure_ascii=False, indent=2), language="json"
            )

        builder.add_separator()
        builder.add_header("Your Task", level=2)
        builder.add_text(
            "**Context**: Dataset files are located in `<workspace>/dataset_candidate/` directory."
        )
        builder.add_text(
            "**Context**: If experiments need GPU, they should check `nvidia-smi` first and use CUDA_VISIBLE_DEVICES to select the GPU with most available memory."
        )
        builder.add_text("")
        builder.add_list(
            [
                "Use tools to inspect iteration_result_dir (especially result_summary.json and per-task logs).",
                "Write a concise report in report_md with evidence links (file paths) and key metrics.",
                "Decide success (goal achieved) and set verdict supported/refuted/inconclusive.",
                "If not successful, write actionable next-iteration instructions in feedback_md (Markdown). Your feedback can include:",
                "  - Experimental design improvements (better hyperparameters, additional experiments, etc.)",
                "  - Code improvements needed (bug fixes, missing features, performance optimizations, etc.)",
                "  - Be specific: mention file paths, function names, and concrete suggestions",
                "Always set optimization_tickets to [] and next_experiments to null.",
            ],
            ordered=True,
        )

        result = await self._run_agent(
            user_prompt=builder.build(),
            system_prompt=self._build_system_prompt(project_root=project_root),
            tools=self._get_tools(),
        )
        return self._normalize_analysis(self._extract_analysis(result))

    def _hydrate_results_from_result_dirs(
        self, results: List[ExperimentResult], project_root: str
    ) -> List[ExperimentResult]:
        """
        Hydrate/override metrics (and optionally error) from files written in each task's result_dir:
        - metrics_{task_id}.json (preferred, written by ExpWorker)
        - meta_{task_id}.json (preferred, written by ExpWorker)
        - stdout_{task_id}.txt / stderr_{task_id}.txt (preferred)

        Backward-compatible fallback:
        - metrics.json
        - meta.json
        - stdout.txt / stderr.txt
        """

        def _safe_task_id(task_id: str) -> str:
            return re.sub(r"[^A-Za-z0-9._-]+", "_", str(task_id or "task"))

        def _read_text(path: str) -> str:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                return ""

        def _read_json(path: str) -> Any:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

        hydrated: List[ExperimentResult] = []
        for r in results:
            result_dir = getattr(r, "result_dir", None) or None
            if not result_dir:
                hydrated.append(r)
                continue

            abs_dir = os.path.join(project_root, result_dir)
            if not os.path.isdir(abs_dir):
                hydrated.append(r)
                continue

            metrics = dict(r.metrics or {})
            error = r.error
            stdout = r.stdout or ""
            stderr = r.stderr or ""
            safe_id = _safe_task_id(r.task_id)

            # Metrics: prefer per-task file written by ExpWorker, then fallback.
            metrics_candidates = [
                os.path.join(abs_dir, f"metrics_{safe_id}.json"),
                os.path.join(abs_dir, "metrics.json"),
            ]
            for metrics_path in metrics_candidates:
                if not os.path.exists(metrics_path):
                    continue
                data = _read_json(metrics_path)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            metrics[k] = float(v)
                    break

            # Meta: prefer per-task file written by ExpWorker, then fallback.
            meta_candidates = [
                os.path.join(abs_dir, f"meta_{safe_id}.json"),
                os.path.join(abs_dir, "meta.json"),
            ]
            for meta_path in meta_candidates:
                if not os.path.exists(meta_path):
                    continue
                meta = _read_json(meta_path)
                if isinstance(meta, dict):
                    # Only fill error if missing; do not override success/task_id
                    if (not error) and meta.get("error"):
                        error = str(meta.get("error"))
                    break

            # Stdout/stderr: prefer per-task files written by ExpWorker, then fallback.
            if not stdout:
                stdout_candidates = [
                    os.path.join(abs_dir, f"stdout_{safe_id}.txt"),
                    os.path.join(abs_dir, "stdout.txt"),
                ]
                for p in stdout_candidates:
                    if os.path.exists(p):
                        stdout = _read_text(p)
                        break

            if not stderr:
                stderr_candidates = [
                    os.path.join(abs_dir, f"stderr_{safe_id}.txt"),
                    os.path.join(abs_dir, "stderr.txt"),
                ]
                for p in stderr_candidates:
                    if os.path.exists(p):
                        stderr = _read_text(p)
                        break

            hydrated.append(
                ExperimentResult(
                    task_id=r.task_id,
                    success=r.success,
                    metrics=metrics,
                    artifacts=list(r.artifacts or []),
                    result_dir=result_dir,
                    error=error,
                    stdout=stdout,
                    stderr=stderr,
                )
            )

        return hydrated

    async def quick_analyze(
        self,
        results: List[ExperimentResult],
        goal: str,
        **kwargs,
    ) -> ScienceAnalysis:
        """Quick analysis without agent (for simple cases)."""
        _ = kwargs
        success_count = sum(1 for r in results if r.success)
        total_count = len(results)

        # Aggregate metrics
        all_metrics: Dict[str, List[float]] = {}
        for result in results:
            for key, value in result.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        # Calculate statistics
        key_findings = []
        for metric_name, values in all_metrics.items():
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                key_findings.append(
                    f"{metric_name}: avg={avg:.4f}, min={min_val:.4f}, max={max_val:.4f}"
                )

        success = success_count == total_count

        if success:
            summary = f"All {total_count} experiments completed successfully."
        else:
            summary = f"{success_count}/{total_count} experiments succeeded."

        return ScienceAnalysis(
            success=success,
            summary=summary,
            key_findings=key_findings,
            optimization_tickets=[],
            next_experiments=None,
        )

    def _build_system_prompt(self, project_root: Optional[str] = None, **kwargs) -> str:
        """Build the system prompt for the scientist agent."""
        tools_section = ""
        if project_root:
            tools_section = f"""
**TOOLS:**
- `bash(command)`: Execute shell commands (for data analysis, plotting, etc.)
- `file_viewer(file_path, start_line, end_line)`: View result files

**PROJECT ROOT:** {project_root}
You can examine result files using these tools if needed.
"""
        prompt_path = os.path.join(
            os.path.dirname(__file__),
            "prompts",
            "exp_integrator",
            "system.txt",
        )
        return load_and_render_prompt(
            prompt_path=prompt_path,
            variables={
                "tools_section": tools_section.strip("\n") if tools_section else ""
            },
        )

    def _build_user_prompt(
        self,
        results: List[ExperimentResult] = None,
        goal: str = "",
        proposal: Optional[Proposal] = None,
        code_blueprint: Optional[Blueprint] = None,
        plan: Optional[ExperimentPlan] = None,
        doc_paths: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> str:
        """Build the user prompt with proposal + code blueprint + experiment results."""
        builder = PromptBuilder()

        builder.add_header("Experiment Analysis Request")
        builder.add_section("Analysis Goal", goal)

        if doc_paths:
            builder.add_header(
                "Science Constraint Docs (absolute; read when needed)", level=2
            )
            for k in [
                "idea_path",
                "spec_path",
                "plan_path",
                "tasks_path",
                "prev_report_path",
                "prev_feedback_path",
            ]:
                if doc_paths.get(k):
                    builder.add_text(f"- **{k}**: `{doc_paths.get(k)}`")
            builder.add_text("")

        # High-level context
        if proposal is not None:
            try:
                proposal_data = (
                    proposal.model_dump()
                    if hasattr(proposal, "model_dump")
                    else dict(proposal)
                )
            except Exception:
                proposal_data = {"proposal": str(proposal)}
            builder.add_header("Idea / Proposal (Input Context)", level=2)
            builder.add_code(
                json.dumps(proposal_data, ensure_ascii=False, indent=2), language="json"
            )

        if code_blueprint is not None:
            try:
                bp_data = (
                    code_blueprint.model_dump()
                    if hasattr(code_blueprint, "model_dump")
                    else dict(code_blueprint)
                )
            except Exception:
                bp_data = {"blueprint": str(code_blueprint)}
            builder.add_header("Code Blueprint (from Code Architect)", level=2)
            # Provide a compact but useful view: entry_point + file_tree (truncated).
            entry_point = bp_data.get("entry_point", "")
            file_tree = bp_data.get("file_tree", []) or []
            builder.add_key_value("Entry Point", str(entry_point))
            builder.add_key_value("Files in Blueprint", str(len(file_tree)))
            preview = file_tree[:80]
            builder.add_text("**File Tree Preview (first 80):**")
            for p in preview:
                builder.add_text(f"- {p}")
            builder.add_text("")

        if plan is not None:
            try:
                plan_data = (
                    plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
                )
            except Exception:
                plan_data = {"plan": str(plan)}
            builder.add_header(
                "Experiment Plan / Blueprint (from Science Manager)", level=2
            )
            builder.add_code(
                json.dumps(plan_data, ensure_ascii=False, indent=2), language="json"
            )

        builder.add_header("Experiment Results Summary", level=2)

        success_count = sum(1 for r in results if r.success)
        builder.add_key_value("Total Experiments", str(len(results)))
        builder.add_key_value("Successful", str(success_count))
        builder.add_key_value("Failed", str(len(results) - success_count))
        builder.add_text("")

        builder.add_header("Detailed Results", level=2)

        for result in results:
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            builder.add_header(f"{result.task_id}: {status}", level=3)

            if result.metrics:
                builder.add_text("**Metrics:**")
                for key, value in result.metrics.items():
                    builder.add_text(f"  - {key}: {value:.6f}")
                builder.add_text("")

            if result.artifacts:
                builder.add_text(f"**Artifacts:** {', '.join(result.artifacts)}")
                builder.add_text("")

            if result.error:
                builder.add_text(f"**Error:** {result.error}")
                builder.add_text("")

            # Include short stdout/stderr snippets to help root-cause analysis without flooding context.
            if result.stdout:
                snippet = (result.stdout or "")[:1500]
                builder.add_text("**Stdout (head):**")
                builder.add_code(snippet, language="")
            if result.stderr:
                snippet = (result.stderr or "")[:1500]
                builder.add_text("**Stderr (head):**")
                builder.add_code(snippet, language="")

        # Metrics comparison table
        all_metrics: Dict[str, Dict[str, float]] = {}
        for result in results:
            if result.metrics:
                all_metrics[result.task_id] = result.metrics

        if all_metrics:
            builder.add_header("Metrics Comparison Table", level=2)

            metric_names = set()
            for metrics in all_metrics.values():
                metric_names.update(metrics.keys())

            sorted_metrics = sorted(metric_names)
            header = "| Task ID | " + " | ".join(sorted_metrics) + " |"
            separator = "|---" * (len(sorted_metrics) + 1) + "|"

            builder.add_text(header)
            builder.add_text(separator)

            for task_id, metrics in all_metrics.items():
                row = f"| {task_id} |"
                for metric in sorted_metrics:
                    value = metrics.get(metric, "-")
                    if isinstance(value, float):
                        row += f" {value:.4f} |"
                    else:
                        row += f" {value} |"
                builder.add_text(row)
            builder.add_text("")

        builder.add_separator()
        builder.add_header("Your Task", level=2)
        builder.add_text(
            "**Context**: Dataset files are located in `<workspace>/dataset_candidate/` directory."
        )
        builder.add_text(
            "**Context**: If experiments need GPU, they should check `nvidia-smi` first and use CUDA_VISIBLE_DEVICES to select the GPU with most available memory."
        )
        builder.add_text("")
        builder.add_list(
            [
                "Analyze the experiment results comprehensively",
                "Determine if the analysis goal was achieved",
                "Identify key findings and patterns",
                "If experiments failed or goal not achieved, you can suggest both experimental AND code improvements:",
                "  - Better experimental design (hyperparameters, metrics, etc.)",
                "  - Code improvements (bug fixes, missing features, performance issues)",
                "  - Be specific: file paths, function names, concrete suggestions",
                "Do NOT populate optimization_tickets (deprecated). Instead use feedback in report/feedback_md.",
                "Output the ScienceAnalysis as JSON",
            ],
            ordered=True,
        )

        return builder.build()

    def _extract_analysis(self, result) -> ScienceAnalysis:
        """Extract ScienceAnalysis from agent result."""
        json_data = self._extract_json(result)

        if json_data is None:
            self._log_warning(
                "Could not extract ScienceAnalysis JSON from agent output"
            )
            output = self._extract_output(result)
            output_lower = output.lower()
            success = "success" in output_lower and "fail" not in output_lower

            return ScienceAnalysis(
                success=success,
                summary="Analysis completed but JSON parsing failed.",
                key_findings=[],
                optimization_tickets=[],
            )

        try:
            # Handle next_experiments if present
            if "next_experiments" in json_data and json_data["next_experiments"]:
                try:
                    json_data["next_experiments"] = ExperimentPlan(
                        **json_data["next_experiments"]
                    )
                except Exception:
                    json_data["next_experiments"] = None

            analysis = ScienceAnalysis(**json_data)
            return analysis
        except Exception as e:
            logger.warning(f"Error validating ScienceAnalysis: {e}")
            return ScienceAnalysis(
                success=json_data.get("success", False),
                summary=json_data.get("summary", "Partial analysis available"),
                key_findings=json_data.get("key_findings", []),
                optimization_tickets=[],
                next_experiments=None,
            )

    def compare_experiments(
        self,
        results: List[ExperimentResult],
        baseline_id: str,
    ) -> Dict[str, Dict[str, float]]:
        """Compare all experiments against a baseline."""
        comparisons = {}

        baseline = next((r for r in results if r.task_id == baseline_id), None)
        if not baseline or not baseline.metrics:
            return comparisons

        for result in results:
            if result.task_id == baseline_id or not result.metrics:
                continue

            comparison = {}
            for metric, value in result.metrics.items():
                if metric in baseline.metrics:
                    baseline_value = baseline.metrics[metric]
                    if baseline_value != 0:
                        relative_change = (
                            (value - baseline_value) / abs(baseline_value)
                        ) * 100
                        comparison[metric] = relative_change

            if comparison:
                comparisons[result.task_id] = comparison

        return comparisons
