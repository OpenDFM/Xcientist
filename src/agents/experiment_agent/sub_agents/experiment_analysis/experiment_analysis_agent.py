"""
Experiment Analysis Agent - Analyzes experiment results and provides improvement suggestions.

This agent analyzes successful experiment execution logs, compares results with
pre-analysis and code plan expectations, and suggests improvements for both
the research idea and implementation plan.
"""

import os
from typing import Optional, Any

from agents import Agent, Runner

from src.agents.experiment_agent.sub_agents.experiment_analysis.output_schemas import (
    ExperimentAnalysisOutput,
)

from src.agents.experiment_agent.logger import create_verbose_hooks

from src.agents.experiment_agent.config import OUTPUT_UNIFIER_MODEL

from src.agents.experiment_agent.utils import *

from src.agents.experiment_agent.utils.print_utils import *


def create_analysis_agent(model: str = "gpt-4o", tools: Optional[list] = None) -> Agent:
    """
    Create the experiment analysis agent.
    """

    instructions = """You are a Principal Researcher writing a comprehensive Research Report based on experiment results.

## Your Workflow

1. **Read**: Use `read_file` to read log files and extract actual metrics
2. **Analyze**: Compare baseline vs proposed, identify patterns
3. **Write**: Create a structured research report
4. **Save**: Use `write_file` to save the report

## Report Structure

# [Title]

## Abstract
Brief summary: problem, approach, key results.

## 1. Introduction
- Task background and motivation
- Proposed approach overview

## 2. Methodology  
- Key innovations and theoretical basis
- Implementation details

## 3. Experimental Setup
- Datasets used
- Hyperparameters and configurations
- Baseline method description

## 4. Results
- **Experiment Summary Table**: All runs with metrics
- **Baseline vs Proposed Comparison**: Per-dataset comparison with improvement percentages
- **Best Configuration**: Which setup worked best and why

## 5. Analysis
- Did the proposed method outperform baseline?
- On which datasets/settings did it work best?
- What are the limitations or failure cases?

## 6. Conclusion
- Main findings
- Recommendations for next steps

## Key Requirements
- Read ALL log files to extract actual numbers
- Include comparison table (baseline vs proposed)
- Be specific with metrics - no vague claims
- Save report using `write_file`
"""

    agent = Agent(
        name="Experiment Analysis Agent",
        instructions=instructions,
        tools=tools or [],
        # output_type=ExperimentAnalysisOutput, # Removed for duplex mode
        model=model,
    )

    return agent


def create_analysis_unifier_agent(model: str = "gpt-4o") -> Agent:
    return Agent(
        name="Analysis Output Unifier",
        instructions="""You are an expert data structuring assistant.
Your task is to convert the Research Report into a structured `ExperimentAnalysisOutput` object.

Input text is a full Research Report containing:
- Abstract
- Introduction
- Methodology
- Setup
- Results
- Analysis
- Conclusion & Future Work

Map these to the schema:
- `overall_analysis`: Combine Abstract, Introduction, and Methodology summaries.
- `metrics_analysis`: Extract metrics from the Results section.
- `innovations_analysis`: Extract from Methodology and Analysis sections.
- `plan_alignment`: Extract from Setup and Analysis.
- `idea_improvements` & `plan_improvements`: Extract from Conclusion & Future Work.
- `next_steps`: Extract from Future Work.

Ensure `metrics_analysis` is a list of `MetricAnalysis` objects.
Ensure boolean flags are correctly inferred from the text.
""",
        output_type=ExperimentAnalysisOutput,
        model=OUTPUT_UNIFIER_MODEL,
    )


class ExperimentAnalysisAgent:
    """
    Main experiment analysis agent that evaluates results and suggests improvements.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        working_dir: str = None,
        tools: Optional[list] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.working_dir = working_dir
        self.verbose = verbose
        # Always create hooks to show tool arguments
        # verbose mode controls whether to show detailed responses and results
        self.hooks = create_verbose_hooks(
            show_llm_responses=verbose,
            show_tools=verbose,
            show_tool_args=True,  # Always show tool arguments
        )

        # Auto-load recommended tools if not provided
        if tools is None:
            from src.agents.experiment_agent.sub_agents.experiment_analysis import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize analysis agent
        self.analysis_agent = create_analysis_agent(model=model, tools=self.tools)

        # Removed output_unifier for md output mode
        # self.output_unifier = create_analysis_unifier_agent(model=model)

        # Expose analysis agent as main agent for handoff compatibility
        self.agent = self.analysis_agent

    async def process(self, context: Any, **kwargs) -> ExperimentAnalysisOutput:
        """Process the analysis step."""
        # Extract full execute output
        execute_output = context.experiment_execute_output

        return await self.analyze(
            pre_analysis=context.pre_analysis_output,
            code_plan=context.code_plan_output,
            execute_output=execute_output,
        )

    async def analyze(
        self,
        pre_analysis,
        code_plan,
        execute_output,
    ) -> ExperimentAnalysisOutput:
        """
        Analyze experiment results and provide improvement suggestions.

        Args:
            pre_analysis: Pre-analysis output with research context
            code_plan: Code plan output with implementation details
            execute_output: Full execution output including all files and configs
        """
        print_section("EXPERIMENT ANALYSIS WORKFLOW", "=")

        # Helper function to get attribute from object or dict
        def get_attr(obj, attr, default="N/A"):
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        print_subsection("Preparing Analysis Context")

        # Extract execution information
        log_path = get_attr(execute_output, "log_path", "")
        execution_status = get_attr(execute_output, "execution_status", "unknown")
        execution_summary = get_attr(execute_output, "execution_summary", "")
        experiment_metrics = get_attr(execute_output, "experiment_metrics", "")

        # Extract output files information
        output_files = get_attr(execute_output, "output_files", [])
        files_info = ""
        if output_files:
            files_info = "\n=== EXPERIMENT OUTPUT FILES ===\n"
            for i, f in enumerate(output_files, 1):
                if isinstance(f, dict):
                    file_path = f.get("file_path", "")
                    file_type = f.get("file_type", "")
                    description = f.get("description", "")
                    run_command = f.get("run_command", "")
                    run_config = f.get("run_config", "")
                else:
                    file_path = getattr(f, "file_path", "")
                    file_type = getattr(f, "file_type", "")
                    description = getattr(f, "description", "")
                    run_command = getattr(f, "run_command", "")
                    run_config = getattr(f, "run_config", "")

                files_info += f"""
{i}. {file_path}
   - Type: {file_type}
   - Description: {description}
   - Command: {run_command}
   - Config: {run_config}
"""
            print_info(f"Found {len(output_files)} output files")
        else:
            print_info("No structured output files found")

        print_info(f"Primary log path: {log_path}")
        print_info(f"Execution status: {execution_status}")

        # Determine report output path - use working_dir if available
        if self.working_dir:
            report_path = os.path.join(self.working_dir, "analysis_report.md")
        elif log_path and os.path.dirname(log_path):
            report_path = os.path.join(os.path.dirname(log_path), "analysis_report.md")
        else:
            report_path = "analysis_report.md"

        # Extract experiment plan info if available
        experiment_plan_info = ""
        if code_plan:
            exp_plan = get_attr(code_plan, "experiment_plan", None)
            if exp_plan:
                baseline = get_attr(exp_plan, "baseline_method", "")
                datasets = get_attr(exp_plan, "datasets", [])
                metrics = get_attr(exp_plan, "primary_metrics", [])
                if baseline:
                    experiment_plan_info += f"Baseline Method: {baseline}\n"
                if datasets:
                    experiment_plan_info += (
                        f"Datasets: {', '.join(str(d) for d in datasets)}\n"
                    )
                if metrics:
                    experiment_plan_info += f"Primary Metrics: {', '.join(metrics)}\n"

        analysis_input = f"""## Task
Analyze experiment results and write a research report.

## Execution Results
- Status: {execution_status}
- Summary: {execution_summary}
- Metrics: {experiment_metrics}
{files_info}

## Research Context
- Goal: {get_attr(pre_analysis, "summary", "Validate the research implementation")}
- Innovations: {get_attr(pre_analysis, "key_innovations", "See implementation")}
{f"- {experiment_plan_info}" if experiment_plan_info else ""}

## Instructions
1. Read log files using `read_file` to extract actual metrics
2. Create comparison table (baseline vs proposed, if applicable)
3. Write comprehensive report following the structure in your instructions
4. Save report to: `{report_path}` using `write_file`

Start by reading the log files.
"""

        print_subsection("Analyzing Experiment Results")
        print_info(f"Report will be saved to: {report_path}")

        # Run analysis agent with streaming
        # Agent will read logs, write report, and save it using write_file tool
        result_stream = Runner.run_streamed(
            self.analysis_agent, analysis_input, hooks=self.hooks, max_turns=100
        )

        async for event in result_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type:
                    if hasattr(event.data, "delta"):
                        delta = event.data.delta
                        if hasattr(delta, "content") and delta.content:
                            print(delta.content, end="", flush=True)
                        elif hasattr(delta, "text") and delta.text:
                            print(delta.text, end="", flush=True)

        print_success("\nAnalysis completed")

        # Check if report was saved
        if os.path.exists(report_path):
            print_success(f"Report saved to: {report_path}")
        else:
            print_warning(f"Report file not found at: {report_path}")

        # Create minimal output - the actual report is in the file
        final_output = ExperimentAnalysisOutput(
            meets_requirements=True,  # Agent completed the task
            overall_analysis=f"Report saved to: {report_path}",
            metrics_analysis=[],
            pre_analysis_alignment="See report file.",
            key_innovations_validated=True,
            innovations_analysis="See report file.",
            plan_alignment="See report file.",
            plan_completeness=1.0,
            idea_needs_improvement=False,
            idea_improvements="See report file.",
            plan_needs_improvement=False,
            plan_improvements="See report file.",
            next_steps="See report file.",
        )

        print_section("EXPERIMENT ANALYSIS COMPLETE", "=")

        return final_output

    def analyze_sync(
        self,
        pre_analysis,
        code_plan,
        execute_output,
    ) -> ExperimentAnalysisOutput:
        import asyncio

        return asyncio.run(self.analyze(pre_analysis, code_plan, execute_output))


def create_experiment_analysis_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: Optional[list] = None,
    verbose: bool = False,
) -> ExperimentAnalysisAgent:
    """
    Factory function to create an experiment analysis agent.
    """
    return ExperimentAnalysisAgent(
        model=model, working_dir=working_dir, tools=tools, verbose=verbose
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_experiment_analysis_agent(model="gpt-4o")
        # Mock execution
        print("Agent created.")

    asyncio.run(main())
