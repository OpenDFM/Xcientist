"""
Experiment Analysis Agent - Analyzes experiment results and provides improvement suggestions.

This agent analyzes successful experiment execution logs, compares results with
pre-analysis and code plan expectations, and suggests improvements for both
the research idea and implementation plan.
"""

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

    instructions = """You are a Principal Researcher. Your task is to write a comprehensive Research Report based on the experiment results.

This report will be used to summarize the current iteration and provide insights for future idea generation.

### INPUTS
1. **Execution Logs**: Contains metrics (Loss, Accuracy, etc.) and stdout.
2. **Pre-Analysis**: The theoretical claims, expected innovations, and motivation.
3. **Code Plan**: The implementation specifications and setup.

### REPORT STRUCTURE
You MUST format your output as a Research Report with the following sections:

# [Title]

## Abstract
Brief summary of the task, approach, and key results.

## 1. Introduction
   - **Task Background**: What is the problem being solved?
   - **Motivation**: Why is this problem important? What are the limitations of existing methods?
   - **Proposed Approach**: Briefly describe the core idea/innovation.

## 2. Methodology
   - **Theoretical Basis**: Explain the key innovations in detail.
   - **Implementation Details**: How was it implemented?

## 3. Experimental Setup
   - **Dataset**: What data was used?
   - **Hyperparameters**: Key settings (LR, Batch Size, Optimizer, etc.).
   - **Environment**: Hardware/Software context.

## 4. Results
   - **Key Metrics**: Present the results (Best Accuracy, Final Loss, etc.).
   - **Training Dynamics**: Describe the convergence behavior.
   - **Comparison**: Compare with baselines or targets.

## 5. Analysis (CRITICAL)
   - **Hypothesis Verification**: Did the innovations work as expected? Why or why not?
   - **Ablation/Deep Dive**: What components contributed most?
   - **Error Analysis**: Identify patterns in failure cases.

## 6. Conclusion & Future Work
   - **Summary**: Recap the main findings.
   - **Suggestions**: Concrete ideas for the next iteration.

### METADATA (MANDATORY)
At the very end of your response, you MUST include a code block with the following boolean flags for system parsing:
```json
{
    "meets_requirements": boolean,
    "key_innovations_validated": boolean,
    "idea_needs_improvement": boolean,
    "plan_needs_improvement": boolean,
    "plan_completeness": float_0_to_1
}
```

### GUIDELINES
- Use professional academic tone.
- Be specific with numbers.
- Be critical in the Analysis section.
- **Output the report in valid Markdown.**
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
        tools: Optional[list] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.verbose = verbose
        self.hooks = (
            create_verbose_hooks(
                show_llm_responses=verbose,
                show_tools=verbose,
            )
            if verbose
            else None
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
        log_path = ""
        if context.experiment_execute_output:
            if isinstance(context.experiment_execute_output, dict):
                log_path = context.experiment_execute_output.get("log_path", "")
            else:
                log_path = getattr(context.experiment_execute_output, "log_path", "")

        return await self.analyze(
            pre_analysis=context.pre_analysis_output,
            code_plan=context.code_plan_output,
            log_path=log_path,
        )

    async def analyze(
        self,
        pre_analysis,
        code_plan,
        log_path: str,
        log_content: Optional[str] = None,
    ) -> ExperimentAnalysisOutput:
        """
        Analyze experiment results and provide improvement suggestions.
        """
        print_section("EXPERIMENT ANALYSIS WORKFLOW", "=")

        # Helper function to get attribute from object or dict
        def get_attr(obj, attr, default="N/A"):
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        print_subsection("Preparing Analysis Context")
        print_info(f"Log path: {log_path}")

        # Prepare input for analysis agent
        log_preview = ""
        if log_content:
            preview = log_content[:2000] if log_content else "Use tools to read log"
            log_preview = f"Log Content Preview:\n{preview}\n..."
            print_info(f"Log content loaded ({len(log_content)} characters)")
        else:
            print_info("Log will be read using tools during analysis")

        analysis_input = f"""
ANALYZE EXPERIMENT & WRITE REPORT

Log Path: {log_path}
{log_preview}

=== CONTEXT ===
Input Type: {get_attr(pre_analysis, "input_type")}
Research Summary: {get_attr(pre_analysis, "summary")}
Innovations: {get_attr(pre_analysis, "key_innovations")}

Plan Targets: {get_attr(code_plan, "performance_targets")}
Plan Overview: {get_attr(code_plan, "plan_overview", "See full plan details if available")}

Task:
1. Read log at `{log_path}` using tools.
2. Extract all relevant metrics and training dynamics.
3. Synthesize information from Context (Pre-Analysis, Code Plan) and Logs.
4. Write a **Comprehensive Research Report** following the specified structure (Intro, Method, Setup, Results, Analysis, Conclusion).
"""

        print_subsection("Analyzing Experiment Results")

        # Run analysis agent with streaming
        result_stream = Runner.run_streamed(
            self.analysis_agent, analysis_input, hooks=self.hooks, max_turns=100
        )
        final_text = ""
        async for event in result_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type:
                    if hasattr(event.data, "delta"):
                        delta = event.data.delta
                        if hasattr(delta, "content") and delta.content:
                            print(delta.content, end="", flush=True)
                            final_text += delta.content
                        elif hasattr(delta, "text") and delta.text:
                            print(delta.text, end="", flush=True)
                            final_text += delta.text

        result = result_stream
        if hasattr(result, "final_output") and isinstance(result.final_output, str):
            final_text = result.final_output
        elif not final_text and hasattr(result, "chat_history") and result.chat_history:
            final_text = result.chat_history[-1].content

        print_success("Analysis text report generated")
        print_subsection("Saving Report & Extracting Metadata")

        # Save MD file
        import os
        import json
        import re

        report_path = "analysis_report.md"
        if log_path:
            log_dir = os.path.dirname(log_path)
            report_path = os.path.join(log_dir, "analysis_report.md")

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            print_success(f"Report saved to: {report_path}")
        except Exception as e:
            print_error(f"Failed to save report: {e}")

        # Extract Metadata JSON
        metadata = {}
        try:
            # Match the last JSON code block
            json_matches = list(
                re.finditer(r"```json\s*(\{.*?\})\s*```", final_text, re.DOTALL)
            )
            if json_matches:
                json_str = json_matches[-1].group(1)
                metadata = json.loads(json_str)
                print_info("Metadata extracted successfully")
            else:
                print_warning("No metadata JSON block found in report")
        except Exception as e:
            print_warning(f"Failed to parse metadata: {e}")

        # Construct Output Object manually
        final_output = ExperimentAnalysisOutput(
            meets_requirements=metadata.get("meets_requirements", False),
            overall_analysis=final_text,  # Put full report here
            metrics_analysis=[],  # Skip detailed parsing
            pre_analysis_alignment="See full report.",
            key_innovations_validated=metadata.get("key_innovations_validated", False),
            innovations_analysis="See full report.",
            plan_alignment="See full report.",
            plan_completeness=metadata.get("plan_completeness", 0.0),
            idea_needs_improvement=metadata.get("idea_needs_improvement", True),
            idea_improvements="See full report.",
            plan_needs_improvement=metadata.get("plan_needs_improvement", True),
            plan_improvements="See full report.",
            next_steps="See full report.",
        )

        print_success("Analysis completed (MD Output Mode)")

        # Display key analysis results
        if hasattr(final_output, "meets_requirements"):
            status = (
                "✓ MEETS REQUIREMENTS"
                if final_output.meets_requirements
                else "✗ NEEDS IMPROVEMENT"
            )
            color = (
                Colors.OKGREEN if final_output.meets_requirements else Colors.WARNING
            )
            print(f"\n{color}{Colors.BOLD}{status}{Colors.ENDC}\n")

        if (
            hasattr(final_output, "idea_needs_improvement")
            and final_output.idea_needs_improvement
        ):
            print_info("💡 Research idea improvements recommended")

        if (
            hasattr(final_output, "plan_needs_improvement")
            and final_output.plan_needs_improvement
        ):
            print_info("📋 Implementation plan improvements recommended")

        print_section("EXPERIMENT ANALYSIS COMPLETE", "=")

        return final_output

    def analyze_sync(
        self,
        pre_analysis,
        code_plan,
        log_path: str,
        log_content: Optional[str] = None,
    ) -> ExperimentAnalysisOutput:
        import asyncio

        return asyncio.run(self.analyze(pre_analysis, code_plan, log_path, log_content))


def create_experiment_analysis_agent(
    model: str = "gpt-4o",
    tools: Optional[list] = None,
) -> ExperimentAnalysisAgent:
    """
    Factory function to create an experiment analysis agent.
    """
    return ExperimentAnalysisAgent(model=model, tools=tools)


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_experiment_analysis_agent(model="gpt-4o")
        # Mock execution
        print("Agent created.")

    asyncio.run(main())
