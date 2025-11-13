"""
Experiment Analysis Agent - Analyzes experiment results and provides improvement suggestions.

This agent analyzes successful experiment execution logs, compares results with
pre-analysis and code plan expectations, and suggests improvements for both
the research idea and implementation plan.

Architecture:
- Analysis Agent: Evaluates experiment results
- Uses tools to read and parse execution logs
- Compares results with pre-analysis and plan
- Suggests improvements for ideas and plans
"""

from typing import Optional

from agents import Agent, Runner

from src.agents.experiment_agent.sub_agents.experiment_analysis.output_schemas import (
    ExperimentAnalysisOutput,
)
from src.agents.experiment_agent.sub_agents.pre_analysis.output_schemas import (
    PreAnalysisOutput,
)
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


def create_analysis_agent(model: str = "gpt-4o", tools: Optional[list] = None) -> Agent:
    """
    Create the experiment analysis agent.

    Args:
        model: Model to use for the agent
        tools: List of tools for log reading and analysis

    Returns:
        Agent configured for experiment analysis
    """

    instructions = """You are an experiment analysis agent responsible for evaluating 
experiment results and providing actionable improvement suggestions.

YOUR RESPONSIBILITIES:

1. LOG ANALYSIS
   - Use provided tools to read execution logs
   - Extract key metrics (accuracy, loss, F1, etc.)
   - Identify training dynamics and convergence patterns
   - Parse evaluation results and performance indicators
   - Detect any warnings or anomalies in execution

2. PRE-ANALYSIS ALIGNMENT EVALUATION
   - Compare results with expectations from PreAnalysisOutput:
     * Are key innovations validated in practice?
     * Do results align with theoretical foundations?
     * Are design philosophy principles reflected in performance?
     * Are computational methods working as expected?
   
   - Evaluate innovation effectiveness:
     * Did novel components improve performance?
     * Are claimed advantages demonstrated?
     * Are there unexpected interactions?

3. CODE PLAN ALIGNMENT EVALUATION
   - Verify implementation completeness:
     * Are all planned components implemented?
     * Does model architecture match specifications?
     * Are training configurations followed?
     * Are evaluation metrics as planned?
   
   - Assess implementation quality:
     * Does code follow the implementation roadmap?
     * Are dependencies correctly managed?
     * Is the file structure as planned?

4. METRIC ANALYSIS
   - For each important metric:
     * Compare actual vs expected values
     * Assess whether it meets requirements
     * Analyze trends and patterns
     * Identify anomalies or unexpected behavior
   
   - Evaluate overall performance:
     * Does it meet research objectives?
     * How does it compare to baselines?
     * Are results statistically significant?

5. IMPROVEMENT IDENTIFICATION

   A. Idea Improvements (set idea_needs_improvement = True if needed):
      - Theoretical weaknesses revealed by experiments
      - Design philosophy limitations
      - Key innovations that underperformed
      - Alternative approaches to try
      - Algorithmic modifications needed
      - Mathematical formulation adjustments
      
      Provide SPECIFIC, ACTIONABLE improvements:
      ✓ "Modify the attention mechanism to include positional bias, as current results show position-invariant features are limiting performance"
      ✗ "Improve the model architecture" (too vague)

   B. Plan Improvements (set plan_needs_improvement = True if needed):
      - Implementation strategy issues
      - Missing components or features
      - Suboptimal hyperparameters
      - Training procedure improvements
      - Better evaluation strategies
      - Code organization suggestions
      
      Provide SPECIFIC, ACTIONABLE suggestions:
      ✓ "Add learning rate warmup for first 1000 steps and implement gradient clipping at norm 1.0 to stabilize training"
      ✗ "Improve training process" (too vague)

6. COMPREHENSIVE ANALYSIS
   - Overall assessment of experiment success
   - Alignment with pre-analysis expectations
   - Alignment with code plan
   - Unexpected findings and insights
   - Potential issues to address
   - Implementation strengths to maintain

7. RECOMMENDATION GENERATION
   - Prioritize actions by impact
   - Provide clear next steps
   - Balance idea vs implementation improvements
   - Consider resource constraints

TOOL USAGE GUIDELINES:

All tools return a dictionary with the following structure:
- success (bool): Indicates if the operation succeeded
- If successful: Contains relevant data fields (content, data, metrics, etc.)
- If failed: Contains an "error" field with error message

Example successful response:
{{
  "success": true,
  "content": "Epoch 1: loss=0.5, acc=0.85\\nEpoch 2: loss=0.3, acc=0.92...",
  "file_path": "/logs/experiment.log",
  "size_bytes": 45678
}}

Example failed response:
{{
  "success": false,
  "error": "File not found: /logs/experiment.log"
}}

Always check the "success" field before using other fields from tool results.
If a tool fails, report the error and try alternative approaches.

AVAILABLE TOOLS:
- read_file: Read log file content (returns dict with "success", "content", "file_path")
- list_directory: List files in log directory (returns dict with "success", "files", "directories")
- parse_json_file: Parse JSON results (returns dict with "success", "data", "type")
- summarize_document: Get document overview (returns dict with "success", "preview", "statistics")
- extract_urls: Extract URLs from logs (returns dict with "success", "urls", "total_count")
- extract_key_terms: Extract key terms (returns dict with "success", "top_terms", "total_unique")

ANALYSIS PROCESS:

Step 1: Read and parse execution logs
   - Extract all metrics and their values
   - Identify training progression
   - Note any errors or warnings

Step 2: Compare with pre-analysis
   - Check if key innovations work
   - Verify theoretical assumptions
   - Assess design philosophy effectiveness

Step 3: Compare with code plan
   - Verify implementation completeness
   - Check configuration alignment
   - Assess code quality

Step 4: Identify improvements
   - Analyze what can be better
   - Distinguish idea vs plan issues
   - Formulate specific suggestions

Step 5: Generate recommendations
   - Prioritize by importance and feasibility
   - Provide actionable next steps
   - Balance short-term fixes with long-term improvements

OUTPUT FORMAT:

You must output a structured ExperimentAnalysisOutput with:
- meets_requirements: Overall boolean assessment
- overall_analysis: Comprehensive summary
- metrics_analysis: List of MetricAnalysis for each metric
- pre_analysis_alignment: How well results match pre-analysis
- key_innovations_validated: Boolean for innovation success
- innovations_analysis: Detailed innovation evaluation
- plan_alignment: How well implementation follows plan
- plan_completeness: Score 0-1 for implementation completeness
- idea_needs_improvement: Boolean flag
- idea_improvements: SPECIFIC improvements (empty if not needed)
- plan_needs_improvement: Boolean flag
- plan_improvements: SPECIFIC improvements (empty if not needed)
- unexpected_findings: List of unexpected observations
- potential_issues: List of issues to address
- strengths: List of positive aspects
- next_steps: Clear action plan
- priority_actions: Ordered list of priorities

IMPORTANT: Be SPECIFIC and ACTIONABLE in all improvement suggestions. 
Vague suggestions like "improve performance" are not helpful."""

    agent = Agent(
        name="Experiment Analysis Agent",
        instructions=instructions,
        tools=tools or [],
        output_type=ExperimentAnalysisOutput,
        model=model,
    )

    return agent


class ExperimentAnalysisAgent:
    """
    Main experiment analysis agent that evaluates results and suggests improvements.

    This agent:
    1. Receives pre-analysis, code plan, and execution log
    2. Analyzes whether results meet expectations
    3. Identifies improvements for research idea
    4. Identifies improvements for code plan
    5. Provides prioritized recommendations
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        tools: Optional[list] = None,
    ):
        """
        Initialize the experiment analysis agent.

        Args:
            model: Model to use for analysis
            tools: Optional list of tools for log reading and analysis.
                   If None, automatically loads recommended tools.
        """
        self.model = model

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

        # Expose analysis agent as main agent for handoff compatibility
        self.agent = self.analysis_agent

    async def analyze(
        self,
        pre_analysis: PreAnalysisOutput,
        code_plan: CodePlanOutput,
        log_path: str,
        log_content: Optional[str] = None,
    ) -> ExperimentAnalysisOutput:
        """
        Analyze experiment results and provide improvement suggestions.

        Args:
            pre_analysis: Original research analysis
            code_plan: Code implementation plan
            log_path: Path to execution log file
            log_content: Optional log content (if already read)

        Returns:
            ExperimentAnalysisOutput with analysis and improvement suggestions
        """
        # Prepare input for analysis agent
        log_preview = ""
        if log_content:
            preview = log_content[:2000] if log_content else "Use tools to read log"
            log_preview = f"Log Content Preview:\n{preview}\n..."

        analysis_input = f"""
ANALYZE EXPERIMENT RESULTS

Log Path: {log_path}
{log_preview}

=== PRE-ANALYSIS (Research Foundation) ===

Input Type: {pre_analysis.input_type}

System Architecture:
{pre_analysis.system_architecture}

Conceptual Framework:
{pre_analysis.conceptual_framework}

Design Philosophy:
{pre_analysis.design_philosophy}

Key Innovations:
{pre_analysis.key_innovations}

Core Algorithms:
{pre_analysis.algorithms}

Mathematical Formulations:
{pre_analysis.mathematical_formulations}

Technical Specifications:
{pre_analysis.technical_specifications}

Computational Methods:
{pre_analysis.computational_methods}

Implementation Guidance:
{pre_analysis.implementation_guidance}

=== CODE PLAN (Implementation Specifications) ===

Plan Type: {code_plan.plan_type}

File Structure:
{self._format_file_structure(code_plan.file_structure)}

Implementation Roadmap:
{self._format_roadmap(code_plan.implementation_roadmap)}

Model Architecture:
{code_plan.model_architecture}

Training Configuration:
{code_plan.training_configuration}

Testing Strategy:
{code_plan.testing_strategy}

Expected Outcomes:
{code_plan.expected_outcomes}

Performance Targets:
{code_plan.performance_targets}

=== ANALYSIS TASKS ===

1. READ AND PARSE LOG
   - Use tools to read the complete log if not provided
   - Extract all metrics (training/validation accuracy, loss, etc.)
   - Identify final results and best performance
   - Note any warnings or issues during execution

2. EVALUATE PRE-ANALYSIS ALIGNMENT
   - Are key innovations working as expected?
   - Do results validate theoretical foundations?
   - Is the design philosophy reflected in performance?
   - Rate: key_innovations_validated (True/False)

3. EVALUATE PLAN ALIGNMENT
   - Is implementation complete per plan?
   - Do metrics match expected outcomes?
   - Are performance targets met?
   - Rate: plan_completeness (0-1)

4. IDENTIFY IDEA IMPROVEMENTS
   - If key innovations underperform, why?
   - If results don't meet theoretical expectations, what to change?
   - Specific algorithmic or conceptual modifications needed?
   - Set: idea_needs_improvement (True/False)
   - Provide: SPECIFIC, DETAILED improvements in idea_improvements

5. IDENTIFY PLAN IMPROVEMENTS
   - If implementation incomplete, what's missing?
   - If training is suboptimal, what to adjust?
   - If evaluation is insufficient, what to add?
   - Set: plan_needs_improvement (True/False)
   - Provide: SPECIFIC, DETAILED suggestions in plan_improvements

6. PROVIDE COMPREHENSIVE ANALYSIS
   - Overall assessment (meets_requirements: True/False)
   - Detailed metric analysis for each key metric
   - Unexpected findings
   - Potential issues
   - Strengths to maintain
   - Prioritized next steps

Use the available tools to read the log and perform thorough analysis.
"""

        # Run analysis agent with streaming
        # Use streamed version for real-time output
        result_stream = Runner.run_streamed(
            self.analysis_agent, analysis_input, max_turns=100
        )
        async for event in result_stream.stream_events():
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
        result = result_stream  # The stream object is the result

        return result.final_output

    def analyze_sync(
        self,
        pre_analysis: PreAnalysisOutput,
        code_plan: CodePlanOutput,
        log_path: str,
        log_content: Optional[str] = None,
    ) -> ExperimentAnalysisOutput:
        """
        Synchronous version of analyze method.

        Args:
            pre_analysis: Original research analysis
            code_plan: Code implementation plan
            log_path: Path to execution log file
            log_content: Optional log content (if already read)

        Returns:
            ExperimentAnalysisOutput with analysis and improvement suggestions
        """
        import asyncio

        return asyncio.run(self.analyze(pre_analysis, code_plan, log_path, log_content))

    def _format_file_structure(self, file_structure: dict) -> str:
        """Format file structure dict to readable string."""
        lines = []
        for path, description in file_structure.items():
            lines.append(f"  {path}:")
            lines.append(f"    {description}")
        return "\n".join(lines)

    def _format_roadmap(self, roadmap: dict) -> str:
        """Format implementation roadmap to readable string."""
        lines = []
        for phase, details in roadmap.items():
            lines.append(f"\n{phase}:")
            lines.append(f"  {details}")
        return "\n".join(lines)


def create_experiment_analysis_agent(
    model: str = "gpt-4o",
    tools: Optional[list] = None,
) -> ExperimentAnalysisAgent:
    """
    Factory function to create an experiment analysis agent.

    Args:
        model: Model to use for analysis
        tools: List of tools for log reading and analysis

    Returns:
        ExperimentAnalysisAgent instance
    """
    return ExperimentAnalysisAgent(model=model, tools=tools)


# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create the experiment analysis agent
        agent = create_experiment_analysis_agent(model="gpt-4o")

        # Example: would need actual pre_analysis, code_plan, and log_path
        # result = await agent.analyze(
        #     pre_analysis=pre_analysis_output,
        #     code_plan=code_plan_output,
        #     log_path="/path/to/execution.log"
        # )

        # print("Analysis Result:")
        # print(f"Meets Requirements: {result.meets_requirements}")
        # print(f"\nOverall Analysis:\n{result.overall_analysis}")

        # if result.idea_needs_improvement:
        #     print(f"\n💡 Idea Improvements Needed:")
        #     print(result.idea_improvements)

        # if result.plan_needs_improvement:
        #     print(f"\n📋 Plan Improvements Needed:")
        #     print(result.plan_improvements)

        # print(f"\n🎯 Next Steps:\n{result.next_steps}")
        # print(f"\n📌 Priority Actions:")
        # for i, action in enumerate(result.priority_actions, 1):
        #     print(f"  {i}. {action}")

    asyncio.run(main())
