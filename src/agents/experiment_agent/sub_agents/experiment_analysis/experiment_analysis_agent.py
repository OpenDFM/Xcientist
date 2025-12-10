"""
Experiment Analysis Agent - Analyzes experiment results and provides improvement suggestions.

This agent analyzes successful experiment execution logs, compares results with
pre-analysis and code plan expectations, and suggests improvements for both
the research idea and implementation plan.
"""

import os
from typing import Optional, Any

from agents import Agent, Runner, RunConfig, ModelSettings

from src.agents.experiment_agent.sub_agents.experiment_analysis.output_schemas import (
    ExperimentAnalysisOutput,
)

from src.agents.experiment_agent.logger import create_verbose_hooks

from src.agents.experiment_agent.utils import *

from src.agents.experiment_agent.utils.print_utils import *
from src.agents.experiment_agent.utils.json_utils import (
    extract_and_parse_json,
    JSONParseError,
)


# Unifier instruction for ExperimentAnalysisOutput
EXPERIMENT_ANALYSIS_UNIFIER_INSTRUCTION = """You are an Output Formatter. Convert the structured analysis output into JSON.

## Input Format
The input follows this structure:
```
=== EXPERIMENT ANALYSIS OUTPUT ===
MEETS_REQUIREMENTS: true/false
=== OVERALL ANALYSIS ===
...
=== METRICS ANALYSIS ===
METRIC #N:
METRIC_NAME: ...
ACTUAL_VALUE: ...
ANALYSIS: ...
=== PLAN IMPROVEMENTS ===
...
=== POTENTIAL ISSUES ===
- issue1
- issue2
=== NEXT STEPS ===
...
```

## Required JSON Output Format

```json
{
  "meets_requirements": true,
  "overall_analysis": "Summary text",
  "metrics_analysis": [
    {
      "metric_name": "accuracy",
      "actual_value": 0.92,
      "analysis": "Analysis text"
    }
  ],
  "plan_improvements": "Improvements text",
  "potential_issues": ["Issue 1", "Issue 2"],
  "next_steps": "Next steps text"
}
```

### Rules:
1. Parse MEETS_REQUIREMENTS -> `meets_requirements` (boolean)
2. Parse OVERALL ANALYSIS section -> `overall_analysis`
3. Parse each METRIC block -> `metrics_analysis` array
4. Parse PLAN IMPROVEMENTS -> `plan_improvements`
5. Parse POTENTIAL ISSUES -> `potential_issues` array
6. Parse NEXT STEPS -> `next_steps`

Output ONLY valid JSON wrapped in ```json ... ``` block.
"""


def create_analysis_output_unifier(model: str = None) -> Agent:
    """Create unifier agent to format analysis output."""
    if model is None:
        from src.agents.experiment_agent.config import UNIFIER_MODEL
        model = UNIFIER_MODEL
    return Agent(
        name="Experiment Analysis Output Unifier",
        instructions=EXPERIMENT_ANALYSIS_UNIFIER_INSTRUCTION,
        model=model,
    )


def create_analysis_agent(model: str = "gpt-4o", tools: Optional[list] = None) -> Agent:
    """
    Create the experiment analysis agent.
    """

    instructions = f"""You are a Principal Researcher analyzing experiment results to provide feedback for the next iteration.

## PURPOSE
Analysis triggers iteration back to code_plan_agent. Provide ACTIONABLE feedback.

## FILE LOCATIONS
| Path | Content |
|------|---------|
| `project/logs/` | Experiment logs (stdout/stderr) |
| `project/results/` | Experiment results (metrics, checkpoints) |

---

## WORKFLOW: READ → ANALYZE → OUTPUT JSON

### 1️⃣ READ
Use `file_viewer` to read log files from `project/logs/` and results from `project/results/`.
🚫 **DO NOT make up numbers** - only report what you find in logs.

### 2️⃣ ANALYZE
- Compare baseline vs proposed
- Identify patterns and evaluate innovations
- Be SPECIFIC with metrics - no vague claims

### 3️⃣ REPORT
🚫 **DO NOT use `write_file`** - just output your analysis.

**⛔ ABSOLUTELY PROHIBITED - DO NOT CREATE THESE FILES:**
- `STEP*.json`, `*_COMPLETION*.json`, `*_EVALUATION*.json`, `*_SUMMARY*.json`
- `*_RESULT*.json`, `*_ANALYSIS*.json`, `*_STATUS*.json`
- **ANY `.md` files**, **ANY summary/status/progress files**
The orchestrator handles step tracking externally.

---

## 4️⃣ OUTPUT (MANDATORY - AS CHAT RESPONSE, NOT FILE!)

**🚨 CRITICAL OUTPUT RULES:**
- **DO NOT use `write_file` to write the output below!**
- **DO NOT write any JSON/summary/progress files to disk!**
- The format below is your **FINAL CHAT RESPONSE** - just type it directly as text!
- A separate unifier agent will convert your text response to JSON.

After completing your analysis, **STOP calling tools** and output this TEXT directly in chat:

```
=== EXPERIMENT ANALYSIS OUTPUT ===

MEETS_REQUIREMENTS: true  # true if experiment meets pre-analysis requirements

=== OVERALL ANALYSIS ===
The experiment successfully demonstrated the proposed method outperforms baseline.
Key strengths: efficient training, stable convergence.
Unexpected finding: better performance on small datasets than large ones.

=== METRICS ANALYSIS ===

METRIC #1:
METRIC_NAME: accuracy
ACTUAL_VALUE: 0.92
ANALYSIS: Achieved 92% accuracy, exceeding the baseline of 87%. Meets the target of >90%.

METRIC #2:
METRIC_NAME: loss
ACTUAL_VALUE: 0.23
ANALYSIS: Final loss of 0.23, indicating good convergence.

=== PLAN IMPROVEMENTS ===
1. Add learning rate scheduling for better convergence
2. Increase batch size for faster training
3. Add data augmentation to improve generalization

=== POTENTIAL ISSUES ===
- Training time is longer than expected
- Memory usage could be optimized
- Model size may be too large for deployment

=== NEXT STEPS ===
Priority 1: Implement cosine learning rate scheduler
Priority 2: Profile memory usage and optimize
Priority 3: Test on larger dataset
Priority 4: Add model compression techniques
```

**🚨 REMINDER**: 
- This output format is your **CHAT RESPONSE** - just type it out!
- **DO NOT call write_file() with this content!**
- Be SPECIFIC with metrics - no vague claims!
"""

    agent = Agent(
        name="Experiment Analysis Agent",
        instructions=instructions,
        tools=tools or [],
        model=model,
    )

    return agent


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

        # Initialize output unifier agent
        self.output_unifier = create_analysis_output_unifier(model=model)

        # Expose analysis agent as main agent for handoff compatibility
        self.agent = self.analysis_agent

    def extract_summary(self, analysis_output: Any) -> tuple:
        """Extract summary, findings, and recommendations from analysis output."""
        if not analysis_output:
            return "", [], ""


        overall_summary = analysis_output.get("overall_analysis", "")
        
        key_findings = []
        for attr in ["unexpected_findings", "potential_issues"]:
            items = analysis_output.get(attr, [])
            if items:
                key_findings.extend(items)

        recommendations = []
        next_steps = analysis_output.get("next_steps", "")
        if next_steps:
            recommendations.append(next_steps)
        priority_actions = analysis_output.get("priority_actions", [])
        if priority_actions:
            recommendations.append("Priority Actions:\n" + "\n".join(f"- {a}" for a in priority_actions))

        return overall_summary, key_findings, "\n\n".join(recommendations)

    async def process(self, context: Any, **kwargs) -> ExperimentAnalysisOutput:
        """Process the analysis step."""
        # Extract full execute output
        execute_output = context.get("experiment_execute_output", None)

        return await self.analyze(
            pre_analysis=context.get("pre_analysis_output", None),
            code_plan=context.get("code_plan_output", None),
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
            if hasattr(obj, "get"):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        print_subsection("Preparing Analysis Context")

        # Extract execution information
        log_path = execute_output.get("log_path", "")
        execution_status = execute_output.get("execution_status", "unknown")
        execution_summary = execute_output.get("execution_summary", "")
        experiment_metrics = execute_output.get("experiment_metrics", "")

        # Extract output files information
        output_files = execute_output.get("output_files", [])
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

        # Extract experiment plan info if available
        experiment_plan_info = ""
        if code_plan:
            exp_plan = code_plan.get("experiment_plan", None)
            if exp_plan:
                baseline = exp_plan.get("baseline_method", "")
                datasets = exp_plan.get("datasets", [])
                metrics = exp_plan.get("primary_metrics", [])
                if baseline:
                    experiment_plan_info += f"Baseline Method: {baseline}\n"
                if datasets:
                    experiment_plan_info += (
                        f"Datasets: {', '.join(str(d) for d in datasets)}\n"
                    )
                if metrics:
                    experiment_plan_info += f"Primary Metrics: {', '.join(metrics)}\n"

        analysis_input = f"""## Task
Analyze experiment results and provide structured feedback for improvement.

## Execution Results
- Status: {execution_status}
- Summary: {execution_summary}
- Metrics: {experiment_metrics}
{files_info}

## Research Context
- Goal: {pre_analysis.get("summary", "Validate the research implementation")}
- Innovations: {pre_analysis.get("key_innovations", "See implementation")}
{f"- {experiment_plan_info}" if experiment_plan_info else ""}

## Instructions
1. Read log files using `file_viewer` to extract actual metrics
2. Analyze results comparing baseline vs proposed (if applicable)
3. Provide structured JSON feedback following the format in your instructions

Focus on:
- What worked well
- What needs improvement for next iteration
- Specific actionable suggestions

Start by reading the log files.
"""

        print_subsection("Analyzing Experiment Results")

        run_config = RunConfig(
            model_settings=ModelSettings(max_tokens=128*1024)
        )

        # Run analysis agent with streaming
        result_stream = Runner.run_streamed(
            self.analysis_agent, analysis_input, hooks=self.hooks, max_turns=100, run_config=run_config
        )

        final_text = ""
        async for event in result_stream.stream_events():
            if hasattr(event, "data"):
                if hasattr(event.data, "delta"):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                        final_text += delta.content
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
                        final_text += delta.text

        # Get final output from stream if not captured
        if not final_text and hasattr(result_stream, "final_output"):
            if isinstance(result_stream.final_output, str):
                final_text = result_stream.final_output

        # Fallback: search chat history
        if not final_text and hasattr(result_stream, "chat_history"):
            for msg in reversed(result_stream.chat_history):
                if hasattr(msg, "role") and msg.role == "assistant":
                    if hasattr(msg, "content") and msg.content and isinstance(msg.content, str) and len(msg.content) > 10:
                        final_text = msg.content
                        break

        print_success("\nAnalysis text generated")

        print_subsection("Unifying Output")

        # Use unifier agent to convert raw output to structured JSON
        unifier_prompt = f"""Convert the following analysis output to JSON:

=== RAW OUTPUT START ===
{final_text}
=== RAW OUTPUT END ===

Extract all analysis results, metrics, and recommendations. Output the structured JSON:"""

        unifier_result = await Runner.run(
            self.output_unifier,
            unifier_prompt,
            run_config=RunConfig(model_settings=ModelSettings(max_tokens=64*1024)),
        )

        unified_text = ""
        if hasattr(unifier_result, "final_output") and isinstance(unifier_result.final_output, str):
            unified_text = unifier_result.final_output
        elif hasattr(unifier_result, "chat_history") and unifier_result.chat_history:
            unified_text = unifier_result.chat_history[-1].content

        print_subsection("Parsing JSON Output")

        # Extract and parse JSON from the unified output
        try:
            final_output = extract_and_parse_json(unified_text, ExperimentAnalysisOutput, raise_on_failure=True)
        except JSONParseError as e:
            print_error(f"JSON parsing failed, will trigger retry: {e}")
            raise

        print_success("Output parsed successfully")
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
