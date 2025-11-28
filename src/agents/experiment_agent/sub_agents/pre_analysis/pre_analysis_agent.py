"""
Pre-Analysis Agent - Main orchestrator for research analysis.

This agent directly routes inputs (papers or ideas) to appropriate analyzers
and produces unified analysis output.
"""

import asyncio
import os
import time
from typing import Dict, Optional, Any

from agents import Agent, Runner

from src.agents.experiment_agent.utils.common_utils import format_list, format_dict
from src.agents.experiment_agent.logger import create_verbose_hooks

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
from src.agents.experiment_agent.sub_agents.pre_analysis.analysis_synthesizer import (
    create_analysis_synthesizer,
)

from src.agents.experiment_agent.config import OUTPUT_UNIFIER_MODEL

from src.agents.experiment_agent.utils.print_utils import *


class PreAnalysisAgent:
    """
    Main pre-analysis agent that orchestrates the entire analysis workflow.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        tools: Optional[Dict[str, list]] = None,
        workspace_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.verbose = verbose

        # Always create hooks to show tool arguments
        # verbose mode controls whether to show detailed responses and results
        self.hooks = create_verbose_hooks(
            show_llm_responses=verbose,
            show_tools=verbose,
            show_tool_args=True,  # Always show tool arguments
        )

        if workspace_dir is None:
            from src.agents.experiment_agent.config import LOCAL_WORKSPACE_DIR

            workspace_dir = LOCAL_WORKSPACE_DIR

        self.workspace_dir = workspace_dir

        if tools is None:
            from src.agents.experiment_agent.tools import (
                DOCUMENT_TOOLS,
                FILE_TOOLS,
                REPOSITORY_TOOLS,
            )

            self.tools = {
                "paper": FILE_TOOLS[:3] + REPOSITORY_TOOLS[:2],
                "idea": FILE_TOOLS[:3] + REPOSITORY_TOOLS[:2],
            }
        else:
            self.tools = tools

        # Initialize analyzers
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

        self.synthesizer = create_analysis_synthesizer(model=model)

        # Initialize Output Unifier
        self.output_unifier = Agent(
            name="Pre-Analysis Output Unifier",
            instructions="""You are an expert data structuring assistant.
Your task is to extract structured information from the provided research analysis text and format it into the `PreAnalysisOutput` schema.

Input text will contain sections for:
- System Architecture
- Conceptual Framework
- Key Innovations
- Algorithms
- Mathematical Formulations
- Technical Specifications
- Executive Summary
- Implementation Guidance

Map these sections to the corresponding fields in the output schema.
Preserve the detail and formatting (including LaTeX) of the original text.
""",
            output_type=PreAnalysisOutput,
            model=OUTPUT_UNIFIER_MODEL,
        )

        # Simplified Orchestrator Prompt
        self.agent = Agent(
            name="Pre-Analysis Agent",
            instructions="""You are the Research Analyst. 

### MISSION
Synthesize raw research inputs (papers or ideas) into a technical specification suitable for implementation planning.

### OUTPUT DELIVERABLES
1. **System Architecture**: High-level components and data flow.
2. **Innovations**: What is new? (Theoretical Delta).
3. **Algorithms**: Mathematical formulations and pseudo-code.
4. **Tech Specs**: Computational requirements.

### PROTOCOL
- **Analyze**: Extract core logic, not just abstract summaries.
- **Synthesize**: Merge findings into a coherent `PreAnalysisOutput`.
- **Handoff**: Once complete, transfer control to the Experiment Master.
""",
            model=model,
        )

    async def process(self, context: Any, **kwargs) -> PreAnalysisOutput:
        """
        Process the current step using context data.
        """
        formatted_input = ""

        if context.input_type == "idea":
            import json

            try:
                idea_data = json.loads(context.research_input)
                idea_obj = idea_data.get("idea", {})
                formatted_input = f"""INPUT_TYPE: IDEA
Analyze ID: {idea_obj.get("title", "N/A")}         
Description: {idea_obj.get("description", "N/A")}
Innovations: {format_list(idea_obj.get("key_innovations", []))}
Methodology: {format_dict(idea_obj.get("methodology", {}))}
"""
            except json.JSONDecodeError:
                formatted_input = (
                    f"INPUT_TYPE: IDEA\nAnalyze Idea:\n{context.research_input}"
                )
        else:
            formatted_input = (
                f"INPUT_TYPE: PAPER\nAnalyze Paper:\n{context.research_input}"
            )

        return await self.analyze(formatted_input)

    def _detect_input_type(self, input_data: str) -> str:
        if "INPUT_TYPE: PAPER" in input_data:
            return "paper"
        elif "INPUT_TYPE: IDEA" in input_data:
            return "idea"
        if "Analyze the following research idea:" in input_data:
            return "idea"
        if "=== IDEA INFORMATION ===" in input_data:
            return "idea"
        if "\\documentclass" in input_data or "\\begin{document}" in input_data:
            return "paper"
        elif '"messages"' in input_data or '"idea_evaluation"' in input_data:
            return "idea"
        return "paper"

    async def analyze(
        self, input_data: str, input_path: Optional[str] = None
    ) -> PreAnalysisOutput:
        """
        Analyze research input (paper or idea) and produce unified output.
        """
        input_type = self._detect_input_type(input_data)

        print_section("PRE-ANALYSIS WORKFLOW", "=")
        print_info(
            f"Input type detected: {Colors.BOLD}{input_type.upper()}{Colors.ENDC}"
        )

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
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type:
                    if hasattr(event.data, "delta"):
                        delta = event.data.delta
                        if hasattr(delta, "content") and delta.content:
                            print(delta.content, end="", flush=True)
                        elif hasattr(delta, "text") and delta.text:
                            print(delta.text, end="", flush=True)

        concept_result = concept_stream
        concept_analysis_text = ""
        if hasattr(concept_result, "final_output") and isinstance(
            concept_result.final_output, str
        ):
            concept_analysis_text = concept_result.final_output
        elif hasattr(concept_result, "chat_history") and concept_result.chat_history:
            concept_analysis_text = concept_result.chat_history[-1].content

        print_success(f"Concept analysis completed")

        # Rate limiting
        print_info(f"Waiting 5 seconds...", indent=0)
        await asyncio.sleep(5)

        # Step 3: Algorithm analysis
        if input_type == "paper":
            algo_prompt = f"Extract algorithms from this paper:\n{input_data}"
        else:  # idea
            algo_prompt = (
                f"Generate algorithmic specifications for this idea:\n{input_data}"
            )

        print_subsection("Main Algorithm Analysis")

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
                if "FunctionCallArguments" not in event_type:
                    if hasattr(event.data, "delta"):
                        delta = event.data.delta
                        if hasattr(delta, "content") and delta.content:
                            print(delta.content, end="", flush=True)
                        elif hasattr(delta, "text") and delta.text:
                            print(delta.text, end="", flush=True)

        algorithm_result = algorithm_stream
        algorithm_analysis_text = ""
        if hasattr(algorithm_result, "final_output") and isinstance(
            algorithm_result.final_output, str
        ):
            algorithm_analysis_text = algorithm_result.final_output
        elif (
            hasattr(algorithm_result, "chat_history") and algorithm_result.chat_history
        ):
            algorithm_analysis_text = algorithm_result.chat_history[-1].content

        print_success(f"Algorithm analysis completed")

        # Step 4: Synthesis
        print_subsection("Merging Analysis Results")

        synthesis_prompt = f"""
Please synthesize the following analysis results:

INPUT TYPE: {input_type}

=== CONCEPT ANALYSIS ===
{concept_analysis_text}

=== ALGORITHM ANALYSIS ===
{algorithm_analysis_text}
"""

        synthesis_stream = Runner.run_streamed(
            self.synthesizer,
            synthesis_prompt,
            hooks=self.hooks,
            max_turns=100,
        )

        async for event in synthesis_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type:
                    if hasattr(event.data, "delta"):
                        delta = event.data.delta
                        if hasattr(delta, "content") and delta.content:
                            print(delta.content, end="", flush=True)
                        elif hasattr(delta, "text") and delta.text:
                            print(delta.text, end="", flush=True)

        synthesis_result = synthesis_stream
        synthesis_text = ""
        if hasattr(synthesis_result, "final_output") and isinstance(
            synthesis_result.final_output, str
        ):
            synthesis_text = synthesis_result.final_output
        elif (
            hasattr(synthesis_result, "chat_history") and synthesis_result.chat_history
        ):
            synthesis_text = synthesis_result.chat_history[-1].content

        # Step 5: Unification
        print_subsection("Unifying Output Format")

        unifier_input = f"""
Input Type: {input_type}

=== CONCEPT ANALYSIS ===
{concept_analysis_text}

=== ALGORITHM ANALYSIS ===
{algorithm_analysis_text}

=== SYNTHESIS ===
{synthesis_text}
"""
        unifier_stream = Runner.run_streamed(
            self.output_unifier, unifier_input, hooks=None
        )

        # Drive the unifier stream
        async for _ in unifier_stream.stream_events():
            pass

        unified_output = unifier_stream.final_output

        print_success("Analysis results merged successfully")
        print_section("PRE-ANALYSIS COMPLETE", "=")

        return unified_output

    def analyze_sync(
        self, input_data: str, input_path: Optional[str] = None
    ) -> PreAnalysisOutput:
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
    """
    return PreAnalysisAgent(
        model=model, tools=tools, workspace_dir=workspace_dir, verbose=verbose
    )


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_pre_analysis_agent(model="gpt-4o")
        paper_content = "\\documentclass{article}..."
        result = await agent.analyze(paper_content)
        print(f"Summary: {result.summary}")

    asyncio.run(main())
