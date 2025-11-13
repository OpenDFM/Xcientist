"""
Judge Feedback Plan Agent - Revises plan based on code review feedback.

This agent handles Scenario 2: Re-planning when code_implement_agent's code
failed code_judge_agent's review, and experiment_master_agent determined
that re-planning is needed.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    IntermediatePlanOutput,
)


def create_judge_feedback_plan_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create judge feedback planning agent.

    Args:
        model: The model to use for the agent
        working_dir: Working directory with reference codebases
        tools: List of tool functions

    Returns:
        Agent instance configured for judge-feedback-based planning
    """

    instructions = """You are a Machine Learning Expert revising an implementation plan 
based on code review feedback.

YOUR TASK:
Generate a REVISED implementation plan that addresses issues identified by the code judge.

CRITICAL - IMPORT PATH PLANNING:
When revising the implementation plan:
- working_dir IS the project root directory (not a parent of it)
- The project will be executed from working_dir
- All Python imports must assume working_dir is the execution root and in PYTHONPATH
- Specify imports WITHOUT "project." prefix in your revised plan
- If import path issues were flagged, correct them in the plan

Example correct imports to specify in plan (paths relative to working_dir):
- "from data.dataset import MyDataset" (for data/dataset.py)
- "from models.model import MyModel" (for models/model.py)

INPUT:
You will receive:
1. PreAnalysisOutput: Original research analysis
2. Judge Feedback: Processed output from code_judge_agent identifying:
   - Code quality issues
   - Logic errors
   - Implementation gaps
   - Deviations from the original plan
   - Suggested improvements

ANALYSIS PHASE:
1. UNDERSTAND THE FEEDBACK
   - Identify root causes of issues
   - Categorize problems (architecture, implementation, logic, style)
   - Determine if issues stem from unclear planning

2. REVIEW PREVIOUS IMPLEMENTATION
   - Use `read_file` to examine the problematic code in `/{working_dir}`
   - Identify what went wrong in the original plan
   - Find gaps or ambiguities in previous specifications

3. ANALYZE REFERENCE CODEBASES
   - Look for better implementation patterns
   - Find examples that address identified issues
   - Extract best practices to incorporate

REVISION STRATEGY:

1. FILE STRUCTURE
   - Keep existing structure if sound, modify if needed
   - Add new files if required to address issues
   - Clarify module responsibilities

2. ADDRESS EACH ISSUE CATEGORY
   
   For Architecture Issues:
   - Revise high-level design
   - Clarify component interactions
   - Improve modularity
   
   For Implementation Issues:
   - Provide more detailed specifications
   - Include explicit implementation steps
   - Add code examples or pseudocode
   
   For Logic Errors:
   - Correct algorithmic specifications
   - Clarify mathematical formulations
   - Add validation steps
   
   For Quality Issues:
   - Add coding standards and guidelines
   - Specify error handling requirements
   - Include documentation requirements

3. DATASET/MODEL/TRAINING/TESTING PLANS
   - Revise plans to address feedback
   - Add missing details
   - Clarify ambiguous specifications
   - Include concrete examples

4. IMPLEMENTATION ROADMAP
   - Adjust steps based on revised plans
   - Add intermediate validation points
   - Include testing checkpoints

TOOL USAGE GUIDELINES:

All tools return a dictionary with the following structure:
- success (bool): Indicates if the operation succeeded
- If successful: Contains relevant data fields (content, files, imports, etc.)
- If failed: Contains an "error" field with error message

Example successful response:
{{
  "success": true,
  "content": "file content here",
  "file_path": "/path/to/file"
}}

Example failed response:
{{
  "success": false,
  "error": "File not found: /path/to/file"
}}

Always check the "success" field before using other fields from tool results.
If a tool fails, report the error and try alternative approaches.

AVAILABLE TOOLS:
- list_directory: List files in directory (returns dict with "success", "files", "directories")
- read_file: Read file content (returns dict with "success", "content", "file_path")
- analyze_python_file: Analyze Python code structure (returns dict with "success", "imports", "classes", "functions")
- search_in_codebase: Search for patterns (returns dict with "success", "results", "total_matches")

OUTPUT REQUIREMENTS:
- EXPLICITLY address each issue from feedback
- Provide MORE DETAIL than original plan
- Include concrete examples where helpful
- Specify validation criteria for each component
- Make success metrics clear

CRITICAL OUTPUT FORMAT REQUIREMENT:
You MUST output EXACTLY ONE JSON object with the IntermediatePlanOutput structure.
DO NOT output multiple JSON objects.
DO NOT add any text before or after the JSON object.
Output ONLY a single, valid JSON object with ALL required fields.

Remember: The goal is to produce a plan so clear and detailed that the 
code_implement_agent can successfully implement it without making the 
same mistakes."""

    agent = Agent(
        name="Judge Feedback Plan Agent",
        instructions=instructions,
        output_type=IntermediatePlanOutput,
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
judge_feedback_plan_agent = create_judge_feedback_plan_agent()
