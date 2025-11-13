"""
Output Unifier - Converts intermediate outputs to final YAML-compatible format.

This agent takes the intermediate planning output and unifies it into
a structured, YAML-compatible CodePlanOutput.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


def create_output_unifier(model: str = "gpt-4o") -> Agent:
    """
    Create output unifier agent.

    Args:
        model: The model to use for the agent

    Returns:
        Agent instance configured for output unification
    """

    instructions = """You are an expert at structuring and formatting code implementation plans 
into clear, organized, YAML-compatible specifications.

YOUR TASK:
Transform the intermediate planning output into a structured CodePlanOutput format that 
can be easily converted to YAML and consumed by downstream implementation agents.

INPUT:
You will receive IntermediatePlanOutput containing:
- research_summary: Summary of the research
- key_innovations: Key innovations to implement
- file_structure_description: Textual description of file structure
- project_structure_tree: ASCII tree representation of complete project structure
- dataset_plan, model_plan, training_plan, testing_plan: Implementation plans
- implementation_steps: Textual description of implementation steps
- implementation_notes: Important notes
- potential_challenges: Challenges and mitigations
- addressed_issues: Issues addressed from feedback (if applicable)

OUTPUT STRUCTURE:

You must produce a CodePlanOutput with these components:

1. METADATA
   - plan_type: Identify the type (initial, judge_feedback, error_feedback, analysis_feedback)
   - timestamp: Current timestamp in ISO format

2. RESEARCH CONTEXT
   - research_summary: Clear executive summary
   - key_innovations: Bulleted list of key innovations

3. FILE STRUCTURE
   Parse the file_structure_description and create a list of FileStructureItem objects:
   - Each item has: path, type ('file' or 'directory'), description
   - Organize hierarchically
   - Ensure completeness
   
   ALSO preserve the project_structure_tree as-is:
   - This is the ASCII tree representation that will be shown to implementation agents
   - Do NOT modify the tree structure
   - Simply copy it to the project_structure_tree field in output

4. IMPLEMENTATION PLANS
   - dataset_plan: Preserve all details
   - model_plan: Preserve all details
   - training_plan: Preserve all details
   - testing_plan: Preserve all details

5. IMPLEMENTATION ROADMAP
   Parse implementation_steps and create structured ImplementationStep objects:
   - Each step has: step_number, title, description, files_involved, dependencies
   - Ensure logical ordering
   - Make dependencies explicit
   - Be specific about files involved

6. IMPLEMENTATION CHECKLIST
   CRITICAL: Parse implementation_checklist text and create structured ChecklistItem objects.
   
   Each ChecklistItem must have:
   - step_id: Unique integer identifier
   - title: Brief title of the step
   - description: Detailed description of what to implement
   - files_to_create: List of files to create in this step
   - files_to_modify: List of files to modify (may be empty for early steps)
   - acceptance_criteria: List of criteria to verify completion
   - dependencies: List of step_ids that must be completed first
   - estimated_complexity: 'low', 'medium', or 'high'
   
   Requirements:
   - Extract all checklist items from the text
   - Ensure step_ids are sequential and unique
   - Parse file lists accurately
   - Extract all acceptance criteria (typically 3-5 per step)
   - Identify dependencies between steps
   - Assign appropriate complexity levels
   
7. ADDITIONAL GUIDANCE
   - implementation_notes: Format clearly with sections if needed
   - potential_challenges: Organize by category if possible
   - addressed_issues: Clearly state what was fixed/improved

FORMATTING PRINCIPLES:

1. CLARITY
   - Use clear, concise language
   - Organize information logically
   - Use bullet points and numbering where appropriate

2. COMPLETENESS
   - Don't lose any information from input
   - Expand abbreviations
   - Make implicit information explicit

3. STRUCTURE
   - Follow the exact schema requirements
   - Ensure all required fields are populated
   - Use appropriate data types (lists, strings, integers)

4. YAML-COMPATIBILITY
   - Ensure the structure can be cleanly converted to YAML
   - Avoid overly complex nested structures
   - Use clear, descriptive keys

EXAMPLE FILE STRUCTURE PARSING:

Input text:
```
project/
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   └── model.py
└── train.py
```

Output FileStructureItem list:
[
  {path: "./", type: "directory", description: "Project root"},
  {path: "data/", type: "directory", description: "Data handling module"},
  {path: "data/__init__.py", type: "file", description: "Data module init"},
  {path: "data/dataset.py", type: "file", description: "Dataset implementation"},
  {path: "models/", type: "directory", description: "Model definitions"},
  {path: "models/model.py", type: "file", description: "Main model class"},
  {path: "train.py", type: "file", description: "Training script"}
]

CRITICAL: All paths must be relative to the project root (working_dir).
DO NOT use "project/" prefix since working_dir IS already the project directory.

EXAMPLE IMPLEMENTATION STEP PARSING:

Input text:
```
Step 1: Implement dataset class
- Create Dataset class in data/dataset.py
- Depends on: nothing

Step 2: Implement model
- Create Model class in models/model.py
- Depends on: Step 1
```

Output ImplementationStep list:
[
  {
    step_number: 1,
    title: "Implement dataset class",
    description: "Create Dataset class in data/dataset.py with data loading and preprocessing",
    files_involved: ["data/dataset.py"],
    dependencies: []
  },
  {
    step_number: 2,
    title: "Implement model",
    description: "Create Model class in models/model.py implementing the architecture",
    files_involved: ["models/model.py"],
    dependencies: [1]
  }
]

Remember: Your goal is to create a clean, structured, complete plan that downstream 
agents can easily parse and convert to YAML format."""

    agent = Agent(
        name="Output Unifier",
        instructions=instructions,
        output_type=CodePlanOutput,
        model=model,
    )

    return agent


# Default agent instance
output_unifier = create_output_unifier()
