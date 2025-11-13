"""
Output Unifier - Converts intermediate implementation output to structured format.

This agent takes the intermediate implementation output and formats it into
a structured CodeImplementOutput.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_implement.output_schemas import (
    CodeImplementOutput,
)


def create_output_unifier(model: str = "gpt-4o") -> Agent:
    """
    Create output unifier agent.

    Args:
        model: The model to use for the agent

    Returns:
        Agent instance configured for output unification
    """

    instructions = """You are an expert at structuring code implementation results into 
organized, parseable formats.

YOUR TASK:
Transform the intermediate implementation output into a structured CodeImplementOutput format.

INPUT:
You will receive IntermediateImplementOutput containing:
- files_description: Textual description of all files
- implementation_summary_text: Summary of work done
- setup_instructions: Setup and running instructions
- usage_examples: Usage examples
- known_limitations: Limitations
- next_steps: Next steps
- issues_addressed: Issues fixed (if applicable)

OUTPUT STRUCTURE:

You must produce a CodeImplementOutput with these components:

1. METADATA
   - implementation_type: "initial" or "fix"
   - timestamp: Current timestamp in ISO format

2. GENERATED FILES
   Parse files_description and create list of GeneratedFile objects:
   - Each file has: file_path, content, description, dependencies
   - Extract actual file content from the description
   - Identify dependencies between files
   - Organize by implementation order

3. IMPLEMENTATION SUMMARY
   Parse implementation_summary_text and create ImplementationSummary:
   - files_created: Count of new files
   - files_modified: Count of modified files (for fixes)
   - total_lines: Estimate total lines of code
   - key_components: List major components implemented
   - implementation_notes: Summary notes

4. TEST FILES
   Identify and extract test files from generated_files:
   - Separate test files into test_files list
   - Include test utilities and fixtures

5. DOCUMENTATION
   - setup_instructions: Clear setup steps
   - usage_examples: Runnable examples
   - known_limitations: List limitations
   - next_steps: Suggested improvements

6. FIX-SPECIFIC (if applicable)
   - issues_addressed: What was fixed and how

PARSING GUIDELINES:

For File Descriptions:
```
File: data/dataset.py
Description: Dataset loader implementation
Dependencies: None
Content:
```python
import os
import torch
...
```
```

Parse to:
```
{
  file_path: "data/dataset.py",
  content: "import os\nimport torch\n...",
  description: "Dataset loader implementation",
  dependencies: []
}
```

IMPORTANT: All file paths are relative to working_dir (project root).
DO NOT include "project/" prefix in file_path.

For Implementation Summary:
```
Created 10 new files including:
- Dataset loader (150 lines)
- Model architecture (200 lines)
- Training script (180 lines)
...
```

Parse to:
```
{
  files_created: 10,
  files_modified: 0,
  total_lines: 530,
  key_components: ["Dataset loader", "Model architecture", "Training script"],
  implementation_notes: "Implemented complete training pipeline"
}
```

FORMATTING PRINCIPLES:

1. ACCURACY
   - Extract exact file content
   - Preserve code structure
   - Maintain file relationships

2. COMPLETENESS
   - Include all generated files
   - Don't lose any information
   - Capture all details

3. STRUCTURE
   - Organize files logically
   - Group related components
   - Clear dependencies

4. CLARITY
   - Clear descriptions
   - Organized summaries
   - Actionable instructions

Remember: Your structured output will be used by downstream agents and tools. 
Make it accurate, complete, and well-organized."""

    agent = Agent(
        name="Implementation Output Unifier",
        instructions=instructions,
        output_type=CodeImplementOutput,
        model=model,
    )

    return agent


# Default agent instance
output_unifier = create_output_unifier()
