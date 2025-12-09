"""
Code Implementation Agent - Unified implementation agent.

This agent implements code strictly following the code plan's instructions.
It handles both initial implementation and iterative fixes based on feedback,
all guided by the code plan from the code_plan agent.
"""

import os
from typing import Dict, Optional, Any

from agents import Agent, Runner, RunConfig, ModelSettings
from src.agents.experiment_agent.sub_agents.code_implement.output_schemas import (
    CodeImplementOutput,
)
from src.agents.experiment_agent.utils.common_utils import extract_core_plan_context
from src.agents.experiment_agent.logger import create_verbose_hooks
from src.agents.experiment_agent.utils.json_utils import (
    extract_and_parse_json,
    JSONParseError,
)

from src.agents.experiment_agent.utils.print_utils import *


# Unifier instruction for CodeImplementOutput
CODE_IMPLEMENT_UNIFIER_INSTRUCTION = """You are an Output Formatter. Convert the structured implementation output into JSON.

## Input Format
The input follows this structure:
```
=== IMPLEMENTATION OUTPUT ===
IMPLEMENTATION_TYPE: initial/fix
=== GENERATED FILES ===
FILE: path
DESCRIPTION: ...
DEPENDENCIES: ...
CONTENT: ...
=== IMPLEMENTATION SUMMARY ===
FILES_CREATED: N
FILES_MODIFIED: N
...
=== TEST FILES ===
FILE: tests/test_xxx.py
DESCRIPTION: ...
CONTENT: ...
```

## Required JSON Output Format

```json
{
  "implementation_type": "initial",
  "generated_files": [
    {
      "file_path": "models/encoder.py",
      "content": "<full file content>",
      "description": "Description of this file",
      "dependencies": ["other/file.py"]
    }
  ],
  "implementation_summary": {
    "files_created": 2,
    "files_modified": 0,
    "total_lines": 150,
    "key_components": ["Component1", "Component2"]
  },
  "test_files": [
    {
      "file_path": "tests/test_encoder.py",
      "content": "<full test file content or empty string if not provided>",
      "description": "Unit tests for encoder",
      "dependencies": []
    }
  ],
  "issues_addressed": ""
}
```

### Rules:
1. Parse IMPLEMENTATION_TYPE -> `implementation_type`
2. Parse each FILE block -> `generated_files` array
3. Parse IMPLEMENTATION SUMMARY section -> `implementation_summary` object
4. Parse TEST FILES section -> `test_files`:
   - If NO test files mentioned: set to `null`
   - If test files listed but no content: set `content` to empty string `""`
   - If test files with content: include full content
   - **CRITICAL**: Each test_file MUST have `file_path`, `content`, `description`, `dependencies` fields
5. Parse ISSUES ADDRESSED -> `issues_addressed` (empty string if none)

Output ONLY valid JSON wrapped in ```json ... ``` block.
"""


def create_implement_output_unifier(model: str = None) -> Agent:
    """Create unifier agent to format implementation output."""
    if model is None:
        from src.agents.experiment_agent.config import UNIFIER_MODEL
        model = UNIFIER_MODEL
    return Agent(
        name="Code Implement Output Unifier",
        instructions=CODE_IMPLEMENT_UNIFIER_INSTRUCTION,
        model=model,
    )


def create_unified_implement_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: list = None,
) -> Agent:
    """
    Create unified implementation agent that follows code plan instructions.
    """

    instructions = f"""You are implementing ONE step of an Implementation Plan.

## WORKSPACE
| Path | Description |
|------|-------------|
| `{working_dir}/project` | Project root (write code here) |
| `{working_dir}/repos` | Reference code (read-only) |

---

## WORKFLOW: DISCOVER → IMPLEMENT → OUTPUT JSON

### 1️⃣ DISCOVER (Before Writing Code)

**Read before you write.** Use `read_file` to:
- Trace inputs: find where arguments are created, their types/keys
- Trace outputs: find consumers of your return values, verify expected format
- Verify calls: read function definitions before calling them

🚫 **NEVER** guess data structures. **NEVER** modify signatures without checking callers.

---

### 2️⃣ IMPLEMENT (Tool Execution)

**File Operations:**
- `list_directory` → check project state
- `run_shell_command mkdir -p` → create directories
- `write_file` → for NEW files (complete, functional code)
- `edit_file` → for EXISTING files (preferred for modifications)
- `list_directory` → verify files exist

**Tool Argument Rules:**
- `contents`/`new_content` must be a SINGLE LINE string
- Replace newlines with `\\n`, escape `"` as `\\"`, escape `\\` as `\\\\`

**Code Quality:**
- Use absolute imports: `from data.loader import X`
- Write complete code, not TODOs or placeholders
- Implement ONLY the current step
- Input validation: check None, ≤0, empty, wrong type
- Float comparison: use `torch.allclose(a, b, atol=1e-5)` not `==`

**⛔ ABSOLUTELY PROHIBITED - DO NOT CREATE THESE FILES:**
- `STEP*.json`, `*_COMPLETION*.json`, `*_EVALUATION*.json`, `*_SUMMARY*.json`
- `*_ISSUE*.json`, `*_ANALYSIS*.json`, `*_RESULT*.json`
- **ANY `.md` files**, **ANY summary/status/progress files**
- **NEVER write JSON files to track step completion or progress**

**📁 TEST FILE LOCATION:**
- ALL test/validation files MUST be placed in `tests/` directory only.

---

### 3️⃣ OUTPUT (MANDATORY - AS CHAT RESPONSE, NOT FILE!)

**🚨 CRITICAL OUTPUT RULES:**
- **DO NOT use `write_file` to write the output below!**
- **DO NOT write any JSON/summary/progress files to disk!**
- The format below is your **FINAL CHAT RESPONSE** - just type it directly as text!
- A separate unifier agent will convert your text response to JSON.

After completing all tool calls, **STOP calling tools** and output this TEXT directly in chat:

```
=== IMPLEMENTATION OUTPUT ===

IMPLEMENTATION_TYPE: initial  # or "fix"

=== GENERATED FILES ===

FILE: models/encoder.py
DESCRIPTION: Encoder module with attention mechanism
DEPENDENCIES: utils/config.py, data/loader.py
CONTENT:
import torch
import torch.nn as nn
...
[full file content here]

FILE: models/decoder.py
DESCRIPTION: Decoder module
DEPENDENCIES: models/encoder.py
CONTENT:
...

=== IMPLEMENTATION SUMMARY ===
FILES_CREATED: 2
FILES_MODIFIED: 0
TOTAL_LINES: 150
KEY_COMPONENTS: Encoder class, Decoder class, forward method

=== TEST FILES ===
FILE: tests/test_encoder.py
DESCRIPTION: Unit tests for encoder

=== ISSUES ADDRESSED ===
[If fix mode, describe what was fixed]
```

**🚨 REMINDER**: 
- This output format is your **CHAT RESPONSE** - just type it out!
- **DO NOT call write_file() with this content!**
- Include ALL generated file contents in full!
"""

    agent = Agent(
        name="Code Implementation Agent",
        instructions=instructions,
        model=model,
        tools=tools or [],
    )

    return agent


class CodeImplementAgent:
    """
    Main code implementation agent that follows code plan instructions.
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
            from src.agents.experiment_agent.sub_agents.code_implement import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize unified implementation agent
        self.implementation_agent = create_unified_implement_agent(
            model=model,
            working_dir=working_dir,
            tools=self.tools,
        )

        # Initialize output unifier agent
        self.output_unifier = create_implement_output_unifier(model=model)

        # Expose implementation agent as main agent for compatibility
        self.agent = self.implementation_agent

    def _get_checklist_from_plan(self, code_plan_output: Any) -> list:
        """Get implementation checklist from code plan output."""
        if not code_plan_output:
            return []
        
        checklist = code_plan_output.get("implementation_checklist", [])
        
        if checklist and isinstance(checklist[0], dict):
            class ChecklistStep:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
            checklist = [ChecklistStep(item) for item in checklist]
        
        return checklist

    def _extract_judge_feedback(self, judge_output: Any) -> str:
        """Extract feedback from judge output."""
        if not judge_output:
            return ""

        # Build feedback section
        feedback_section = "=== LATEST FEEDBACK (from current judge review) ===\n"
        issues = judge_output.get("issues", [])
        if issues:
            feedback_section += f"DETAILED ISSUES FROM CODE JUDGE: {len(issues)} found\n"
            for i, issue in enumerate(issues, 1):
                desc = issue.get("description", "No description")
                sugg = issue.get("suggestion", "No suggestion")
                file_p = issue.get("file_path", "Unknown file")
                severity = issue.get("severity", "unknown")
                expected = issue.get("expected", "N/A")
                actual = issue.get("actual", "N/A")
                
                feedback_section += f"\n--- Issue #{i} [{severity.upper()}] ---\n"
                feedback_section += f"File: {file_p}\n"
                feedback_section += f"Problem: {desc}\n"
                feedback_section += f"Expected: {expected}\n"
                feedback_section += f"Actual: {actual}\n"
                feedback_section += f"Fix: {sugg}\n"
            
            feedback_section += f"\n{'='*60}\n"

        return feedback_section
    
    def _extract_issue_history(self, context: Any) -> str:
        """Extract issue history from context's issue tracker."""
        if not hasattr(context, 'get_issue_tracker') or not hasattr(context, 'issue_tracker_data'):
            return ""
        
        if context.issue_tracker_data is None:
            return ""
        
        try:
            tracker = context.get_issue_tracker()
            return tracker.format_for_implement_agent()
        except Exception as e:
            print(f"[CODE_IMPLEMENT] Warning: Could not extract issue history: {e}")
            return ""

    async def process(self, context: Any, **kwargs) -> CodeImplementOutput:
        """
        Process the current step using context data.
        """
        # Extract data from context
        plan = getattr(context, "code_plan_output", None)
        checklist = self._get_checklist_from_plan(plan)
        current_step_idx = getattr(context, "current_checklist_step", 0)
        current_step = None
        checklist_progress = ""
        
        if checklist and current_step_idx < len(checklist):
            current_step = checklist[current_step_idx]
            checklist_progress = f"Step {current_step_idx + 1}/{len(checklist)}"
            
        # Determine feedback
        judge_output = None
        
        # Check for retry (retry count > 0 or explicit feedback type)
        retry_count = getattr(context, "checklist_step_retry_count", 0)
        pending_feedback_type = getattr(context, "pending_feedback_type", None)
        
        feedback_section = ""
        if retry_count > 0 or pending_feedback_type == "judge_rejection":
            judge_output = getattr(context, "code_judge_output", None)
            feedback_section = self._extract_judge_feedback(judge_output)

        # Build current step section
        step_section = ""
        if current_step:
            step_section = f"""
=== CURRENT STEP ===
Current Checklist Step: {checklist_progress}
Title: {current_step.title}
Description: {current_step.description}
Files to Create: {', '.join(current_step.files_to_create) if current_step.files_to_create else 'None'}
Files to Modify: {', '.join(current_step.files_to_modify) if current_step.files_to_modify else 'None'}
Acceptance Criteria:
{chr(10).join(f'  - {c}' for c in current_step.acceptance_criteria)}
"""



        # Extract file structure from plan
        file_structure_info = ""
        file_structure = context.code_plan_output.get("file_structure", []) if context.code_plan_output else []
        
        if file_structure:
            structure_lines = ["=== FILE STRUCTURE ===\n"]
            for item in file_structure:
                # Handle item being dict or object
                path = item.get("path") if hasattr(item, "get") else getattr(item, "path", None)
                if path:
                    structure_lines.append(f"- {path}")
            file_structure_info = "\n".join(structure_lines)

        core_plan_context = extract_core_plan_context(context.code_plan_output)
        
        # Extract code repos info from pre_analysis_output
        code_repos_info_section = ""
        if hasattr(context, "pre_analysis_output") and context.pre_analysis_output:
            code_repos_info = context.pre_analysis_output.get("code_repos_info", "")
            if code_repos_info:
                code_repos_info_section = f"=== REFERENCE CODE REPOSITORIES ===\n{code_repos_info}\n"
        
        # Extract issue history from tracker
        issue_history_section = self._extract_issue_history(context)

        mode_instructions = "You should follow the instructions to implement the code." if not feedback_section else f"You should follow the feedback to fix the code."
        
        # Add priority note for recurring issues
        priority_note = ""
        if issue_history_section:
            priority_note = """
⚠️ IMPORTANT: Review the ISSUE HISTORY section below. Issues marked as "RECURRING" have appeared multiple times 
and MUST be prioritized. Failing to address recurring issues will result in continued rejection.
"""
        
        input_prompt = f"""
{mode_instructions}
{priority_note}

{step_section}

{file_structure_info}

{code_repos_info_section}

{issue_history_section}

{feedback_section}

Global Context:
{core_plan_context}
"""

        return await self.implement(input_prompt)

    async def implement(self, input_data: str) -> CodeImplementOutput:
        """
        Generate code implementation based on code plan and current step.
        """
        print_section("CODE IMPLEMENTATION WORKFLOW", "=")

        print_subsection("Implementing Current Step")

        run_config = RunConfig(
            model_settings=ModelSettings(max_tokens=128*1024)
        )

        implementation_stream = Runner.run_streamed(
            self.implementation_agent, 
            input_data, 
            hooks=self.hooks, 
            max_turns=100,
            run_config=run_config
        )
        final_text = ""
        async for event in implementation_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if hasattr(event.data, "delta"):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                        final_text += delta.content
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
                        final_text += delta.text

        implementation_result = implementation_stream
        # Handle case where text is in chat history or final_output
        if hasattr(implementation_result, "final_output") and isinstance(
            implementation_result.final_output, str
        ):
            final_text = implementation_result.final_output

        # If no text captured from stream, search chat_history for assistant text messages
        if (
            not final_text
            and hasattr(implementation_result, "chat_history")
            and implementation_result.chat_history
        ):
            # Iterate backwards to find the last assistant message with actual text content
            for msg in reversed(implementation_result.chat_history):
                if hasattr(msg, "role") and msg.role == "assistant":
                    if (
                        hasattr(msg, "content")
                        and msg.content
                        and isinstance(msg.content, str)
                        and len(msg.content) > 10
                    ):
                        final_text = msg.content
                        break

        print_subsection("Unifying Output")

        # Use unifier agent to convert raw output to structured JSON
        unifier_prompt = f"""Convert the following implementation output to JSON:

=== RAW OUTPUT START ===
{final_text}
=== RAW OUTPUT END ===

Extract all file information and output the structured JSON:"""

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
            final_output = extract_and_parse_json(unified_text, CodeImplementOutput, raise_on_failure=True)
        except JSONParseError as e:
            print_error(f"JSON parsing failed, will trigger retry: {e}")
            raise

        print_success("Implementation completed")
        print_info(f"Generated {len(final_output.generated_files)} files")
        print_section("CODE IMPLEMENTATION COMPLETE", "=")

        return final_output

    def implement_sync(self, input_data: str) -> CodeImplementOutput:
        import asyncio

        return asyncio.run(self.implement(input_data))


def create_code_implement_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: Optional[list] = None,
    verbose: bool = False,
) -> CodeImplementAgent:
    """
    Factory function to create a code implementation agent.
    """
    return CodeImplementAgent(
        model=model, working_dir=working_dir, tools=tools, verbose=verbose
    )
