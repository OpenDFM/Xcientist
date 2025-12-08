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
    generate_json_schema_instruction,
    JSONParseError,
)

from src.agents.experiment_agent.utils.print_utils import *


# Generate JSON output instruction for CodeImplementOutput
CODE_IMPLEMENT_JSON_OUTPUT_INSTRUCTION = generate_json_schema_instruction(CodeImplementOutput)


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

---

### 3️⃣ OUTPUT JSON (MANDATORY)

After completing all tool calls, you **MUST** output a JSON object.

---

## 🚫 PROHIBITED

**DO NOT create these files:**
- `STEP*_COMPLETION*.json`, `STEP*_REPORT*.json`
- `*_EVALUATION*.json`, `*_SUMMARY*.json`, `*_SUMMARY*.md`
- Any report/completion markdown files

**Test files MUST go in `tests/` directory only.**

---

## ⚠️ CRITICAL: REQUIRED OUTPUT FORMAT

🚨🚨🚨 **YOUR FINAL OUTPUT MUST BE ONLY A JSON OBJECT** 🚨🚨🚨

**DO NOT** write markdown summaries like "I have successfully completed..." or "✅ Step Complete".
**DO NOT** write any explanatory text after completing tool calls.
**ONLY** output a valid JSON wrapped in ```json ... ``` code block.

**REQUIRED JSON STRUCTURE:**
```json
{{
  "implementation_type": "initial",
  "timestamp": "2025-12-08T12:00:00",
  "generated_files": [
    {{
      "file_path": "project/path/to/file.py",
      "content": "",
      "description": "Description of file",
      "dependencies": []
    }}
  ],
  "implementation_summary": {{
    "files_created": 5,
    "files_modified": 0,
    "total_lines": 500,
    "key_components": ["component1", "component2"]
  }},
  "test_files": [],
  "issues_addressed": ""
}}
```

**JSON Field Guide:**
- `implementation_type`: "initial" or "fix"
- `timestamp`: ISO format datetime
- `generated_files`: List with file_path, content (can be ""), description, dependencies
- `implementation_summary`: files_created, files_modified, total_lines, key_components
- `test_files`: empty list (tests handled by Code Judge)
- `issues_addressed`: issues fixed (for fix type only)

❌ WRONG: "I have completed the task! Here's what I did: ..."
✅ CORRECT: Only output the JSON block above, nothing else.

**If you output markdown text instead of JSON, the system will FAIL and retry.**
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
                    ):
                        # Skip if it looks like a tool call response
                        if not msg.content.startswith("{") and len(msg.content) > 50:
                            final_text = msg.content
                            break

        print_subsection("Parsing JSON Output")

        # Extract and parse JSON from the implementation output
        # Use raise_on_failure=True to trigger retry in master agent
        try:
            final_output = extract_and_parse_json(final_text, CodeImplementOutput, raise_on_failure=True)
        except JSONParseError as e:
            # Re-raise JSONParseError to trigger retry in master agent
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
