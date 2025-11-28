"""
Code Implementation Agent - Unified implementation agent.

This agent implements code strictly following the code plan's instructions.
It handles both initial implementation and iterative fixes based on feedback,
all guided by the code plan from the code_plan agent.
"""

import os
from typing import Dict, Optional, Any

from agents import Agent, Runner
from src.agents.experiment_agent.sub_agents.code_implement.output_schemas import (
    CodeImplementOutput,
)
from src.agents.experiment_agent.utils.common_utils import extract_core_plan_context
from src.agents.experiment_agent.logger import create_verbose_hooks
from src.agents.experiment_agent.config import OUTPUT_UNIFIER_MODEL

from src.agents.experiment_agent.utils.print_utils import *


def create_unified_implement_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: list = None,
) -> Agent:
    """
    Create unified implementation agent that follows code plan instructions.
    """

    instructions = f"""You are the Lead Developer executing ONE step of the Implementation Plan.

### CRITICAL: TWO-PHASE EXECUTION MODEL

⚠️ **PHASE 1: PHYSICAL EXECUTION** (Tools Required)
You MUST complete ALL actions using tools BEFORE returning any output:

**1.1 Analysis Phase:**
   - Use `list_directory` to check current project state
   - Identify what needs to be created, modified, or fixed
   - Check `files_to_create` and `files_to_modify` in the task

**1.2 Directory Setup:**
   - Use `run_shell_command` with `mkdir -p` to create required directories
   - Ensure all parent directories exist before creating files

**1.3 File Operations:**
   - For NEW files: Use `write_file` to create each file
   - For EXISTING files: Use `read_file` first, then `write_file` to update
   - For FIXES: Read error context, modify code, write back
   - Write complete, functional code (not placeholders or comments)
   - Use absolute imports: `from data.loader import X` (not relative imports)

**1.4 Verification:**
   - Use `list_directory` to confirm all required files exist
   - Use `read_file` to spot-check critical files if needed
   - Verify acceptance criteria are met

🚫 **DO NOT PROCEED TO PHASE 2 UNTIL ALL FILES ARE PHYSICALLY CREATED/MODIFIED!**

---

⚠️⚠️⚠️ **PHASE 2: TEXT REPORT IS MANDATORY** ⚠️⚠️⚠️

**YOU MUST ALWAYS END WITH A TEXT RESPONSE.** After completing all tool calls, you MUST write a detailed implementation report as plain text. This is NOT optional - the system will FAIL if you only call tools without returning text.

Your final text response MUST include:

```
## Implementation Report

### Execution Summary
[Describe what was implemented in this step]

### Files Created
- path/to/file1.py: [brief description]
- path/to/file2.py: [brief description]

### Files Modified
- path/to/existing.py: [what was changed]

### Key Components Implemented
- ClassName: [purpose]
- function_name: [purpose]

### Implementation Notes
[Any important details about the implementation]

### Issues Addressed
[Any feedback or bugs that were fixed, or "None" if not applicable]
```

🚨 **CRITICAL**: Even if all tool calls succeed, you MUST write this text report. No text report = SYSTEM FAILURE.

### CONTEXT
- **Project Root**: `{working_dir}/project`.
- **Workspace**: `{working_dir}`.
- **Reference Code**: `{working_dir}/repos` (Read-only).

### CRITICAL CONSTRAINTS
1. **Tool Usage is Mandatory**: You must call tools to create/modify files
2. **Text Report is Mandatory**: You must end with a text report (see Phase 2)
3. **Scope Control**: Implement ONLY the current step, not future steps
4. **Content Quality**: Write functional code, not TODOs or placeholders
5. **Import Style**: Always use absolute imports from project root

### SELF-CHECK BEFORE FINISHING
- ✓ Did I call `write_file` for every required file?
- ✓ Did I verify files exist using `list_directory`?
- ✓ **Did I write a text report summarizing what I did?** ← MOST IMPORTANT
"""

    agent = Agent(
        name="Code Implementation Agent",
        instructions=instructions,
        # output_type=CodeImplementOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


def create_code_implement_unifier_agent(model: str = "gpt-4o") -> Agent:
    return Agent(
        name="Code Implement Unifier",
        instructions="""You are an expert data structuring assistant.
Your task is to convert the implementation report into a structured `CodeImplementOutput` object.

Input text will contain:
- Execution Summary
- Files Created/Modified
- Key Components
- Notes
- Issues Addressed

Map these to the schema.
For `generated_files`:
- Extract `file_path` and `description`.
- Set `content` to an empty string "" (it will be filled by the system).
- Set `dependencies` based on imports if mentioned, or empty list.

For `implementation_summary`:
- Extract `files_created`, `files_modified` counts.
- `key_components` list.
- `implementation_notes`.
""",
        output_type=CodeImplementOutput,
        model=OUTPUT_UNIFIER_MODEL,
    )


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

        # Initialize output unifier
        self.output_unifier = create_code_implement_unifier_agent(model=model)

        # Expose implementation agent as main agent for compatibility
        self.agent = self.implementation_agent

    async def process(self, context: Any, **kwargs) -> CodeImplementOutput:
        """
        Process the current step using context data.
        """
        data = kwargs
        current_step = data.get("current_step", None)
        checklist_progress = data.get("checklist_progress", "")
        completed_steps = data.get("completed_steps", [])
        feedback = data.get("feedback", "")
        judge_output = data.get("judge_output", None)

        # Build current step section
        step_section = ""
        if current_step:
            step_section = f"""
=== CURRENT STEP ===
ID: {current_step.step_id}
Title: {current_step.title}
Description: {current_step.description}
Files to Create: {', '.join(current_step.files_to_create) if current_step.files_to_create else 'None'}
Files to Modify: {', '.join(current_step.files_to_modify) if current_step.files_to_modify else 'None'}
Acceptance Criteria:
{chr(10).join(f'  - {c}' for c in current_step.acceptance_criteria)}
Dependencies: {', '.join(map(str, current_step.dependencies)) if current_step.dependencies else 'None'}
"""

        # Build feedback section if code was rejected
        feedback_section = ""
        if feedback or judge_output:
            feedback_section = f"\n=== FEEDBACK ===\n{feedback}\n"
            if judge_output and hasattr(judge_output, "issues"):
                feedback_section += (
                    f"\nJudge Issues: {len(judge_output.issues)} found.\n"
                )

        # Get reference codebases information
        reference_codebases_info = data.get(
            "reference_codebases_info", "(Codebase information not available)"
        )

        # Extract file structure from plan
        file_structure_info = ""
        if hasattr(context.code_plan_output, "file_structure"):
            structure_lines = ["=== FILE STRUCTURE ===\n"]
            for item in context.code_plan_output.file_structure:
                if hasattr(item, "path"):
                    structure_lines.append(f"- {item.path}")
            file_structure_info = "\n".join(structure_lines)

        core_plan_context = extract_core_plan_context(context.code_plan_output)

        input_prompt = f"""
IMPLEMENTATION TASK

{step_section}

{file_structure_info}

{feedback_section}

Reference Info:
{reference_codebases_info}

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

        # Use streamed version for real-time output
        implementation_stream = Runner.run_streamed(
            self.implementation_agent, input_data, hooks=self.hooks, max_turns=100
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

        # If still no text, the agent only made tool calls without a final report
        # This is acceptable - we'll generate a minimal report based on tool activity
        if not final_text:
            print_warning(
                "Agent did not produce text output, generating minimal report..."
            )
            # Create a minimal report based on what we know
            final_text = """## Implementation Report

### Execution Summary
Implementation step executed via tool calls. The agent performed file operations but did not generate a detailed text report.

### Files Created
See tool call logs for details.

### Implementation Notes
The implementation was performed through tool calls. Please verify the files were created correctly.
"""

        print_success("Implementation text report generated")
        print_subsection("Unifying Output Format")

        # Unify
        unifier_input = f"""
Please convert the following implementation report into the structured `CodeImplementOutput` format.

=== IMPLEMENTATION REPORT ===
{final_text}
"""
        unifier_stream = Runner.run_streamed(
            self.output_unifier, unifier_input, hooks=None
        )

        async for _ in unifier_stream.stream_events():
            pass

        final_output = unifier_stream.final_output

        project_root = os.path.join(self.working_dir, "project")

        print_info("Reading generated files from disk...")
        for gen_file in final_output.generated_files:
            # Handle absolute paths that might be returned by LLM
            raw_path = gen_file.file_path.strip()

            # If it's an absolute path and starts with project_root, use it directly
            # Also handle potential double-slash issues or symlinks by resolving
            try:
                # Normalize paths for comparison
                norm_project_root = os.path.normpath(project_root)
                norm_raw_path = os.path.normpath(raw_path)

                if os.path.isabs(norm_raw_path) and norm_raw_path.startswith(
                    norm_project_root
                ):
                    full_path = norm_raw_path
                    # Update relative path for display/storage consistency
                    rel_path = os.path.relpath(full_path, start=project_root)
                else:
                    # Treat as relative path
                    rel_path = raw_path.lstrip("/").lstrip("\\")
                    full_path = os.path.join(project_root, rel_path)
            except Exception:
                # Fallback to original behavior if path manipulation fails
                rel_path = raw_path.lstrip("/").lstrip("\\")
                full_path = os.path.join(project_root, rel_path)

            if os.path.exists(full_path) and os.path.isfile(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        gen_file.content = f.read()
                except Exception as e:
                    gen_file.content = f"[Error reading file: {str(e)}]"
                    print_error(f"Failed to read {rel_path}: {e}")
            else:
                gen_file.content = "[File not found on disk]"
                print_error(f"File not found: {full_path}")

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


if __name__ == "__main__":
    import asyncio

    async def main():
        agent = create_code_implement_agent(model="gpt-4o", working_dir="/workspace")
        print("Agent created.")

    asyncio.run(main())
