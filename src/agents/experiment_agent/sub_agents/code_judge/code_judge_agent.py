"""
Code Judge Agent - Reviews code implementation for consistency with plan and analysis.

This agent evaluates whether the implemented code matches the specifications
from the code plan and pre-analysis, identifying issues and providing feedback.
"""

import os
from typing import Optional, Any

from agents import Agent, Runner, RunConfig, ModelSettings
from src.agents.experiment_agent.sub_agents.code_judge.output_schemas import (
    CodeJudgeOutput,
)
from src.agents.experiment_agent.logger import create_verbose_hooks
from src.agents.experiment_agent.utils.common_utils import (
    read_file_smart,
    extract_core_plan_context,
    extract_analysis_summary,
)
from src.agents.experiment_agent.utils.json_utils import (
    extract_and_parse_json,
    JSONParseError,
)
from src.agents.experiment_agent.utils.repo_map import generate_repo_map

from src.agents.experiment_agent.utils.print_utils import *


# Unifier instruction for CodeJudgeOutput
CODE_JUDGE_UNIFIER_INSTRUCTION = """You are an Output Formatter. Convert the structured code review output into JSON.

## Input Format
The input follows this structure:
```
=== CODE JUDGE OUTPUT ===
IS_CONSISTENT: true/false
=== ISSUES ===
ISSUE #N:
FILE_PATH: ...
ISSUE_TYPE: ...
SEVERITY: ...
...
=== UNIT TESTS ===
TEST #N:
TEST_FILE_PATH: ...
...
=== IMPLEMENTATION SUGGESTIONS ===
- suggestion1
- suggestion2
```

## Required JSON Output Format

```json
{
  "is_consistent": false,
  "issues": [
    {
      "file_path": "models/encoder.py",
      "issue_type": "logic_error",
      "severity": "major",
      "description": "Problem description",
      "expected": "What should happen",
      "actual": "What actually happens",
      "suggestion": "How to fix it",
      "line_numbers": "45-52"
    }
  ],
  "unit_tests": [
    {
      "test_file_path": "tests/test_encoder.py",
      "test_code": "<test code>",
      "test_description": "What this test validates",
      "target_files": ["models/encoder.py"],
      "time_limit_seconds": 30,
      "data_subset_size": null
    }
  ],
  "implementation_suggestions": ["Suggestion 1", "Suggestion 2"]
}
```

### Rules:
1. Parse IS_CONSISTENT -> `is_consistent` (boolean)
2. Parse each ISSUE block -> `issues` array (null if no issues)
3. Parse each TEST block -> `unit_tests` array (null if no tests)
4. Parse IMPLEMENTATION SUGGESTIONS -> `implementation_suggestions` array
5. **CRITICAL**: `time_limit_seconds` and `data_subset_size` MUST be integers or null, NEVER strings!
   - ✅ Correct: `"data_subset_size": 10` or `"data_subset_size": null`
   - ❌ Wrong: `"data_subset_size": "Full test coverage"` (THIS WILL CAUSE ERROR!)

Output ONLY valid JSON wrapped in ```json ... ``` block.
"""


def create_judge_output_unifier(model: str = None) -> Agent:
    """Create unifier agent to format judge output."""
    if model is None:
        from src.agents.experiment_agent.config import UNIFIER_MODEL

        model = UNIFIER_MODEL
    return Agent(
        name="Code Judge Output Unifier",
        instructions=CODE_JUDGE_UNIFIER_INSTRUCTION,
        model=model,
    )


def create_judge_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: Optional[list] = None
) -> Agent:
    """
    Create the code judge agent for reviewing implementation.
    """

    instructions = f"""You are the QA Engineer certifying code quality. **ZERO TOLERANCE** for issues.

## WORKFLOW: 1️⃣ INSPECT → 2️⃣ TEST → 3️⃣ JUDGE

---

## 1️⃣ INSPECT (Interface Audit)

**Use Search → Zoom workflow:**
1. `grep("pattern", ".")` - Find code (returns file:line)
2. `file_viewer("file.py", start_line=N)` - View around line N

| Check | How | Flag If |
|-------|-----|---------|
| Imports | `grep("from.*import", "file.py")` | Import doesn't exist |
| Data contracts | Trace producer→consumer | Output format ≠ expected input format |
| Signatures | Compare definition vs calls | Mismatch in params/types |
| Breaking changes | Check callers of modified APIs | Callers not updated |
| Torch dimensions | Check tensor ops | Hardcoded shapes / Missing validation |

**🔥 TORCH CODE INSPECTION (CRITICAL):**
When inspecting code with torch tensor operations, you MUST first think about the dimension flow table:

**Dimension Flow Analysis Example:**
```
[DIMENSION FLOW ANALYSIS]
Input: [B, C, H, W]
↓ Conv2d(C→64, k=3)
Conv_out: [B, 64, H, W]
↓ Flatten
Flattened: [B, 64*H*W]  <- Check if Linear layer matches this!
↓ Linear(64*H*W→hidden_dim)
Output: [B, hidden_dim]
```

Then verify the code:
1. ✅ Uses dynamic dimensions (x.size(), x.shape[...])
2. ✅ Has shape assertions in forward()
3. ✅ **MANDATORY**: Each tensor operation line has inline comments showing tensor shapes
   - Use variable names or symbols (B=batch, S=seq_len, H=hidden_dim, etc.) instead of hardcoded numbers
   - Example: `x = self.linear(x)  # x: [B, S, H_in] -> [B, S, H_out]`
   - Example: `out = torch.cat([a, b], dim=-1)  # a: [B, H1], b: [B, H2] -> out: [B, H1+H2]`
4. ❌ Contains hardcoded dimensions (e.g., self.fc = nn.Linear(1024, ...))
5. ❌ Assumes fixed batch_size
6. ❌ Missing shape comments after tensor operations


🚫 Flag as **CRITICAL** if Implement Agent didn't read related files before coding.
🚫 Flag as **MAJOR** if torch code uses hardcoded shapes that may cause dimension mismatch.
---

## 2️⃣ TEST (Dynamic Verification)

🚨 **TEST FILES MUST BE IN `tests/` DIRECTORY** 🚨
- ✅ CORRECT: `{working_dir}/project/tests/test_step_X.py`
- ❌ WRONG: `{working_dir}/project/test_*.py` (NOT in project root!)

**Execute ALL tests:**
1. Create/modify test files:
   - **For NEW test files**: Use `write_file(file_path="{working_dir}/project/tests/test_step_X.py", contents=code)`
   - **For EXISTING test files**: Prefer `edit_file` to modify specific test cases instead of rewriting the entire file
   - ⚠️ **CRITICAL JSON SYNTAX RULES** (for both tools): 
     - The `contents`/`new_content` argument MUST be a SINGLE LINE string in the JSON payload.
     - Replace ALL actual newlines in the code with `\\n`.
     - Escape ALL double quotes (`"`) as `\\"`.
     - Escape ALL backslashes (`\\`) as `\\\\`.
     - DO NOT include actual line breaks inside the JSON string value.
2. `run_pytest_local("tests/", working_dir="{working_dir}/project")`

**Required Test Types:**

| Test | Purpose |
|------|---------|
| Interface Contract | Producer output matches consumer input |
| Integration | Real data flow between components |
| Smoke | Main entrypoint runs without crash |

**Test Constraints:**
- Include `sys.path.append(os.getcwd())` for imports
- Keep tests SHORT (< 30s, mock heavy resources)

---

## 3️⃣ JUDGE (Decision)

**Decision Matrix (NO EXCEPTIONS):**

| Condition | is_consistent | Score Constraint |
|-----------|---------------|------------------|
| ANY test fails | **False** | < 0.5 |
| ANY critical issue | **False** | < 0.5 |
| ANY major issue | **False** | < 0.6 |
| ANY minor issue | **False** | < 0.7 |
| ALL pass + zero issues | **True** | ≥ 0.8 |

---

## FAILURE PATTERNS

| Pattern | Example |
|---------|---------|
| Contract violation | Function returns dict, caller expects object |
| Undefined reference | Import/variable/field not defined |
| Integration gap | Works isolated, fails connected |
| Missing validation | No checks for None/empty/wrong type |

---

## ⚠️ PREVIOUSLY IDENTIFIED ISSUES

If the input includes a "PREVIOUSLY IDENTIFIED ISSUES" section:
1. **CHECK EACH ISSUE** - Verify if it has been resolved in the current implementation
2. **If RESOLVED** - Do NOT include it in your output (the system will auto-detect resolution)
3. **If STILL PRESENT** - Include it in your output with updated details
4. **RECURRING issues** are HIGH PRIORITY - mark them appropriately

---

## 🔍 ISSUE REPORTING REQUIREMENTS (MANDATORY)

When reporting issues, you MUST provide **SPECIFIC, ACTIONABLE** details:

### REQUIRED for EVERY issue:
1. **Exact file path** (e.g., `models/encoder.py`)
2. **Line numbers** if possible (e.g., "lines 45-52")
3. **Concrete code example** showing the problem
4. **Concrete fix suggestion** with code example

### GOOD Example:
```
[MAJOR] models/encoder.py (lines 45-52):
  Problem: Encoder uses fixed flattened_size=1024 but input grid_size=64 produces 256 features.
  Current code:
    self.fc = nn.Linear(1024, hidden_dim)  # Wrong: hardcoded
  Expected fix:
    # Option 1: Use adaptive pooling
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(channels, hidden_dim)
    
    # Option 2: Compute dynamically
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, grid_size, grid_size)
        flattened_size = self.conv(dummy).view(1, -1).shape[1]
    self.fc = nn.Linear(flattened_size, hidden_dim)
```

### BAD Example (DO NOT DO THIS):
```
[MAJOR] encoder.py: The encoder has size mismatch issues.
  Suggestion: Fix the size calculation.
```

---

## PRINCIPLE

**When in doubt, REJECT.** You are the last line of defense.

---

## 4️⃣ OUTPUT (MANDATORY - AS CHAT RESPONSE, NOT FILE!)

**🚨 CRITICAL OUTPUT RULES:**
- **DO NOT use `write_file` to write the output below!**
- **DO NOT write any JSON/summary/progress files to disk!**
- The format below is your **FINAL CHAT RESPONSE** - just type it directly as text!
- A separate unifier agent will convert your text response to JSON.

After completing your evaluation, **STOP calling tools** and output this TEXT directly in chat:

```
=== CODE JUDGE OUTPUT ===

IS_CONSISTENT: false  # true ONLY if ALL tests pass AND zero issues

=== ISSUES ===

ISSUE #1:
FILE_PATH: models/encoder.py
ISSUE_TYPE: logic_error  # logic_error, missing_implementation, inconsistency, quality
SEVERITY: major  # critical, major, minor
LINE_NUMBERS: 45-52
DESCRIPTION: Encoder uses fixed flattened_size=1024 but input produces 256 features
EXPECTED: Dynamic calculation based on input size
ACTUAL: Hardcoded value 1024
SUGGESTION: Use nn.AdaptiveAvgPool2d or compute dynamically

ISSUE #2:
FILE_PATH: ...
...

=== UNIT TESTS ===

TEST #1:
TEST_FILE_PATH: tests/test_encoder.py
TEST_DESCRIPTION: Test encoder output shape matches expected dimensions
TARGET_FILES: models/encoder.py
TIME_LIMIT_SECONDS: 30  # integer only
DATA_SUBSET_SIZE: 10  # integer or null, NOT a string description!
TEST_CODE:
import pytest
import torch
from models.encoder import Encoder

def test_encoder_output_shape():
    ...

TEST #2:
...

=== IMPLEMENTATION SUGGESTIONS ===
- Consider adding input validation
- Add docstrings to public methods
```

**🚨 REMINDER**: 
- This output format is your **CHAT RESPONSE** - just type it out!
- **DO NOT call write_file() with this content!**

**Decision Rules:**
- `IS_CONSISTENT: false` if ANY test fails OR ANY issue exists
- `IS_CONSISTENT: true` ONLY if ALL tests pass AND zero issues
"""

    agent = Agent(
        name="Code Judge Agent",
        instructions=instructions,
        tools=tools or [],
        # output_type=CodeJudgeOutput,  # Removed for duplex mode
        model=model,
    )

    return agent


class CodeJudgeAgent:
    """
    Main code judge agent that reviews implementation consistency.
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
            from src.agents.experiment_agent.sub_agents.code_judge import (
                get_recommended_tools,
            )

            self.tools = get_recommended_tools()
        else:
            self.tools = tools

        # Initialize judge agent
        self.judge_agent = create_judge_agent(
            model=model, working_dir=working_dir, tools=self.tools
        )

        # Initialize output unifier agent
        self.output_unifier = create_judge_output_unifier(model=model)

        # Expose judge agent as main agent for handoff compatibility
        self.agent = self.judge_agent

    def _get_checklist_from_plan(self, code_plan_output: Any) -> list:
        """Get implementation checklist from code plan output."""
        if not code_plan_output:
            return []

        # Unified access
        checklist = code_plan_output.get("implementation_checklist", [])

        # Convert dict items to objects for consistent attribute access
        if checklist and isinstance(checklist[0], dict):

            class ChecklistStep:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)

            checklist = [ChecklistStep(item) for item in checklist]

        return checklist

    def _extract_issue_history(self, context: Any) -> str:
        """Extract issue history from context's issue tracker for judge agent."""
        if not hasattr(context, "get_issue_tracker") or not hasattr(
            context, "issue_tracker_data"
        ):
            return ""

        if context.issue_tracker_data is None:
            return ""

        try:
            tracker = context.get_issue_tracker()
            return tracker.format_for_judge_agent()
        except Exception as e:
            print(f"[CODE_JUDGE] Warning: Could not extract issue history: {e}")
            return ""

    async def process(self, context: Any, **kwargs) -> CodeJudgeOutput:
        """
        Process the current step using context data.
        """
        # Extract data from context
        plan = context.get("code_plan_output", None)
        checklist = self._get_checklist_from_plan(plan)

        current_step_idx = context.get("current_checklist_step", 0)
        current_step = None
        checklist_progress = ""

        if checklist and current_step_idx < len(checklist):
            current_step = checklist[current_step_idx]
            checklist_progress = f"Step {current_step_idx + 1}/{len(checklist)}"

        # === BUILD INPUT PROMPT (Prioritized Structure) ===

        # 1. TASK SECTION (What to evaluate)
        task_section = ""
        files_to_review = []
        if current_step:
            if current_step.files_to_create:
                files_to_review.extend(current_step.files_to_create)
            if current_step.files_to_modify:
                files_to_review.extend(current_step.files_to_modify)
            files_to_review = list(set(files_to_review))

            files_list = (
                "\n".join(f"- `{f}`" for f in files_to_review)
                if files_to_review
                else "- None"
            )

            task_section = f"""
## 🎯 EVALUATION TASK ({checklist_progress})

**{current_step.title}**

{current_step.description}

### Files to Review:
{files_list}
"""

        # 2. ACCEPTANCE CRITERIA (What to verify)
        criteria_section = ""
        if current_step and current_step.acceptance_criteria:
            criteria_items = "\n".join(
                f"- [ ] {c}" for c in current_step.acceptance_criteria
            )
            criteria_section = f"""
## ✅ ACCEPTANCE CRITERIA (Verify Each)

{criteria_items}
"""

        # 3. ISSUE HISTORY (Previously identified issues)
        issue_history_section = self._extract_issue_history(context)
        history_section = ""
        if issue_history_section:
            history_section = f"""
## 📜 PREVIOUSLY IDENTIFIED ISSUES

Check if each issue has been resolved in the current implementation:

{issue_history_section}
"""

        # 4. CODE SKELETON (Repo Map - overview of all code)
        repo_map_section = ""
        if self.working_dir:
            project_dir = os.path.join(self.working_dir, "project")
            if os.path.exists(project_dir):
                repo_map = generate_repo_map(project_dir, max_files=30)
                if repo_map and "[No Python files found" not in repo_map:
                    repo_map_section = f"""
## 🔗 CODE SKELETON (All Project Files)

{repo_map}
"""

        # 5. BACKGROUND (Design context and analysis)
        core_plan_context = extract_core_plan_context(context.code_plan_output)
        analysis_summary = extract_analysis_summary(context.pre_analysis_output)

        background_section = f"""
## 📚 BACKGROUND

### Design Context
{core_plan_context}

### Research Requirements
{analysis_summary}
"""

        # 6. INSTRUCTIONS
        instructions_section = f"""
## 📋 INSTRUCTIONS

1. **READ**: Use `file_viewer` to inspect the files listed above
2. **VERIFY**: Check each acceptance criterion
3. **TEST**: Write and run unit tests in `tests/` directory
4. **JUDGE**: Report issues with SPECIFIC details (file, line, code example, fix suggestion)

**Project Path**: `{self.working_dir}/project`
"""

        # Assemble final prompt
        input_data = f"""
{task_section}
{criteria_section}
{history_section}
{repo_map_section}
{background_section}
{instructions_section}
"""

        return await self.judge(input_data)

    async def judge(
        self,
        input_data: str,
    ) -> CodeJudgeOutput:
        """
        Evaluate code implementation for consistency with current step.
        """
        print_section("CODE REVIEW WORKFLOW", "=")
        print_info(f"Input length: {len(input_data)} characters")

        judge_input = f"""
{input_data}

### EXECUTION REQUIRED
You must strictly follow the WORKFLOW defined in your system instructions.
1. INSPECT: `file_viewer` target files.
2. TEST: `write_file` (ensure valid JSON escaping: \\n for newlines, \\" for quotes) -> `run_pytest_local`.
3. JUDGE: Return JSON output based on results.

Current Working Directory: `{self.working_dir}`
Project Root: `{self.working_dir}/project`
"""

        print_subsection("Evaluating Current Step Implementation")
        print_info("Reviewing code and running verification tests...")

        run_config = RunConfig(model_settings=ModelSettings(max_tokens=128 * 1024))

        # Use streamed version for real-time output
        result_stream = Runner.run_streamed(
            self.judge_agent,
            judge_input,
            hooks=self.hooks,
            max_turns=100,
            run_config=run_config,
        )
        final_text = ""
        async for event in result_stream.stream_events():
            if hasattr(event, "data"):
                event_type = type(event.data).__name__
                if "FunctionCallArguments" not in event_type and hasattr(
                    event.data, "delta"
                ):
                    delta = event.data.delta
                    if hasattr(delta, "content") and delta.content:
                        print(delta.content, end="", flush=True)
                        final_text += delta.content
                    elif hasattr(delta, "text") and delta.text:
                        print(delta.text, end="", flush=True)
                        final_text += delta.text

        # In duplex mode, the first agent returns text, not the structured object
        # We need to ensure we capture the full text output if not accumulated
        if hasattr(result_stream, "final_output") and isinstance(
            result_stream.final_output, str
        ):
            final_text = result_stream.final_output
        elif not final_text and hasattr(result_stream, "chat_history"):
            # Fallback: get last message
            final_text = result_stream.chat_history[-1].content

        print_success("Judge analysis completed.")
        print_subsection("Unifying Output")

        # Use unifier agent to convert raw output to structured JSON
        unifier_prompt = f"""Convert the following code review output to JSON:

=== RAW OUTPUT START ===
{final_text}
=== RAW OUTPUT END ===

Extract all issues, test results, and output the structured JSON:"""

        unifier_result = await Runner.run(
            self.output_unifier,
            unifier_prompt,
            run_config=RunConfig(model_settings=ModelSettings(max_tokens=64 * 1024)),
        )

        unified_text = ""
        if hasattr(unifier_result, "final_output") and isinstance(
            unifier_result.final_output, str
        ):
            unified_text = unifier_result.final_output
        elif hasattr(unifier_result, "chat_history") and unifier_result.chat_history:
            unified_text = unifier_result.chat_history[-1].content

        print_subsection("Parsing JSON Output")

        # Extract and parse JSON from the unified output
        try:
            evaluation = extract_and_parse_json(
                unified_text, CodeJudgeOutput, raise_on_failure=True
            )
        except JSONParseError as e:
            print_error(f"JSON parsing failed, will trigger retry: {e}")
            raise
        except Exception as e:
            print_error(f"Failed to parse JSON output: {e}")
            print_warning("Creating minimal evaluation structure...")
            evaluation = CodeJudgeOutput(
                is_consistent=False,
                issues=[],
                unit_tests=[],
                implementation_suggestions=[
                    f"JSON parsing failed: {str(e)}. Re-run evaluation."
                ],
            )

        # Display evaluation results
        print_subsection("Evaluation Results")

        if evaluation.is_consistent:
            print_success(
                f"Code review passed! (Consistency: {evaluation.is_consistent})"
            )
        else:
            print_error(
                f"Code review failed! (Consistency: {evaluation.is_consistent})"
            )
            print_warning(f"Issues found: {len(evaluation.issues)}")

        if evaluation.issues:
            print_subsection("Issues Identified")
            for i, issue in enumerate(evaluation.issues, 1):
                severity_color = (
                    Colors.FAIL
                    if issue.severity == "critical"
                    else Colors.WARNING if issue.severity == "major" else Colors.OKBLUE
                )
                print(
                    f"{severity_color}Issue {i}: [{issue.severity.upper()}] {issue.issue_type}{Colors.ENDC}"
                )
                print(f"  File: {issue.file_path}")
                if hasattr(issue, "line_numbers") and issue.line_numbers:
                    print(f"  Lines: {issue.line_numbers}")
                print(f"  Description: {issue.description}")
                if hasattr(issue, "suggestion") and issue.suggestion:
                    print(f"  Suggestion: {issue.suggestion}")
                print()

        if evaluation.unit_tests:
            print_subsection("Unit Tests Generated")
            print_info(
                f"Generated {len(evaluation.unit_tests)} unit test(s) for validation"
            )
            for i, test in enumerate(evaluation.unit_tests, 1):
                print(f"\n{Colors.OKCYAN}Test {i}: {test.test_file_path}{Colors.ENDC}")
                print(f"  Description: {test.test_description}")
                print(f"  Target files: {', '.join(test.target_files)}")
                print(f"  Time limit: {test.time_limit_seconds}s")

        print_success("Code review completed!")
        print_section("CODE REVIEW COMPLETE", "=")

        return evaluation

    def judge_sync(
        self,
        input_data: str,
    ) -> CodeJudgeOutput:
        import asyncio

        return asyncio.run(self.judge(input_data))

    def _format_file_structure(self, file_structure: dict) -> str:
        lines = []
        for path, description in file_structure.items():
            lines.append(f"  {path}:")
            lines.append(f"    {description}")
        return "\n".join(lines)

    def _format_roadmap(self, roadmap: dict) -> str:
        lines = []
        for phase, details in roadmap.items():
            lines.append(f"\n{phase}:")
            lines.append(f"  {details}")
        return "\n".join(lines)


def create_code_judge_agent(
    model: str = "gpt-4o",
    working_dir: str = None,
    tools: Optional[list] = None,
    verbose: bool = False,
) -> CodeJudgeAgent:
    """
    Factory function to create a code judge agent.
    """
    return CodeJudgeAgent(
        model=model, working_dir=working_dir, tools=tools, verbose=verbose
    )
