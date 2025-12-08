"""
Error Feedback Plan Agent - Revises plan based on runtime errors.

This agent handles Scenario 3: Re-planning when experiment_execute_agent
encountered runtime errors, and experiment_master_agent determined
that re-planning is needed.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


# Hand-written JSON output instruction for CodePlanOutput (Error Feedback)
CODE_PLAN_JSON_OUTPUT_INSTRUCTION = """
## Required JSON Output Format: CodePlanOutput

You MUST output a JSON object with this EXACT structure:

```json
{
  "plan_type": "error_feedback",
  "file_structure": [
    {"path": "data/dataset.py", "description": "Dataset class - FIX: add None check"},
    {"path": "models/encoder.py", "description": "Encoder - FIX: correct dimension at line 45"}
  ],
  "dataset_plan": "Fix data loading: add validation for empty files, handle missing keys.",
  "model_plan": "Fix encoder: change dim=0 to dim=1 at line 45 to match expected tensor shape.",
  "training_plan": "No changes needed - training loop is correct.",
  "implementation_checklist": [
    {
      "step_id": 1,
      "title": "Fix Encoder Dimension Bug",
      "description": "Change dim=0 to dim=1 at line 45 in encoder.py",
      "files_to_create": null,
      "files_to_modify": ["models/encoder.py"],
      "acceptance_criteria": ["No dimension mismatch error", "Forward pass completes"]
    }
  ],
  "implementation_notes": "MINIMAL FIX: Only change the exact line causing the error. Do not refactor.",
  "experiment_plan": {
    "baseline_method": "Standard CNN autoencoder",
    "datasets": ["dataset_candidate/mnist"],
    "hyperparameter_space": "lr: [0.001], batch_size: [32]",
    "experiment_matrix": [
      {"exp_id": "E1", "method": "proposed", "dataset": "mnist", "hyperparameters": "lr=0.001", "seeds": [42]}
    ],
    "primary_metrics": ["loss"]
  }
}
```

### Key Fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plan_type` | string | YES | Must be "error_feedback" |
| `file_structure` | array | YES | Files to fix (minimal list) |
| `implementation_checklist` | array | YES | 1-3 targeted fix steps (MAX 15 steps) |
| `implementation_notes` | string | YES | Emphasize MINIMAL fix |

### Fix Guidelines:
- Target EXACT file and line number
- Change MINIMUM code necessary
- MAX 15 steps, prefer 1-3 targeted fixes
- If fix requires > 20 lines, reconsider

⚠️ **CRITICAL**: Output ONLY valid JSON, no markdown explanations!
"""


def create_error_feedback_plan_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create error feedback planning agent.
    """

    instructions = f"""You are a Debugging Specialist. The code crashed. Propose the MINIMUM fix to resolve the error.

## PHILOSOPHY
**You are DEBUGGING, not redesigning.**
- The code was close to working (it got far enough to crash)
- Find the EXACT bug and fix ONLY that
- The best fix changes the FEWEST lines possible

## WORKSPACE
| Path | Description |
|------|-------------|
| `{working_dir}/project` | Project root (code to fix) |

---

## WORKFLOW: DIAGNOSE → FIX → OUTPUT JSON

### 1️⃣ DIAGNOSE (Parse Error)
From the error feedback, extract:
- Error TYPE (TypeError, ValueError, KeyError, etc.)
- Error MESSAGE
- FILE and LINE NUMBER
- STACK TRACE

Use `read_file` to read:
- The file that threw the error
- Functions in the stack trace
- Related imports/dependencies

🚫 **DO NOT propose fixes until you have read the actual failing code.**

---

### 2️⃣ FIX (Minimum Viable Patch)
Priority order (try simpler fixes first):
1. Typo/Simple: Wrong variable name, missing import
2. Type: Wrong data type, shape mismatch
3. Logic: Incorrect condition, wrong order
4. Interface: Mismatched signature

**CONSTRAINTS:**
- Target EXACT file and line
- Change MINIMUM code necessary
- If fix requires > 20 lines, you're overcomplicating
- **Max Steps**: 1-3 targeted fixes (max 15 steps)

**ANTI-PATTERNS:**
❌ "Rewrite the data loading pipeline"
✅ "Fix line 45: change `dim=0` to `dim=1`"

---

## ⚠️ CRITICAL: REQUIRED OUTPUT FORMAT

🚨🚨🚨 **YOUR FINAL OUTPUT MUST BE ONLY A JSON OBJECT** 🚨🚨🚨

**DO NOT** write markdown summaries like "The error is caused by..." or "I'll fix this by...".
**ONLY** output a valid JSON wrapped in ```json ... ``` code block.

{CODE_PLAN_JSON_OUTPUT_INSTRUCTION}

❌ WRONG: "The error occurs because of a dimension mismatch. Here's how to fix it..."
✅ CORRECT: Only output the JSON block above, nothing else.

**If you output markdown text instead of JSON, the system will FAIL and retry.**
"""

    agent = Agent(
        name="Error Feedback Plan Agent",
        instructions=instructions,
        # output_type=CodePlanOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
error_feedback_plan_agent = create_error_feedback_plan_agent()
