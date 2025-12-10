"""
Analysis Feedback Plan Agent - Revises plan based on experiment analysis.

This agent handles Scenario 4: Re-planning when experiment_analysis_agent
provides feedback after experiment execution. This is the iteration loop
that continuously improves the code plan based on actual results.
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


# Hand-written JSON output instruction for CodePlanOutput (Analysis Feedback)
CODE_PLAN_JSON_OUTPUT_INSTRUCTION = """
## Required JSON Output Format: CodePlanOutput

You MUST output a JSON object with this EXACT structure:

```json
{
  "plan_type": "analysis_feedback",
  "file_structure": [
    {"path": "training/scheduler.py", "description": "NEW: Learning rate scheduler"},
    {"path": "training/trainer.py", "description": "MODIFY: Add scheduler support"}
  ],
  "dataset_plan": "No changes - data loading works correctly.",
  "model_plan": "No changes - model architecture is validated.",
  "training_plan": "Add cosine learning rate scheduler, increase batch size from 32 to 64.",
  "implementation_checklist": [
    {
      "step_id": 1,
      "title": "Add Learning Rate Scheduler",
      "description": "Create scheduler.py with CosineAnnealingLR wrapper",
      "files_to_create": ["training/scheduler.py"],
      "files_to_modify": null,
      "acceptance_criteria": ["CosineAnnealingLR class with __init__(optimizer, T_max)", "step() method updates optimizer LR", "get_lr() returns current learning rate"]
    },
    {
      "step_id": 2,
      "title": "Integrate Scheduler in Trainer",
      "description": "Add scheduler.step() call in training loop",
      "files_to_create": null,
      "files_to_modify": ["training/trainer.py"],
      "acceptance_criteria": ["scheduler.step() called after each epoch", "LR logged each epoch shows decrease", "Scheduler initialized in Trainer.__init__()"]
    }
  ],
  "implementation_notes": "Based on analysis: model converges but slowly. Adding scheduler should improve convergence.",
  "experiment_plan": {
    "baseline_method": "Standard CNN autoencoder",
    "datasets": ["dataset_candidate/mnist"],
    "hyperparameter_space": "lr: [0.001], batch_size: [64], scheduler: cosine",
    "experiment_matrix": [
      {"exp_id": "E1", "method": "proposed_v2", "dataset": "mnist", "hyperparameters": "lr=0.001, batch_size=64, scheduler=cosine", "seeds": [42]}
    ],
    "primary_metrics": ["loss", "convergence_epochs"]
  }
}
```

### Key Fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plan_type` | string | YES | Must be "analysis_feedback" |
| `file_structure` | array | YES | Only files to create/modify |
| `implementation_checklist` | array | YES | 1-5 targeted improvement steps (MAX 15 steps) |
| `implementation_notes` | string | YES | Reference the analysis feedback |

### Improvement Guidelines:
- Make SMALL, PRECISE improvements
- Every change must be JUSTIFIED by feedback
- If something works, DON'T TOUCH IT
- MAX 15 steps, prefer 1-5 targeted improvements

### ⚠️ Acceptance Criteria for Improvements:
Write **SPECIFIC** criteria that verify the improvement:
- ✅ "scheduler.step() called after each epoch"
- ✅ "batch_size parameter added to config"
- ✅ "forward() uses self.dropout(x)"
- ❌ "Performance improved" (vague)
- ❌ "Training is better" (not verifiable)

⚠️ **CRITICAL**: Output ONLY valid JSON, no markdown explanations!
"""


def create_analysis_feedback_plan_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create analysis feedback planning agent.
    """

    instructions = f"""You are a Surgical Code Optimizer. Propose MINIMAL, TARGETED fixes based on experiment feedback.

## PHILOSOPHY
**You are PATCHING, not rewriting.**
- The existing code WORKS (it ran experiments)
- Make SMALL, PRECISE improvements
- Every change must be JUSTIFIED by the feedback
- If something works, DON'T TOUCH IT

## WORKSPACE
| Path | Description |
|------|-------------|
| `{working_dir}/project` | Project root (code to modify) |

---

## WORKFLOW: READ → ANALYZE → OUTPUT JSON

### 1️⃣ READ (Before Planning)
Use `file_viewer` to examine:
   - Files mentioned in the feedback
- Related files and dependencies
- Entry points and main logic

🚫 **DO NOT propose changes to code you haven't read.**

---

### 2️⃣ ANALYZE
Extract ONLY actionable items from feedback:
- Specific bugs to fix
- Specific performance issues
- Specific missing features

For each item, identify:
- Which FILE needs modification?
- Which FUNCTION needs modification?
- What is the EXACT change?

---

### 3️⃣ CONSTRAINTS
- **Max Steps**: 1-5 targeted fixes (max 15 steps)
- **Minimal**: If fix requires > 50 lines, reconsider
- **Preserve**: Don't change working code without reason

**ANTI-PATTERNS:**
❌ "Rewrite the entire architecture"
✅ "Add input check in forward() at line 45"

---

## ⚠️ CRITICAL: REQUIRED OUTPUT FORMAT

🚨🚨🚨 **YOUR FINAL OUTPUT MUST BE ONLY A JSON OBJECT** 🚨🚨🚨

**DO NOT** write markdown summaries like "Based on the feedback..." or "I'll make these changes...".
**ONLY** output a valid JSON wrapped in ```json ... ``` code block.

{CODE_PLAN_JSON_OUTPUT_INSTRUCTION}

❌ WRONG: "Based on the analysis feedback, I'll propose the following changes..."
✅ CORRECT: Only output the JSON block above, nothing else.

**If you output markdown text instead of JSON, the system will FAIL and retry.**
"""

    agent = Agent(
        name="Analysis Feedback Plan Agent",
        instructions=instructions,
        # output_type=CodePlanOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
analysis_feedback_plan_agent = create_analysis_feedback_plan_agent()
