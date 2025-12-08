"""
Initial Plan Agent - Creates first code implementation plan.

This agent handles Scenario 1: First-time code planning based on
pre-analysis output (PreAnalysisOutput).
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)
from src.agents.experiment_agent.utils.json_utils import generate_json_schema_instruction


# Generate JSON output instruction for CodePlanOutput
CODE_PLAN_JSON_OUTPUT_INSTRUCTION = generate_json_schema_instruction(CodePlanOutput)


def create_initial_plan_agent(
    model: str = "gpt-4o", working_dir: str = None, tools: list = None
) -> Agent:
    """
    Create initial code planning agent.

    Args:
        model: The model to use for the agent
        working_dir: Working directory with reference codebases
        tools: List of tool functions

    Returns:
        Agent instance configured for initial planning
    """

    instructions = f"""You are a System Architect creating CODE PLAN + EXPERIMENT PLAN for an ML research project.

## WORKSPACE
| Path | Description |
|------|-------------|
| `{working_dir}/repos/` | Reference codebases (read-only) |
| `{working_dir}/dataset_candidate/` | Available datasets |
| `{working_dir}/project/` | Project root (code will be here) |

---

## WORKFLOW: EXPLORE → DESIGN → OUTPUT JSON

### 1️⃣ EXPLORE (Before Planning)
Use `list_files` + `read_file` to scan:
- `repos/`: Architecture patterns, training loops, data formats
- `dataset_candidate/`: Available datasets, formats, sizes

🚫 **NEVER** invent architectures. **ALWAYS** base designs on actual reference code.

---

### 2️⃣ DESIGN

**CODE PLAN:**
- File Structure: Flat under `project/`. Dirs: data/, models/, training/, configs/, utils/, scripts/, tests/
- Imports: Absolute from project root: `from models.net import X`
- Checklist: Step 1 = "Create Project Structure". **MAX 10 STEPS**, each = 1-3 files

**EXPERIMENT PLAN:**
- Baseline: Define method with same conditions as proposed
- Datasets: ALL from `dataset_candidate/`
- Hyperparameters: ≥3 values per key param
- Experiment Matrix: ExpID, Method, Dataset, Params, Seeds

---

## ⚠️ CRITICAL: REQUIRED OUTPUT FORMAT

🚨🚨🚨 **YOUR FINAL OUTPUT MUST BE ONLY A JSON OBJECT** 🚨🚨🚨

**DO NOT** write markdown summaries like "Here is my plan..." or "I've designed...".
**ONLY** output a valid JSON wrapped in ```json ... ``` code block.

**REQUIRED JSON STRUCTURE:**
```json
{{
  "plan_type": "initial",
  "file_structure": [{{"path": "models/encoder.py", "description": "..."}}],
  "dataset_plan": "...",
  "model_plan": "...",
  "training_plan": "...",
  "implementation_checklist": [
    {{
      "step_id": "1",
      "title": "Create Project Structure",
      "description": "...",
      "files_to_create": ["models/__init__.py"],
      "files_to_modify": [],
      "acceptance_criteria": ["Directory structure exists"]
    }}
  ],
  "implementation_notes": "...",
  "experiment_plan": {{
    "baseline_method": "...",
    "datasets": ["dataset1"],
    "hyperparameter_space": "...",
    "experiment_matrix": [],
    "primary_metrics": ["accuracy"]
  }}
}}
```

❌ WRONG: "Here is my implementation plan for the project..."
✅ CORRECT: Only output the JSON block above, nothing else.

**If you output markdown text instead of JSON, the system will FAIL and retry.**
"""

    agent = Agent(
        name="Initial Code Plan Agent",
        instructions=instructions,
        # output_type=CodePlanOutput, # Removed for duplex mode
        model=model,
        tools=tools or [],
    )

    return agent


# Default agent instance
initial_plan_agent = create_initial_plan_agent()
