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

## WORKFLOW: 1️⃣ EXPLORE → 2️⃣ DESIGN → 3️⃣ OUTPUT

---

## 1️⃣ EXPLORE (Mandatory Before Planning)

| Resource | Action | Extract |
|----------|--------|---------|
| `{working_dir}/repos/` | `list_files` + `read_file` | Architecture patterns, training loops, data formats |
| `{working_dir}/dataset_candidate/` | Scan contents | Available datasets, formats, sizes |

🚫 **NEVER** invent architectures. **ALWAYS** base designs on actual reference code you read.

---

## 2️⃣ DESIGN

### PART I: CODE PLAN

| Section | Requirements |
|---------|--------------|
| File Structure | Flat under `{working_dir}/project/`. Dirs: data/, models/, training/, configs/, utils/, scripts/, tests/ |
| Tests Directory | ALL test files and test-related folders MUST be placed in `tests/` directory |
| Imports | Absolute from project root: `from models.net import X` |
| Checklist | Step 1 = "Create Project Structure". Each step = 1-3 files, atomic, verifiable. **MAX 15 STEPS** |
| Baseline Support | Same interface for proposed AND baseline methods |

⚠️ **CHECKLIST CONSTRAINTS:**
- **Maximum 10 steps** - combine related tasks if needed
- Each step should be **actionable**
- Prioritize core functionality over edge cases
- Group related files into single steps (e.g., "Create model components" can include 2-3 model files)

### PART II: EXPERIMENT PLAN

| Section | Requirements |
|---------|--------------|
| Baseline | Define method, justify why, same conditions as proposed |
| Datasets | ALL relevant datasets from `../dataset_candidate/` |
| Hyperparameters | ≥3 values per key param (lr, method-specific) |
| Experiment Matrix | Complete table: ExpID, Method, Dataset, Params, Seeds |
| Metrics | Primary + secondary metrics, success criteria |

---

## 3️⃣ OUTPUT (JSON FORMAT - CRITICAL)

After completing your analysis and design, you MUST output your final plan as a JSON object.

{CODE_PLAN_JSON_OUTPUT_INSTRUCTION}

**Important JSON Field Mappings:**
- `plan_type`: Set to "initial"
- `file_structure`: List of FileStructureItem objects with path, description
- `dataset_plan`: Dataset preparation and loading plan
- `model_plan`: Model implementation plan (Module Descriptions)
- `training_plan`: Training pipeline plan
- `implementation_checklist`: List of ChecklistItem objects with step_id, title, description, files_to_create, files_to_modify, acceptance_criteria
- `implementation_notes`: Reference Code Analysis + important notes
- `experiment_plan`: ExperimentPlan object with baseline_method, datasets, hyperparameter_space, experiment_matrix, primary_metrics

---

## KEY PRINCIPLE

The CODE PLAN must provide ALL infrastructure to execute EVERY experiment in the EXPERIMENT PLAN.
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
