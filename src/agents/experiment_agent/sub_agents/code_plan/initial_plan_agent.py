"""
Initial Plan Agent - Creates first code implementation plan.

This agent handles Scenario 1: First-time code planning based on
pre-analysis output (PreAnalysisOutput).
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


# Hand-written JSON output instruction for CodePlanOutput
CODE_PLAN_JSON_OUTPUT_INSTRUCTION = """
## Required JSON Output Format: CodePlanOutput

You MUST output a JSON object with this EXACT structure:

```json
{
  "plan_type": "initial",
  "file_structure": [
    {"path": "data/", "description": "Data loading and preprocessing"},
    {"path": "data/dataset.py", "description": "Dataset class for loading data"},
    {"path": "models/", "description": "Model architecture definitions"},
    {"path": "models/encoder.py", "description": "Encoder network"},
    {"path": "training/", "description": "Training pipeline"},
    {"path": "training/trainer.py", "description": "Main training loop"},
    {"path": "configs/", "description": "Configuration files"},
    {"path": "configs/default.yaml", "description": "Default hyperparameters"},
    {"path": "scripts/", "description": "Entry point scripts"},
    {"path": "scripts/train.py", "description": "Main training script"}
  ],
  "dataset_plan": "Load data from dataset_candidate/, implement Dataset class with __getitem__ and __len__, apply normalization transforms.",
  "model_plan": "Implement Encoder with Conv2d layers, Decoder with ConvTranspose2d, connect via latent space.",
  "training_plan": "Use Adam optimizer, MSE loss, train for 100 epochs with early stopping, save best checkpoint.",
  "implementation_checklist": [
    {
      "step_id": 1,
      "title": "Create Project Structure",
      "description": "Create directories and __init__.py files",
      "files_to_create": ["data/__init__.py", "models/__init__.py", "training/__init__.py"],
      "files_to_modify": null,
      "acceptance_criteria": ["All directories exist"]
    },
    {
      "step_id": 2,
      "title": "Implement Dataset Class",
      "description": "Create Dataset with __getitem__ and __len__",
      "files_to_create": ["data/dataset.py"],
      "files_to_modify": null,
      "acceptance_criteria": ["Dataset class works"]
    },
    {
      "step_id": 3,
      "title": "Implement Encoder",
      "description": "Create Encoder network with Conv2d layers",
      "files_to_create": ["models/encoder.py"],
      "files_to_modify": null,
      "acceptance_criteria": ["Encoder forward pass works"]
    },
    {
      "step_id": 4,
      "title": "Implement Decoder",
      "description": "Create Decoder network with ConvTranspose2d",
      "files_to_create": ["models/decoder.py"],
      "files_to_modify": null,
      "acceptance_criteria": ["Decoder forward pass works"]
    }
  ],
  "implementation_notes": "Use PyTorch 2.0+, ensure reproducibility with fixed seeds, use absolute imports.",
  "experiment_plan": {
    "baseline_method": "Standard CNN autoencoder",
    "datasets": ["dataset_candidate/mnist", "dataset_candidate/cifar10"],
    "hyperparameter_space": "lr: [0.001, 0.0001], batch_size: [32, 64, 128], hidden_dim: [64, 128, 256]",
    "experiment_matrix": [
      {"exp_id": "E1", "method": "baseline", "dataset": "mnist", "hyperparameters": "lr=0.001, batch_size=32", "seeds": [42, 123, 456]},
      {"exp_id": "E2", "method": "proposed", "dataset": "mnist", "hyperparameters": "lr=0.001, batch_size=32", "seeds": [42, 123, 456]}
    ],
    "primary_metrics": ["reconstruction_loss", "accuracy"]
  }
}
```

### Field Descriptions:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `plan_type` | string | YES | "initial", "error_feedback", or "analysis_feedback" |
| `file_structure` | array | YES | List of {path, description} objects |
| `dataset_plan` | string | YES | Data loading strategy |
| `model_plan` | string | YES | Model architecture plan |
| `training_plan` | string | YES | Training pipeline plan |
| `implementation_checklist` | array | YES | List of ChecklistItem objects |
| `implementation_notes` | string | YES | Important implementation notes |
| `experiment_plan` | object | YES | ExperimentPlan object |

### ChecklistItem Object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step_id` | integer | YES | Unique step ID (1, 2, 3...) |
| `title` | string | YES | Brief step title |
| `description` | string | YES | What to implement |
| `files_to_create` | array or null | NO | Files to create |
| `files_to_modify` | array or null | NO | Files to modify |
| `acceptance_criteria` | array or null | NO | Verification criteria |

### ExperimentPlan Object:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `baseline_method` | string | YES | Baseline method name |
| `datasets` | array or null | NO | Dataset paths |
| `hyperparameter_space` | string | YES | Hyperparameter search space |
| `experiment_matrix` | array or null | NO | List of ExperimentItem objects |
| `primary_metrics` | array or null | NO | Evaluation metrics |

⚠️ **CRITICAL**: Output ONLY valid JSON, no markdown explanations!
"""


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
- Checklist: Step 1 = "Create Project Structure", then logical steps as needed

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

{CODE_PLAN_JSON_OUTPUT_INSTRUCTION}

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
