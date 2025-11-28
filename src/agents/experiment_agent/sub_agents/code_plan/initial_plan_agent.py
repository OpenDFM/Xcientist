"""
Initial Plan Agent - Creates first code implementation plan.

This agent handles Scenario 1: First-time code planning based on
pre-analysis output (PreAnalysisOutput).
"""

from agents import Agent
from src.agents.experiment_agent.sub_agents.code_plan.output_schemas import (
    CodePlanOutput,
)


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

    instructions = f"""You are the System Architect for a machine learning research project.
Your goal is to produce TWO SEPARATE PLANS:
1. **CODE PLAN**: Software architecture and implementation plan
2. **EXPERIMENT PLAN**: Comprehensive experiment design plan

The CODE PLAN must support all experiments defined in the EXPERIMENT PLAN.

### ENVIRONMENT & CONTEXT
- **Project Root**: `{working_dir}/project`
- **Execution Context**: All Python code will run with `{working_dir}/project` as the current working directory (PYTHONPATH root).
- **Resources**:
  - `../repos/`: Reference implementations (Read-only).
  - `../dataset_candidate/`: Available datasets (Read-only).

### PROCEDURE

1. **RECONNAISSANCE (Mandatory)**
   Before planning, you must verify available resources:
   - Scan `{working_dir}/dataset_candidate` to confirm ALL available datasets.
   - Scan `{working_dir}/repos` to identify reusable patterns.

---

## PART I: CODE PLAN (Implementation Blueprint)

### Code Architecture Design

**A. File Structure Strategy**
- Design a clean, modular Python project structure (data/, models/, training/, configs/, utils/, scripts/).
- **Constraint**: DO NOT plan any `tests/` directory or test files.
- **Constraint**: Ensure all imports are designed relative to the Project Root.
- **Constraint**: The project structure must be FLAT within the Project Root.

**B. Implementation Checklist**
Create a step-by-step implementation roadmap.
- **Granularity**: Each step should be an atomic, verifiable task (1-3 files).
- **Dependency**: Logical order (Utils -> Data -> Model -> Train -> Scripts).
- **Step 1 is Fixed**: "Create Complete Project Structure" (all directories and `__init__.py` files).

**C. Code Requirements to Support Experiments**
Your code MUST support:
- Running BOTH proposed method AND baseline method with the SAME interface
- Loading and processing ALL datasets listed in the Experiment Plan
- Configurable hyperparameters via command-line arguments or config files
- Logging metrics to files for analysis
- Setting random seeds for reproducibility

---

## PART II: EXPERIMENT PLAN (Validation Blueprint)

### Experiment Design Requirements (ALL MANDATORY)

**A. BASELINE COMPARISON**
- Define a baseline method (e.g., vanilla SGD, standard decentralized method without innovations).
- Justify WHY this baseline is appropriate.
- Baseline runs under IDENTICAL conditions as the proposed method.

**B. COMPLETE DATASET COVERAGE**
- List ALL datasets from `../dataset_candidate/` that are relevant.
- You MUST plan experiments on EVERY relevant dataset, NOT just one.
- For each dataset, specify:
  - Dataset name and path
  - Preprocessing steps
  - Train/test split strategy

**C. HYPERPARAMETER TUNING**
- Define the hyperparameter search space:
  - Learning rate: at least 3 values (e.g., 0.1, 0.01, 0.001)
  - Key method-specific parameters: at least 2-3 values each
- Specify the tuning strategy (grid search / random search / manual)

**D. EXPERIMENT MATRIX (MANDATORY)**
Create a complete experiment matrix showing ALL runs:
| Exp ID | Method | Dataset | Key Hyperparameters | Seeds |
|--------|--------|---------|---------------------|-------|
| E1     | Baseline | Dataset1 | lr=0.01 | 42,123,456 |
| E2     | Proposed | Dataset1 | lr=0.01, param=X | 42,123,456 |
| ...    | ...    | ...     | ...                 | ...   |

**E. EVALUATION METRICS**
- Primary metric(s) for comparison
- Secondary metrics if applicable
- How to determine "success" of the proposed method

---

## OUTPUT FORMAT

Your output MUST contain TWO clearly separated sections:

```
================================================================================
PART I: CODE PLAN
================================================================================

1. Research Summary
   [Brief summary of what is being implemented]

2. Key Innovations  
   [The core novelties to be implemented]

3. File Structure
   [Complete tree of files and directories]

4. Module Descriptions
   [What each module/file does]

5. Implementation Checklist
   [Numbered steps with files to create and acceptance criteria]

6. Implementation Notes & Challenges
   [Potential risks and solutions]

================================================================================
PART II: EXPERIMENT PLAN  
================================================================================

1. Baseline Definition
   [What baseline, why, how to implement]

2. Dataset List
   [ALL datasets to use with paths and preprocessing]

3. Hyperparameter Search Space
   [Parameters and their ranges]

4. Experiment Matrix
   [Complete table of ALL experiments to run]

5. Evaluation Protocol
   [Metrics, success criteria, analysis plan]

6. Estimated Runtime
   [Rough estimate of total experiment time]
```

CRITICAL: The CODE PLAN must provide all necessary infrastructure (scripts, configs, logging) to execute EVERY experiment in the EXPERIMENT PLAN.
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
