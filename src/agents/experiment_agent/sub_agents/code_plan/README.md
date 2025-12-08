

# Code Plan Agent System

A comprehensive multi-agent system for generating detailed code implementation plans in YAML format, supporting multiple scenarios including initial planning and various feedback-based re-planning.

## Architecture

```
Code Plan Agent System
├── Router Agent                  → Detects input scenario
├── Scenario-Specific Planners
│   ├── Initial Plan Agent        → First-time planning
│   ├── Error Feedback Plan       → Re-plan after runtime errors
│   └── Analysis Feedback Plan    → Re-plan after analysis
└── Output Formatter              → YAML-compatible formatting
```

## Scenarios

### 1. Initial Planning (`initial`)
**Trigger**: First-time code planning  
**Input**: `UnifiedAnalysisOutput` from pre_analysis_agent  
**Purpose**: Create comprehensive initial implementation plan

### 2. Error Feedback Planning (`error_feedback`)
**Trigger**: Code execution failed at runtime  
**Input**:
- `UnifiedAnalysisOutput` from pre_analysis_agent
- Processed error information from experiment_execute_agent (via experiment_master_agent)

**Purpose**: Revise plan to fix runtime errors and add robustness

### 3. Analysis Feedback Planning (`analysis_feedback`)
**Trigger**: Experimental results were unsatisfactory  
**Input**:
- `UnifiedAnalysisOutput` from pre_analysis_agent
- Processed analysis conclusions from experiment_analysis_agent (via experiment_master_agent)

**Purpose**: Improve plan to achieve better experimental results

## Components

### 1. Router Agent (`code_plan_agent.py`)
Determines which of the four scenarios applies to the input.

**Detection Strategy**:
- Analyzes input structure and content
- Identifies presence of feedback/error/analysis information
- Routes to appropriate scenario-specific planner

### 2. Initial Plan Agent (`initial_plan_agent.py`)
Creates the first implementation plan from research analysis.

**Workflow**:
1. Review reference codebases
2. Generate file structure
3. Create dataset/model/training/testing plans
4. Build implementation roadmap

**Output**: `IntermediatePlanOutput`

### 3. Error Feedback Plan Agent (`error_feedback_plan_agent.py`)
Revises plan based on runtime errors.

**Focus**:
- Fix runtime errors
- Add validation and error handling
- Improve robustness
- Specify defensive programming

**Output**: `IntermediatePlanOutput`

### 4. Analysis Feedback Plan Agent (`analysis_feedback_plan_agent.py`)
Revises plan based on experimental analysis.

**Focus**:
- Address performance issues
- Incorporate experimental insights
- Improve architectural choices
- Optimize hyperparameters

**Output**: `IntermediatePlanOutput`

### 5. Output Formatter (`output_formatter.py`)
Converts intermediate output to YAML-compatible format.

**Tasks**:
- Parse file structure descriptions
- Structure implementation steps
- Organize all components
- Ensure YAML compatibility

**Output**: `CodePlanOutput`

## Output Schema

### CodePlanOutput (YAML-compatible)

```yaml
plan_type: "initial" | "error_feedback" | "analysis_feedback"
timestamp: "2025-11-05T..."

research_summary: "..."
key_innovations: "..."

file_structure:
  - path: "project/"
    type: "directory"
    description: "Project root"
  - path: "project/data/dataset.py"
    type: "file"
    description: "Dataset implementation"
  # ... more items

dataset_plan: "Detailed dataset plan..."
model_plan: "Detailed model plan..."
training_plan: "Detailed training plan..."
testing_plan: "Detailed testing plan..."

implementation_roadmap:
  - step_number: 1
    title: "Implement dataset"
    description: "..."
    files_involved: ["project/data/dataset.py"]
    dependencies: []
  # ... more steps

implementation_notes: "Important notes..."
potential_challenges: "Challenges and mitigations..."
addressed_issues: "How feedback was addressed..."
```

## Usage

### Basic Usage

```python
import asyncio
from src.agents.experiment_agent.sub_agents.code_plan_agent import (
    create_code_plan_agent
)

async def main():
    # Create the agent system
    agent = create_code_plan_agent(
        model="gpt-4o",
        working_dir="/workspace",
        tools={
            "initial": [gen_code_tree_structure, read_file],
            "error_feedback": [gen_code_tree_structure, read_file, read_logs],
            "analysis_feedback": [gen_code_tree_structure, read_file, read_logs, read_results]
        }
    )
    
    # Scenario 1: Initial planning
    research_analysis = """
    UnifiedAnalysisOutput:
    system_architecture: ...
    algorithms: ...
    ...
    """
    
    plan = await agent.plan(research_analysis)
    
    # Access the plan
    print(f"Plan Type: {plan.plan_type}")
    print(f"File Structure: {plan.file_structure}")
    print(f"Implementation Roadmap: {plan.implementation_roadmap}")
    
    # Convert to YAML
    import yaml
    yaml_output = yaml.dump(plan.dict(), default_flow_style=False)
    print(yaml_output)

asyncio.run(main())
```

### Scenario-Specific Usage

#### Scenario 1: Initial Planning
```python
input_data = """
UnifiedAnalysisOutput:
  input_type: paper
  system_architecture: ...
  algorithms: ...
  implementation_guidance: ...
"""

plan = await agent.plan(input_data)
```

#### Scenario 2: Error Feedback
```python
input_data = """
UnifiedAnalysisOutput:
  ...

Error Feedback (from experiment_master_agent):
  error_type: RuntimeError
  error_message: ...
  stack_trace: ...
  failure_context: ...
  recommended_fixes: ...
"""

plan = await agent.plan(input_data)
```

#### Scenario 3: Analysis Feedback
```python
input_data = """
UnifiedAnalysisOutput:
  ...

Analysis Feedback (from experiment_master_agent):
  performance_metrics:
    accuracy: 0.75
    loss: 0.45
  identified_issues:
    - Issue 1: ...
  suggested_improvements:
    - Improvement 1: ...
"""

plan = await agent.plan(input_data)
```

### Synchronous Usage

```python
from src.agents.experiment_agent.sub_agents.code_plan_agent import (
    create_code_plan_agent
)

# Create the agent system
agent = create_code_plan_agent(model="gpt-4o", working_dir="/workspace")

# Plan synchronously
plan = agent.plan_sync(input_data)
```

### Converting to YAML File

```python
import yaml

# Generate plan
plan = await agent.plan(input_data)

# Convert to YAML
yaml_content = yaml.dump(plan.dict(), default_flow_style=False, sort_keys=False)

# Save to file
with open("code_plan.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)
```

## Tool Requirements

Tools should be implemented in `src/agents/experiment_agent/tools/`.

### Common Tools (all scenarios):
- `gen_code_tree_structure(directory: str) -> str`: Get directory structure
- `read_file(file_path: str) -> str`: Read file content
- `terminal_page_down()`: Scroll terminal output
- `terminal_page_up()`: Scroll terminal output
- `terminal_page_to(page: int)`: Jump to page

### Additional Tools for Error Feedback:
- `read_logs(log_file: str) -> str`: Read execution logs

### Additional Tools for Analysis Feedback:
- `read_logs(log_file: str) -> str`: Read training logs
- `read_results(results_file: str) -> dict`: Read experimental results

## Integration Example

### With Pre-Analysis Agent

```python
from src.agents.experiment_agent.sub_agents.pre_analysis import (
    create_pre_analysis_agent
)
from src.agents.experiment_agent.sub_agents.code_plan_agent import (
    create_code_plan_agent
)

# Step 1: Analyze research input
pre_analysis = create_pre_analysis_agent()
analysis = await pre_analysis.analyze(paper_content)

# Step 2: Generate code plan
code_planner = create_code_plan_agent(working_dir="/workspace")
plan = await code_planner.plan(str(analysis.dict()))

# Step 3: Save plan as YAML
import yaml
with open("implementation_plan.yaml", "w", encoding="utf-8") as f:
    yaml.dump(plan.dict(), f, default_flow_style=False)
```

### Complete Workflow (with feedback loop)

```python
# Initial planning
analysis = await pre_analysis_agent.analyze(input_content)
plan = await code_plan_agent.plan(str(analysis.dict()))

# ... implementation happens ...

# If execution errors occur
error_feedback = "..."  # from experiment_master_agent
revised_plan = await code_plan_agent.plan(
    f"{analysis.dict()}\n\nError Feedback:\n{error_feedback}"
)

# If analysis suggests improvements
analysis_feedback = "..."  # from experiment_master_agent
improved_plan = await code_plan_agent.plan(
    f"{analysis.dict()}\n\nAnalysis Feedback:\n{analysis_feedback}"
)
```

## Design Principles

### 1. Scenario-Based Planning
Each scenario has specialized planning logic:
- **Initial**: Comprehensive, exploratory
- **Error Feedback**: Focused on robustness
- **Analysis Feedback**: Focused on performance

### 2. Consistent Output Format
All scenarios produce the same CodePlanOutput structure, ensuring downstream compatibility.

### 3. YAML-Compatible Structure
Output is designed to be easily serialized to YAML for human readability and tool consumption.

### 4. Iterative Refinement
Supports multiple re-planning iterations with different feedback types.

## Environment Setup

Ensure PYTHONPATH includes the ResearchAgent root:

```bash
export PYTHONPATH="/hpc_stor03/sjtu_home/hanqi.li/agent_workspace/ResearchAgent:$PYTHONPATH"
```

## File Structure

```
code_plan_agent/
├── __init__.py                          # Module exports
├── output_schemas.py                    # Pydantic models
├── code_plan_agent.py                  # Main orchestrator
├── initial_plan_agent.py               # Scenario 1
├── error_feedback_plan_agent.py        # Scenario 2
├── analysis_feedback_plan_agent.py     # Scenario 3
├── output_formatter.py                 # YAML formatting
└── README.md                           # Documentation
```

## Notes

- All agents use OpenAI Agents SDK
- Async/await is the primary execution model
- YAML format ensures human-readable output
- Plans are comprehensive and directly implementable
- Supports iterative refinement through multiple scenarios

