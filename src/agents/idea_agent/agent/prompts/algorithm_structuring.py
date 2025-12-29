ALGORITHM_STRUCTURING_PROMPT = """
You are an algorithm architect. Convert the idea below into executable algorithm specs.

Topic: {topic}
Idea JSON:
{idea}

Baseline Inputs:
{base_inputs}

Baseline Outputs:
{base_outputs}

Latest Analysis Snapshot:
{analysis}

Reference Hints (may be empty):
{references}

Return ONLY a JSON object:
{{
  "algorithms": [
    {{
      "name": "Concise algorithm name (<12 words)",
      "input": ["List of required inputs, datasets, sensors, or preconditions"],
      "output": ["List of expected outputs, measurements, or deliverables"],
      "pipeline": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ..."
      ]
    }}
  ]
}}

Rules:
- Pipeline steps must describe the actual method execution (no meta statements like 'run MCTS').
- Keep inputs/outputs concrete and derived from the idea details and references.
- If multiple sub-algorithms exist, include each as its own entry.
- Do not add commentary outside the JSON.
"""
