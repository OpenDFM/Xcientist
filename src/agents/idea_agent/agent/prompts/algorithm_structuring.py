ALGORITHM_STRUCTURING_PROMPT = """
You are an expert algorithm architect. Your task is to translate an abstract research idea into a rigorous, executable algorithm specification.

== Topic == 
{topic}

== Idea Title ==
{idea_title}

== Idea Abstract ==
{idea_abstract}

== Idea JSON ==
{idea}

First, think step-by-step in the `architect_scratchpad` to deconstruct the idea into computable mathematical or logical components. Then, generate the concrete algorithm specs.

Return ONLY a valid JSON object matching this schema exactly:
{{
  "algorithms": [
    {{
      "name": "Concise algorithm name (<12 words)",
      "input": ["List of required concrete inputs, datasets, or states"],
      "output": ["List of expected outputs, decisions, or updated states"],
      "pipeline": [
        "Step 1: ...",
        "Step 2: ...",
        "Step 3: ..."
      ]
    }}
  ]
}}

== Rules (Strict) ==
- You MUST complete the `architect_scratchpad` FIRST before writing the `algorithms` array.
- The `pipeline` steps must be derived directly from your reasoning in the `black_box_elimination`.
- NO MAGIC WORDS: Pipeline steps must describe actual method execution. Instead of 'run MCTS', specify 'Select node via UCB1, expand by querying LLM policy, evaluate state, and backpropagate value'. Instead of 'use dual-system', specify the exact routing or gating mechanism.
- Keep inputs/outputs concrete (e.g., 'action trajectory dictionary' instead of 'data').
- If multiple distinct sub-algorithms are needed to realize the idea, include each as its own entry in the `algorithms` array.
- Do not add any markdown formatting (like ```json) or commentary outside the JSON object.
"""
