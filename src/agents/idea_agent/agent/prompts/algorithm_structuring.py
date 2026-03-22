ALGORITHM_STRUCTURING_PROMPT = """
You are an expert algorithm architect. Your task is to translate an abstract research idea into a rigorous, executable algorithm specification.

== Topic == 
{topic}

== Idea Title ==
{idea_title}

== Idea Abstract ==
{idea_abstract}

== Compact Idea JSON ==
{idea}

Return ONLY a valid JSON object matching this schema exactly:
{{
  "algorithms": [
    {{
      "name": "Concise algorithm name (<8 words)",
      "input": [
        "concrete input description"
      ],
      "output": [
        "concrete output description"
      ],
      "pipeline": [
        "Step 1: concrete execution action using exact component names"
      ]
    }}
  ]
}}

== Rules (Strict) ==
- The top-level JSON value MUST be an object, never a raw array.
- Even if there is only one algorithm, still return `{{"algorithms": [ ... ]}}`.
- Do NOT return `[{{...}}]` at the top level.
- `input`, `output`, and `pipeline` must each be arrays of strings, not arrays of objects.
- Use ONLY exact component names copied from `components` whenever an input line or pipeline step touches an existing module.
- Every `pipeline` item must start with `Step k:` and describe one concrete execution step, not an overview, heading, or prose summary.
- Each pipeline step should explicitly name the exact components it uses and the concrete state, signal, or artifact it produces.
- Write the algorithm as a standalone specification for THIS idea only. Assume the downstream experiment agent can see only this idea.
- Do NOT mention other ideas, alternative candidates, sibling modes, source modes, fusion, host idea, parent idea, rejected components, or comparisons against other ideas anywhere in `name`, `input`, `output`, or `pipeline`.
- NO MAGIC WORDS: do not write vague summaries like 'integrate the system' or 'optimize performance'. Write the actual mechanism.
- Keep inputs/outputs concrete (e.g., `slot_records` instead of `data`).
- If multiple distinct sub-algorithms are needed to realize the idea, include each as its own entry in the `algorithms` array.
- Do not add any markdown formatting (like ```json) or commentary outside the JSON object.
"""
