ALGORITHM_ALIGNMENT_PROMPT = """
You are a research alignment editor. A single idea (title + abstract) is implemented by several algorithm specs, but some may drift away from the idea. Rewrite the algorithms so every entry clearly supports the given idea, removing anything that cannot be justified.

== Idea Title == 
{idea_title}

== Idea Abstract == 
{idea_abstract}

== Idea Method ==
{idea_method}

== Exact Components ==
{components}

== Component Explanations ==
{component_explanations}

== Candidate Algorithms == 
{algorithms}

== Requirements ==
- The top-level JSON value MUST be an object, never a raw array.
- Even if only one algorithm remains, still return `{{"algorithms": [ ... ]}}`.
- Do NOT return `[{{...}}]` at the top level.
- Maintain the schema exactly: each algorithm has "name", "input", "output", "pipeline".
- Edit minimally. Preserve concrete operations and structure when already aligned.
- Do NOT rewrite detailed steps into higher-level summaries, overviews, or motivation prose.
- `input`, `output`, and `pipeline` must remain arrays of strings, not arrays of objects.
- `pipeline[*]` must keep the `Step k:` form.
- Whenever a pipeline step or input references an existing module, use the exact component names copied from `components`.
- The final algorithm must read as a standalone specification for THIS idea only. Assume the downstream experiment agent can see only this idea.
- Remove any mention of other ideas, alternative candidates, sibling modes, source modes, fusion, host idea, parent idea, rejected components, or comparisons against other ideas.
- You may repair, merge, or drop entries, but only if needed to align them with the idea title, abstract, and method.
- If you keep multiple algorithms, ensure each pipeline explicitly references the mechanisms and exact components in the method.
- No commentary outside the JSON.

Return JSON:
{{
  "algorithms": [
    {{
      "name": "...",
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
"""
