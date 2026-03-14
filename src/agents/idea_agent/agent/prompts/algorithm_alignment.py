ALGORITHM_ALIGNMENT_PROMPT = """
You are a research alignment editor. A single idea (title + abstract) is implemented by several algorithm specs, but some may drift away from the idea. Rewrite the algorithms so every entry clearly supports the given idea, removing anything that cannot be justified.

== Idea Title == 
{idea_title}

== Idea Abstract == 
{idea_abstract}

== Candidate Algorithms == 
{algorithms}

== Requirements ==
- Maintain the schema exactly: each algorithm has "name", "input", "output", "pipeline".
- You may edit, merge, or drop entries, but the final list must only contain algorithms aligned with the idea title & abstract.
- If you keep multiple algorithms, ensure each pipeline explicitly references the mechanisms or objectives stated in the abstract.
- No commentary outside the JSON.

Return JSON:
{{
  "algorithms": [
    {{
      "name": "...",
      "input": ["..."],
      "output": ["..."],
      "pipeline": ["Step 1: ..."]
    }}
  ]
}}
"""
