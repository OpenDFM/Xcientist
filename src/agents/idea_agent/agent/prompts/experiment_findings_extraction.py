EXPERIMENT_FINDINGS_EXTRACTION_PROMPT = """
You are a research-analysis agent.

== Task == 
Read the raw ablation / experiment JSON below and extract structured findings for LigAgent.

== Important assumptions == 
- Treat every key under `components` as a REAL component name.
- Do not invent new component names.
- Base all findings strictly on the provided raw JSON.
- The output must stay grounded in the raw JSON. Do not add action suggestions, theme taxonomies, or other speculative labels.
- Every component under `components` MUST appear exactly once in `component_findings`.
- The `component` field must exactly match the original key from `components`.
- If `summary.feasible`, `summary.confidence`, or `summary.key_findings` exist in the raw JSON, preserve them faithfully unless the raw JSON is clearly inconsistent.

== Raw ablation JSON ==
{raw_ablation}

Return STRICT JSON with this schema:
{{
  "summary": {{
    "hypothesis_status": "supported|challenged|falsified|mixed|inconclusive",
    "feasible": true,
    "overall_confidence": 0.0,
    "tldr": "short summary of the overall ablation outcome",
    "key_findings": ["..."]
  }},
  "component_findings": [
    {{
      "component": "string",
      "result": "positive|negative|inconclusive",
      "metric": "string",
      "value": "string",
      "confidence": 0.0,
      "analysis": "short grounded explanation of what this result means"
    }}
  ]
}}
"""
