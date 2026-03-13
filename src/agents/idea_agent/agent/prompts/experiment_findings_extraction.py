EXPERIMENT_FINDINGS_EXTRACTION_PROMPT = """
You are a research-analysis agent.

Task:
Read the raw ablation / experiment JSON below and extract structured findings for LigAgent.

Important assumptions:
- Treat every key under `components` as a REAL component name.
- Do not invent new component names.
- Base all findings strictly on the provided raw JSON.
- Distinguish hypothesis-level conclusions from component-level conclusions.

Raw ablation JSON:
{raw_ablation}

Return STRICT JSON with this schema:
{{
  "hypothesis_status": "supported|challenged|falsified|mixed|inconclusive",
  "feasible": true,
  "overall_confidence": 0.0,
  "tldr": "short summary",
  "key_findings": ["..."],
  "component_findings": [
    {{
      "component": "string",
      "result": "positive|negative|inconclusive",
      "metric": "string",
      "value": "string",
      "confidence": 0.0,
      "analysis": "string",
      "action_hint": "keep_or_strengthen|keep_under_review|remove_or_replace|replace_or_reframe|investigate_or_replace|investigate",
      "is_critical": false,
      "theme_tags": ["..."]
    }}
  ],
  "negative_components": ["..."],
  "positive_components": ["..."],
  "inconclusive_components": ["..."],
  "critical_failures": [
    {{
      "component": "string",
      "result": "negative",
      "metric": "string",
      "value": "string",
      "confidence": 0.0,
      "analysis": "string",
      "action_hint": "string",
      "is_critical": true,
      "theme_tags": ["..."]
    }}
  ],
  "promising_components": [
    {{
      "component": "string",
      "result": "positive",
      "metric": "string",
      "value": "string",
      "confidence": 0.0,
      "analysis": "string",
      "action_hint": "string",
      "is_critical": false,
      "theme_tags": ["..."]
    }}
  ],
  "dominant_themes": [
    {{
      "theme": "string",
      "count": 1,
      "components": ["..."]
    }}
  ]
}}
"""
