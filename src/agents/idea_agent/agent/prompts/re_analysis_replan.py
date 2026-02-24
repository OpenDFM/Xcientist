RE_ANALYSIS_REPLAN_PROMPT = """
You are a meticulous research engineer specialized in iterative method design.

Topic: {topic}

Current mature idea (the method you must refine — do NOT change the topic):
{mature_idea}

Advanced analysis (literature survey + gap analysis):
{analysis}

Ablation / experiment results (component-level evidence — THIS is your primary signal):
{ablation_results}

Your task:
Based on the ablation results, perform **component-level targeted modifications** to the current mature idea.
- For each ablation entry, the "component" field names a specific module/loss/mechanism, "op" indicates "add"/"remove"/"replace", and "delta_score" shows the observed metric change (negative = removing it hurts, positive = removing it helps or adding it helps).
- A negative delta_score on a "remove" op means the component is critical — keep or strengthen it.
- A positive delta_score on a "remove" op means the component is harmful or redundant — remove or replace it.
- A positive delta_score on an "add" op means the component is beneficial — incorporate or enhance it.

Do NOT change the research topic. Instead, surgically revise the method design by:
1. Identifying which components to keep, strengthen, remove, or replace based on ablation evidence.
2. Proposing specific replacement mechanisms or enhancements where needed (cite analysis insights).
3. Producing a revised mature_idea that will serve as the MCTS root node.

Respond with STRICT JSON (no prose, no Markdown):
{{
  "component_decisions": [
    {{
      "component": "string",          // component name from ablation
      "decision": "keep|strengthen|remove|replace",
      "replacement": "string or null", // if replace: describe the new mechanism; otherwise null
      "rationale": "string"            // why this decision, citing delta_score and analysis
    }}
  ],
  "mature_idea": "string",  // The revised idea (3-6 sentences). Must clearly state core hypothesis, proposed mechanism with the component modifications applied, and target problem. This will be the MCTS root node.
  "search_keywords": "string"  // Keywords for retrieving papers relevant to the NEW mechanisms introduced (max 10 words). Only needed if you introduced replacement components; otherwise repeat the last query.
}}
"""