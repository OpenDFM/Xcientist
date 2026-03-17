RE_ANALYSIS_REPLAN_PROMPT = """
You are a meticulous research engineer specialized in iterative method design.

== Topic ==
{topic}

== Current mature idea (the method you must refine — do NOT change the topic) ==
{mature_idea}

== Advanced analysis ==
{analysis}

== Ablation / experiment results (component-level evidence — THIS is your primary signal) ==
{ablation_results}

Your task:
Based on the ablation results, perform **minimal component-level targeted modifications** to the current mature idea.
- For each ablation entry, the "component" field names a specific module/loss/mechanism, "op" usually indicates "remove", "result" is "positive|negative|inconclusive", and "confidence" tells you how reliable the evidence is.
- If `op=remove` and `result=negative`, removing the component hurt; the component is likely critical, so keep or strengthen it.
- If `op=remove` and `result=positive`, removing the component helped; the component is likely harmful or redundant, so remove or replace it.
- If `result=inconclusive`, treat the evidence as weak and avoid overcommitting unless analysis strongly supports a change.
- This stage is a 1.0 -> 1.1 patch stage, NOT a 2.0 invention stage.
- Preserve the same topic, core hypothesis, and overall method axis.
- If you use `replace`, it must be a local substitute for the same functional slot, not a new architecture or new research paradigm.

Do NOT change the research topic. Instead, surgically revise the method design by:
1. Identifying which components to keep, strengthen, remove, or replace based on ablation evidence.
2. Proposing specific local replacement mechanisms or enhancements where needed (cite analysis insights).
3. Producing a revised mature_idea that stays very close to the current one and will serve as the MCTS root node.

Respond with STRICT JSON (no prose, no Markdown):
{{
  "component_decisions": [
    {{
      "component": "string",          // component name from ablation
      "decision": "keep|strengthen|remove|replace",
      "replacement": "string or null", // if replace: describe the local substitute; otherwise null
      "rationale": "string"            // why this decision, citing result/confidence and analysis
    }}
  ],
  "mature_idea": "string",  // The revised idea (3-6 sentences). It must read like a minimally patched version of the current idea, not a new paradigm. This will be the MCTS root node.
  "search_keywords": "string"  // Keywords for retrieving papers relevant to the local replacement or refined mechanism (max 10 words). If nothing important changed, repeat the last query.
}}
"""
