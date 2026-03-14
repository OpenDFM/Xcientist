COMPONENT_EXTRACTION_PROMPT = """
You are an expert research scientist. Given a mature research idea, extract its key architectural / methodological components.

Each component should be a concise name (2-5 words, snake_case) representing a distinct module, mechanism, loss term, data pipeline step, or evaluation protocol described in the idea.

== Mature Idea ==
{mature_idea}

== Topic ==
{topic}

== Previous idea component inventory (reuse these exact names whenever the revised idea still contains the same component role) ==
{prior_components}

== Latest component decisions from re_analysis_replan ==
{component_decisions}

Return STRICT JSON (no Markdown wrapping):
{{
  "components": ["component_name_1", "component_name_2", ...],
  "component_explanations": {{
    "component_name_1": "Short explanation of the role this component plays in the idea.",
    "component_name_2": "Short explanation of the role this component plays in the idea."
  }}
}}

== Rules (Strict) ==
-  Extract at least 1 and at most 5 components.
-  Each component must be a distinct, non-overlapping part of the idea's architecture or methodology.
-  Use short, descriptive snake_case names (e.g. "flow_matching_generator", "controllability_gramian", "value_guidance_head").
-  Do NOT include generic placeholders like "backbone_model" or "data_pipeline" — be specific to this idea.
-  If the revised mature idea keeps or strengthens a prior component, reuse the exact prior component name instead of inventing a synonym.
-  Component explanations may change; name reuse matters more than explanation reuse.
-  If a component is genuinely replaced with a different mechanism, create a new name only when the functional role has materially changed.
-  Prefer the smallest stable rename set: reuse old names wherever reasonable, introduce new names only for genuinely new components.
-  Order components by architectural importance (most critical first).
"""
