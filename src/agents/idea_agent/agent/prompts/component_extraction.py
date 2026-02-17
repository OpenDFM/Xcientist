COMPONENT_EXTRACTION_PROMPT = """
You are an expert research scientist. Given a mature research idea, extract its key architectural / methodological components.

Each component should be a concise name (2-5 words, snake_case) representing a distinct module, mechanism, loss term, data pipeline step, or evaluation protocol described in the idea.

== Mature Idea ==
{mature_idea}

== Topic ==
{topic}

== Rules ==
1. Extract at least 1 and at most 5 components.
2. Each component must be a distinct, non-overlapping part of the idea's architecture or methodology.
3. Use short, descriptive snake_case names (e.g. "flow_matching_generator", "controllability_gramian", "value_guidance_head").
4. Do NOT include generic placeholders like "backbone_model" or "data_pipeline" — be specific to this idea.
5. Order components by architectural importance (most critical first).

Return STRICT JSON (no Markdown wrapping):
{{
  "components": ["component_name_1", "component_name_2", ...]
}}
"""
