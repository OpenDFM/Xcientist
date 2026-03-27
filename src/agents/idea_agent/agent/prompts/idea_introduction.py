IDEA_INTRODUCTION_PROMPT = """
You are an expert academic writer drafting the introduction section of a research paper for the following idea.

== Topic == 
{topic}

== Mature idea anchor (optional; if empty, ignore) ==
{mature_idea}

== Idea payload (JSON) ==
{idea}

== Grounding papers (summaries extracted from our knowledge acquisition stage) ==
{papers}

First, plan the narrative arc in the `outline_scratchpad`. Then, write 2-4 coherent paragraphs that situate the idea within the referenced literature and explain the proposed mechanism.

Requirements & Constraints:
- NARRATIVE FLOW: Follow a logical progression (e.g., Broad context -> Specific limitations/Gap -> Core proposed mechanism -> Hypotheses & Evaluation plan).
- ANCHORING: If a mature idea anchor is provided, explicitly explain that the proposal is a refinement of that idea and identify the specific limitation being repaired.
- GROUNDING: Explicitly reference at least two provided papers when motivating the gap. Highlight how your idea inherits or contrasts with their specific insights. (Use standard academic phrasing, e.g., "Building on the insights of [Paper Title]...").
- COMPARISON BOUNDARY: If you compare against prior methods, only use the provided papers or the mature idea anchor. Never compare against internal MCTS intermediate artifacts, parent nodes, or temporary aliases.
- STANCE: Treat the idea as a PROPOSAL. Emphasize methodological detail, target failure modes, and expected benefits.
- ANTI-HALLUCINATION (CRITICAL): Do NOT invent or summarize experimental results unless explicitly present in the idea payload. Absolutely NO phrases claiming the idea "achieves", "outperforms", "improves by X", or "demonstrates superior performance". 
- FORMATTING: Return the introduction as a list of strings, where each string is exactly one paragraph.

Return STRICT JSON ONLY matching this schema:
{{
  "introduction": ""
}}
"""
