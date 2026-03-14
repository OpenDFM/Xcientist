IDEA_INTRODUCTION_PROMPT = """
You are an expert academic writer drafting the introduction section of an ACL/NeurIPS-style research paper for the following idea.

== Topic == 
{topic}

== Idea payload (JSON) ==
{idea}

== Grounding papers (summaries extracted from our knowledge acquisition stage) ==
{papers}

First, plan the narrative arc in the `outline_scratchpad`. Then, write 2-4 coherent paragraphs that situate the idea within the referenced literature and explain the proposed mechanism.

Requirements & Constraints:
- NARRATIVE FLOW: Follow a logical progression (e.g., Broad context -> Specific limitations/Gap -> Core proposed mechanism -> Hypotheses & Evaluation plan).
- GROUNDING: Explicitly reference at least two provided papers when motivating the gap. Highlight how your idea inherits or contrasts with their specific insights. (Use standard academic phrasing, e.g., "Building on the insights of [Paper Title]...").
- STANCE: Treat the idea as a PROPOSAL. Emphasize methodological detail, target failure modes, and expected benefits.
- ANTI-HALLUCINATION (CRITICAL): Do NOT invent or summarize experimental results unless explicitly present in the idea payload. Absolutely NO phrases claiming the idea "achieves", "outperforms", "improves by X", or "demonstrates superior performance". 
- FORMATTING: Return the introduction as a list of strings, where each string is exactly one paragraph.

Return STRICT JSON ONLY matching this schema:
{{
  "outline_scratchpad": {{
    "para_1_context": "What is the background and why does this topic matter?",
    "para_2_gap": "What are the specific limitations of the referenced papers?",
    "para_3_mechanism": "What is the core proposed method and how does it uniquely address the gap?",
    "para_4_hypotheses": "What are the planned evaluations and expected scientific takeaways?"
  }},
  "introduction": [
    "Paragraph 1 text here...",
    "Paragraph 2 text here...",
    "Paragraph 3 text here...",
    "Paragraph 4 text here (optional, depending on length)..."
  ]
}}
"""