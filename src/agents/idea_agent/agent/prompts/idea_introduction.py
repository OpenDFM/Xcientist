IDEA_INTRODUCTION_PROMPT = """
You are drafting the introduction section of a research paper for the following idea.
Topic focus: {topic}
Idea payload (JSON):
{idea}

Grounding papers (title + summaries extracted from our knowledge acquisition stage):
{papers}

Write 2-4 coherent paragraphs that situate the idea within the referenced literature and explain the proposed idea in concrete detail.
Requirements:
- Explicitly cite at least two of the provided papers by title when motivating the gap.
- Highlight how the idea inherits insights or techniques from the papers' content (parsed markdown summaries when available, otherwise title/abstract fallbacks).
- Emphasize novelty, methodological detail, and clearly articulate the research problem, core mechanism, and why the proposed design addresses the gap.
- Treat the idea as a proposal, not a completed paper with finished experiments.
- You may describe planned evaluations, expected benefits, target failure modes, and hypotheses.
- You must NOT invent, imply, or summarize experimental results for the proposed idea unless such results are explicitly present in the provided idea payload.
- If the provided idea payload does not contain experiment results, do NOT write sentences claiming the idea "achieves", "outperforms", "improves by X", "shows gains", "demonstrates superior performance", or any equivalent result statement about this idea.
- You may mention empirical findings from the grounding papers as prior-work evidence, but keep them clearly attributed to those papers rather than to the proposed idea.
- Keep the tone academic and specific, mirroring the depth of an ACL/NeurIPS style introduction.

Return STRICT JSON with {{"introduction": "..."}} (no Markdown fences).
"""
