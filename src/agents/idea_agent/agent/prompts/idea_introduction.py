IDEA_INTRODUCTION_PROMPT = """
You are drafting the introduction section of a research paper for the following idea.
Topic focus: {topic}
Idea payload (JSON):
{idea}

Grounding papers (title + summaries extracted from our knowledge acquisition stage):
{papers}

Write 2-3 coherent paragraphs that situate the idea within the referenced literature.
Requirements:
- Explicitly cite at least two of the provided papers by title when motivating the gap.
- Highlight how the idea inherits insights or techniques from the papers' content (parsed markdown summaries when available, otherwise title/abstract fallbacks).
- Emphasize novelty, methodological detail, and clearly articulate the research problem and proposed solution.
- Keep the tone academic and specific, mirroring the depth of an ACL/NeurIPS style introduction.

Return STRICT JSON with {{"introduction": "..."}} (no Markdown fences).
"""
