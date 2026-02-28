COMPONENT_NOVELTY_EVALUATION_PROMPT = """
You are a novelty-only evaluator for a research idea during MCTS search.

Topic:
{topic}

Candidate idea state (JSON):
{idea_state}

Component explanations used as semantic queries:
{components_with_explanations}

Nearest paper-graph evidence nodes. These are the most frequently retrieved core nodes after running top-{retrieval_top_k} component-summary semantic search for EACH component explanation.
Treat them as the closest known neighborhood in the existing paper graph.
{retrieved_nodes}

Novelty scoring policy:
- Score ONLY novelty on a 0-5 rubric.
- Use the retrieved nodes as the main prior-art neighborhood.
- If the candidate mostly restates or lightly renames mechanisms already covered by the retrieved nodes, score low.
- If the candidate meaningfully recombines nearby mechanisms but still stays close to known patterns, score mid-range.
- If the candidate introduces a concrete mechanism-level departure not well captured by the retrieved neighborhood, score high.
- Penalize vague claims of novelty that are not grounded in explicit component behavior.
- Focus on mechanism novelty, not impact, feasibility, or writing quality.

Rubric anchors:
- 0 = trivial restatement or cosmetic rename of the retrieved neighborhood.
- 1 = very small edit that preserves the same underlying mechanism.
- 2 = modest variation with limited new mechanism content.
- 3 = meaningful recombination or scoped mechanism change, but still close to nearby prior art.
- 4 = clear mechanism-level departure relative to the retrieved neighborhood.
- 5 = strongly differentiated mechanism with concrete, defensible novelty.

Return STRICT JSON only:
{{
  "rubric_score": 0-5,
  "rationale": "Short justification grounded in retrieved nodes and component behavior"
}}
"""
