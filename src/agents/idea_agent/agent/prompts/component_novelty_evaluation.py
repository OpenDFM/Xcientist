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
- Use the retrieved nodes as the main prior-art neighborhood, but do not stop at nearest-neighbor similarity.
- Make TWO judgments before assigning the final novelty score:
  1. Neighborhood similarity: how close is the candidate to the retrieved mechanisms?
  2. Expert novelty judgment: after accounting for those nearby works, would a domain expert still regard this idea as novel?
- If the candidate mostly restates or lightly renames mechanisms already covered by the retrieved nodes, score low.
- If the candidate meaningfully recombines nearby mechanisms but still stays close to known patterns, score mid-range.
- If the candidate introduces a concrete mechanism-level departure not well captured by the retrieved neighborhood, and would still read as novel to an expert, score high.
- High similarity usually lowers novelty, but it does not automatically force a low score if the mechanism shift is concrete and substantive.
- Low similarity does not automatically imply high novelty if the idea is vague, generic, or just repackages common patterns not captured in the retrieved set.
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
  "retrieval_similarity": 0-5,
  "perceived_novelty": 0-5,
  "rubric_score": 0-5,
  "rationale": "Short justification grounded in both retrieved-node similarity and your overall novelty judgment"
}}
"""
