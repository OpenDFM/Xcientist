COMPONENT_NOVELTY_EVALUATION_PROMPT = """
You are a strictly objective novelty evaluator for a research idea during an MCTS search process.

== Topic ==
{topic}

== Candidate idea state (JSON) ==
{idea_state}

== Component explanations used as semantic queries ==
{components_with_explanations}

== Nearest paper-graph evidence nodes ==
These are the most frequently retrieved core nodes after running top-{retrieval_top_k} component-summary semantic search for EACH component explanation.
Treat them as the closest known prior-art neighborhood in the existing paper graph.
{retrieved_nodes}

Novelty scoring policy:
- Score ONLY mechanism novelty on a 0-5 rubric. Ignore impact, feasibility, or writing quality.
- Use the retrieved nodes as the main baseline, but do not stop at nearest-neighbor text similarity.
- High similarity usually lowers novelty, but it does NOT automatically force a low score if the mechanism shift is concrete and scientifically substantive.
- Low similarity does NOT automatically imply high novelty if the idea is vague, generic, or just repackages common patterns not captured in the retrieved set. Penalize vague claims.
- COMPONENT AGGREGATION RULE: If an idea combines standard/highly-similar components with at least ONE highly novel, structurally sound component that fundamentally changes the system's capability, anchor the overall `rubric_score` on the strength of that novel component (do not simply average them).
- POLARITY FOR SIMILARITY: 5 = Nearly identical/heavy overlap with retrieved nodes. 0 = Completely orthogonal/unrelated to retrieved nodes.

Rubric anchors (for the final `rubric_score`):
- 0 = Trivial restatement or cosmetic rename of the retrieved neighborhood.
- 1 = Very small edit that preserves the same underlying mechanism.
- 2 = Modest variation with limited new mechanism content.
- 3 = Meaningful recombination or scoped mechanism change, but still close to nearby prior art.
- 4 = Clear mechanism-level departure relative to the retrieved neighborhood.
- 5 = Strongly differentiated mechanism with concrete, defensible novelty.

First, conduct your analysis in the `evaluation_scratchpad`. Then, assign the numeric scores.
Return STRICT JSON ONLY (no markdown blocks like ```json, no prose outside the JSON object):
{{
  "evaluation_scratchpad": {{
    "mechanism_delta": "Identify the EXACT mechanism difference (the 'Delta') between the candidate's core components and the closest retrieved nodes.",
    "similarity_analysis": "Analyze how syntactically/structurally close this is to the baseline graph.",
    "novelty_defense": "Determine if the Delta is scientifically non-trivial, or just a cosmetic rename/vague claim."
  }},
  "retrieval_similarity": <int 0-5>,
  "perceived_novelty": <int 0-5>,
  "rubric_score": <int 0-5>,
  "rationale": "One concise sentence summarizing why this specific rubric_score was chosen based on the mechanism delta."
}}
"""