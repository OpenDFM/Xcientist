ROOT_DOMAIN_CLASSIFICATION_PROMPT = """
You classify the home research domain of the ROOT idea in a memory-guided MCTS run.

Choose 1 or 2 domains from this exact catalog:
- cs.AI: Artificial Intelligence
- cs.CL: Computation and Language
- cs.CR: Cryptography and Security
- cs.CV: Computer Vision and Pattern Recognition
- cs.DS: Data Structures and Algorithms
- cs.GT: Computer Science and Game Theory
- cs.LG: Machine Learning
- cs.NE: Neural and Evolutionary Computing
- cs.RO: Robotics
- cs.SI: Social and Information Networks
- stat.ML: Machine Learning (Statistics)

Selection rules:
1. Pick the ROOT idea's home domain(s), not possible inspiration domains.
2. Use only codes from the catalog.
3. Return 1 or 2 codes only.
4. Prefer the most central domain(s) of the method, evaluation setting, and intended contribution.

== Topic ==
{topic}

== Root idea snapshot ==
{root_idea}

== Return STRICT JSON ==
{{
  "domains": ["code_1", "optional_code_2"],
  "reasoning": "brief explanation"
}}
"""
