You are assisting in filtering papers for a literature review.

Below is the SEED PAPER that defines the target research direction.

--- SEED PAPER ---
TITLE: {seed_title}
ABSTRACT: {seed_abstract}
-------------------

Evaluate the following CANDIDATE PAPER.

TITLE: {candidate_title}
ABSTRACT: {candidate_abstract}

Task:
Compare the candidate paper to the seed paper and judge how strongly it fits the same research direction.
Identify both key similarities and key differences.

Return a JSON object with:
- relevance_score: integer from 0 to 5
- category: "core", "related", or "irrelevant"
- reason: one concise sentence describing the key similarity/difference

Scoring guideline:
5 = Direct continuation / same problem / same method line  
4 = Strongly related  
3 = Some peripheral similarity  
1–2 = Weak overlap  
0 = Unrelated