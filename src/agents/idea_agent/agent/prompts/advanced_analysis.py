ADVANCED_ANALYSIS_PROMPT = """
You are the lead author preparing an ICML-caliber paper on the topic "{topic}".
You have curated the following paper capsules (title, abstract, metadata):
{papers}

Treat this as a mini literature review followed by moonshot ideation. Perform the steps below explicitly before answering:
1. Map the dominant method clusters, contrasting their assumptions, supervision signals, and compute budgets.
2. Stress-test the clusters to expose unresolved limitations, evaluation blind spots, and why those issues persist.
3. Brainstorm bold, non-incremental hypotheses that could unlock new capabilities (think novel mechanisms, new training contracts, or cross-discipline transfers). Each hypothesis must cite at least one supporting paper anchor but must go materially beyond it.
4. Identify what experimental or theoretical tooling would be required to validate those hypotheses at a top-tier (ICML/NeurIPS) standard.

Return STRICT JSON (no prose) with the schema:
{{
    "key_methods": ["..."],
    "existing_problems": ["..."],
    "evaluation_gaps": [
        {{
            "gap": "concise description of a measurement blind spot",
            "why_it_matters": "impact on reliability or scientific insight",
            "icml_expectation": "what the ICML bar would demand instead"
        }}
    ],
    "future_directions": ["..."],  # incremental yet useful next steps
    "divergent_idea_seeds": [
        {{
            "title": "short memorable name",
            "hypothesis": "what new capability emerges",
            "why_it_is_not_incremental": "specific contrast vs known tricks (e.g., gating/MoE/ensembles)",
            "method_sketch": "core mechanism, modules, or objective",
            "evaluation_plan": "new protocol/dataset/stress-test to validate it",
            "risk": "dominant scientific or engineering risk",
            "supporting_papers": ["paper title or id anchors"]
        }}
    ],
    "cross_domain_inspiration": [
        {{
            "source_field": "e.g., control theory, neuroscience",
            "transferable_mechanism": "what we borrow",
            "application_hook": "how it maps to the current topic"
        }}
    ],
    "tldr": "≤50 word synthesis tying gaps to moonshot opportunities"
}}

Rules:
- Always put at least three divergent_idea_seeds; force yourself to be expansive.
- If you cannot find enough explicit paper evidence, point to adjacent subfields in supporting_papers.
- Keep the JSON valid and free of commentary.
"""
