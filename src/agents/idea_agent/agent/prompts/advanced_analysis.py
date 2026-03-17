ADVANCED_ANALYSIS_PROMPT_SURVEY_LED_WITH_PAPERS = """
You are the lead author preparing an top-tier paper on the topic "{topic}".

== Mature idea (optional; if empty, ignore) ==
{mature_idea}

You have:
(1) Survey contents (PRIMARY source of problem framing, clusters, and gaps):
{survey_contents}

(2) Curated core reference capsules (SECONDARY source; use for evidence, baselines, feasibility details, and concrete instantiations ONLY):
{papers}

(3) Experiment findings extracted from raw ablation results (OPTIONAL; use only if present):
{experiment_findings}

Core principle (must follow):
- Survey drives the agenda: method clusters + unresolved gaps + evaluation blind spots MUST be derived from survey_contents.
- Core references may refine or substantiate the survey-derived gaps, but MUST NOT redefine the agenda or introduce a new main axis not present in the survey framing.
- This stage is a 1.0 -> 1.1 calibrator, NOT a 2.0 invention stage. Diagnose, constrain, and lightly refine the current idea; leave major novelty jumps to MCTS.
- If mature_idea is provided, keep the same topic, core hypothesis, and primary mechanism axis. Only make localized corrections, clarifications, or feasibility-driven adjustments.
- If experiment findings are present, use them as failure evidence, feasibility evidence, and mechanism constraints. They may invalidate a weak component or suggest a local replacement, but they MUST NOT trigger a paradigm shift.

Perform the steps below explicitly before answering:
1) Survey-led clustering:
   Map the dominant method clusters mentioned in the survey. For each cluster, summarize:
   - assumptions
   - supervision/training signals
   - compute/latency/memory budgets (as described or implied by the survey)
   Core references can be used only to add concrete examples, representative baselines, or implementation constraints for clusters already defined by the survey.

2) Gap-first stress test (survey is the ground truth for "what is missing"):
   Extract unresolved limitations + evaluation blind spots from the survey, and explain why they persist.
   You MAY use core references to corroborate a gap (e.g., show multiple methods still exhibit the limitation), but you MUST keep the gap statement aligned with survey framing.

3) Conservative root calibration:
   Produce a search-ready root idea that directly targets the extracted SURVEY gaps while staying close to the current mature_idea when one is provided.
   The root idea should be a minimal but meaningful refinement:
   - preserve the same main method axis,
   - keep most of the existing structure if it is still defensible,
   - only adjust components/objectives/protocols that are weak, unsupported, or contradicted by experiment findings.

4) Validation tooling:
   Specify what experiments/protocols/tools are required to validate the calibrated root idea at ICML/NeurIPS bar.
   Evaluation ideas are allowed only if they are tightly coupled to proving the proposed mechanism patch.

Return STRICT JSON (no prose, no Markdown) with the schema:
{{
  "key_methods": ["..."],                          // survey-led cluster names or dominant families
  "field_consensus": ["..."],                      // constraints / assumptions / consensus points the new idea should respect
  "existing_problems": ["..."],                    // survey-led limitations (can be supported by papers)
  "evaluation_gaps": [
    {{
      "gap": "concise description of a measurement blind spot (MUST originate from survey_contents)",
      "why_it_matters": "impact on reliability or scientific insight",
      "icml_expectation": "what the ICML bar would demand instead"
    }}
  ],
  "future_directions": ["..."],                    // incremental but useful; must still be anchored in survey
  "root_idea": {{
    "title": "one calibrated root idea title",
    "abstract": "one concrete root idea abstract; should read like a refined version of the current idea when mature_idea is provided",
    "core_contribution": "main mechanism-level claim; keep it close to the current idea unless evidence forces a local correction",
    "method": "specific method sketch with modules/objective/training contract; prefer local edits over new paradigms",
    "experiments": "fair validation protocol that proves the mechanism patch against named gaps",
    "risks": "main scientific/engineering risks",
    "target_defects": ["..."],
    "rationale": "why this is the best calibrated 1.1 starting point from the extracted survey gaps",
    "supporting_papers": ["SURVEY_ANCHOR:...", "PAPER_ANCHOR:..."]
  }},
  "divergent_idea_seeds": [
    {{
      "title": "short memorable name",
      "hypothesis": "small neighboring alternative only if useful",
      "why_it_is_not_incremental": "leave empty or describe only a local contrast",
      "method_sketch": "near-neighbor mechanism patch, not a new paradigm",
      "evaluation_plan": "protocol/stress-test to validate the patch",
      "risk": "dominant scientific or engineering risk",
      "supporting_papers": ["SURVEY_ANCHOR:...", "PAPER_ANCHOR:..."]
    }}
  ],
  "cross_domain_inspiration": [
    {{
      "source_field": "e.g., control theory, neuroscience",
      "transferable_mechanism": "what we borrow only if it can be expressed as a local patch",
      "application_hook": "explicit mapping: which survey gap/cluster it improves and how without changing the main method axis"
    }}
  ],
  "tldr": "≤50 word synthesis tying SURVEY gaps to the calibrated 1.1 root idea"
}}

== Rules (Strict) ==
- Always output exactly one `root_idea`; it must be concrete enough to act as the MCTS root node.
- `root_idea` must directly address at least one named `existing_problems` or `evaluation_gaps`, and must stay on the survey-led axis.
- If `mature_idea` is provided, `root_idea` should usually be a minimally revised version of it, not a new paradigm.
- Preserve the same primary method axis unless experiment findings clearly invalidate a local component; even then, prefer local replacement over architecture reset.
- Every divergent_idea_seed MUST:
  (a) cite at least one SURVEY_ANCHOR and name which evaluation_gaps/existing_problems it addresses,
  (b) propose a concrete local mechanism patch, not just instrumentation,
  (c) keep the main axis consistent with survey framing and close to the current idea.
- `divergent_idea_seeds` are optional supporting alternatives; return 0-1 only if they help contrast the chosen `root_idea`.
- Papers may contribute:
  - representative baselines and fair comparison protocol details,
  - feasibility constraints (latency/memory/compute),
  - evidence that a survey gap persists across recent works.
  Core references may NOT contribute:
  - a new main problem statement that is absent from survey,
  - a new core mechanism term as the central novelty unless it is only a local substitute for a weak component.
- If survey_contents lacks explicit anchors, create anchors by quoting a short distinctive phrase from the survey and prefix it with "SURVEY_QUOTE:".
- Keep the JSON valid and free of commentary.
"""

# Backward-compatible alias expected by prompt registry/imports.
ADVANCED_ANALYSIS_PROMPT = ADVANCED_ANALYSIS_PROMPT_SURVEY_LED_WITH_PAPERS
