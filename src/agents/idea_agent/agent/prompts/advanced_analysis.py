ADVANCED_ANALYSIS_PROMPT_SURVEY_LED_WITH_PAPERS = """
You are the lead author preparing an top-tier paper on the topic "{topic}".

== Mature idea (optional; if empty, ignore) ==
{mature_idea}

== Mature idea source ==
{mature_idea_source}

== Refinement scope (optional; if empty, ignore) ==
{refinement_scope}

== Refinement scope source ==
{refinement_scope_source}

You have:
(1) Survey contents (PRIMARY source of problem framing, clusters, and gaps):
{survey_contents}

(2) Curated cited-paper capsules derived from survey keynotes (SECONDARY source; use for evidence, baselines, feasibility details, and concrete instantiations ONLY):
{papers}

(3) Experiment findings extracted from raw ablation results (OPTIONAL; use only if present):
{experiment_findings}

Core principle (must follow):
- Survey drives the agenda: method clusters + unresolved mechanism bottlenecks + evaluation blind spots MUST be derived from survey_contents.
- Cited-paper capsules may refine or substantiate the survey-derived bottlenecks and gaps, but MUST NOT redefine the agenda or introduce a new main axis not present in the survey framing.
- This stage is a 1.0 -> 1.1 calibrator, NOT a 2.0 invention stage. Diagnose, constrain, and lightly refine the current idea; leave major novelty jumps to MCTS.
- If mature_idea is provided and mature_idea_source is `config_explicit` or `input_explicit`, keep the same topic, core hypothesis, and primary mechanism axis. Only make localized corrections, clarifications, or feasibility-driven adjustments.
- If mature_idea_source is `input_inferred` or `empty`, treat the current mature_idea as provisional. You may rewrite it into a more grounded mature idea as long as you stay on the same overall direction suggested by the input and survey.
- If refinement_scope is provided, treat it as a hard edit boundary. Keep repairs inside that scope and do not move the proposal's novelty to another subsystem or layer.
- If refinement_scope_source is `input_inferred` or `empty`, you may sharpen or generate a clearer grounded refinement_scope from the survey and papers.
- If experiment findings are present, use them as failure evidence, feasibility evidence, and mechanism constraints. They may invalidate a weak component or suggest a local replacement, but they MUST NOT trigger a paradigm shift.
- Prioritize mechanism bottlenecks over evaluation tooling. Evaluation blind spots are secondary unless they expose a missing mechanism that blocks a credible root idea.
- Do NOT directly introduce a new primary mechanism family. First identify which part of the current idea is weak, underspecified, brittle, or unsupported, then propose the smallest mechanism patch that repairs that limitation while preserving the original thesis.
- Do NOT default to gate/router/controller/monitor style patches just because they are easy local edits. Prefer sharpening an existing update rule, objective, representation, write policy, consolidation rule, or training contract already implicit in the current idea.
- A new gate/threshold/router is allowed only when the mature_idea already centers that decision surface and the patch merely calibrates or simplifies it. Otherwise, treat gate-like control logic as a likely sign of drift away from the original idea.
- If no clearly valuable mechanism-level local patch exists, preserve the current mature_idea instead of inventing a gate/router/controller/threshold/budget fix.
- If the only available local change mainly improves explainability, diagnosability, observability, auditing, monitoring, or validation tooling, preserve the current mature_idea instead of elevating that safe repair into the main contribution.
- If the current mature_idea is training-free or inference-time only, preserve that character when possible. Do NOT introduce a new training stage, trainable controller, auxiliary loss, fine-tuning loop, or learned module unless it is indispensable and central to repairing the identified limitation.
- For training-free ideas, prefer rule-level, objective-free, or inference-time repairs first. If you newly introduce training, explicitly justify why a training-free patch is insufficient.
- If no convincing local repair exists within refinement_scope, preserve the current mature_idea instead of escaping that scope.

Perform the steps below explicitly before answering:
1) Survey-led clustering:
   Map the dominant method clusters mentioned in the survey. For each cluster, summarize:
   - assumptions
   - supervision/training signals
   - implementation constraints or operating assumptions (as described or implied by the survey)
   Cited-paper capsules can be used only to add concrete examples, representative baselines, or implementation constraints for clusters already defined by the survey.

2) Mechanism bottleneck stress test (survey is the ground truth for "what is missing"):
   Extract unresolved mechanism-level limitations from the survey first, then list any evaluation blind spots that prevent those limitations from being measured cleanly.
   You MAY use cited-paper capsules to corroborate a bottleneck (e.g., show multiple methods still exhibit the limitation), but you MUST keep the bottleneck statement aligned with survey framing.
   Do NOT let an evaluation gap become the main novelty unless it directly reveals a missing mechanism on the current method axis.

3) Conservative root calibration:
   Produce a search-ready root idea that directly targets the extracted SURVEY mechanism bottlenecks while staying close to the current mature_idea when one is provided.
   The root idea should be a minimal but meaningful refinement:
   - read like a v1.1 update of the current idea, not a replacement idea,
   - preserve the same main method axis,
   - keep most of the existing structure if it is still defensible,
   - explicitly identify which existing component/assumption/objective is weak and what local patch fixes it,
   - prefer repairing an existing rule or mechanism already present in the idea over adding a new gating or routing layer,
   - if the only available delta is easier explanation, diagnosis, monitoring, budgeting, or validation, keep the mature_idea unchanged instead of turning that safe repair into the new thesis,
   - if the current idea is training-free, prefer training-free local repairs over introducing new optimization or fine-tuning machinery,
   - only adjust components/objectives/protocols that are weak, unsupported, or contradicted by experiment findings,
   - use evaluation gaps only as evidence for what must be measured to validate the mechanism patch, not as the main proposal.

4) Validation tooling:
   Specify what experiments/protocols/tools are required to validate the calibrated root idea credibly.
   Evaluation ideas are allowed only if they are tightly coupled to proving the proposed mechanism patch.
   Do not let the validation protocol become the primary contribution.

Return STRICT JSON (no prose, no Markdown) with the schema:
{{
  "key_methods": ["..."],                          // survey-led cluster names or dominant families
  "field_consensus": ["..."],                      // constraints / assumptions / consensus points the new idea should respect
  "existing_problems": ["..."],                    // survey-led mechanism or structural limitations (primary driver of root_idea)
  "evaluation_gaps": [
    {{
      "gap": "concise description of a measurement blind spot (MUST originate from survey_contents)",
      "why_it_matters": "impact on validating the mechanism bottleneck or proposed patch",
      "validation_expectation": "what the validation setup must include to measure that mechanism credibly"
    }}
  ],
  "future_directions": ["..."],                    // incremental but useful; must still be anchored in survey
  "preserve_current_idea": {{
    "keep_original": false,
    "reason": "empty unless no convincing valuable mechanism-level local patch exists and the mature_idea should be preserved"
  }},
  "grounded_mature_idea": "3-6 sentences; if mature_idea is empty or provisional, provide a grounded mature idea derived from the survey/papers; otherwise return empty or a minimally clarified restatement",
  "grounded_refinement_scope": "1-3 sentences; if refinement_scope is empty or provisional, provide a crisp edit boundary that matches the grounded mature idea; otherwise return empty or a minimally clarified restatement",
  "root_idea": {{
    "title": "one calibrated root idea title",
    "abstract": "one concrete root idea abstract; should read like a refined v1.1 version of the current idea when mature_idea is provided",
    "core_contribution": "main mechanism-level claim; keep it close to the current idea and express the smallest meaningful repair unless evidence forces a local correction. Prefer a repaired rule/objective/update contract over a new gate/router/controller",
    "method": "specific method sketch with modules/objective/training contract; prefer local edits over new paradigms, state which original limitation is being repaired, avoid introducing a fresh gate/router/controller unless the mature idea already depends on one, and keep training-free ideas training-free unless new training is truly necessary",
    "risks": "main scientific/engineering risks",
    "target_defects": ["..."],
    "rationale": "why this is the best calibrated 1.1 starting point from the extracted survey mechanism bottlenecks",
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
      "application_hook": "explicit mapping: which survey bottleneck/cluster it improves and how without changing the main method axis"
    }}
  ],
  "tldr": "≤50 word synthesis tying SURVEY mechanism bottlenecks to the calibrated 1.1 root idea"
}}

== Rules (Strict) ==
- Always output exactly one `root_idea`; it must be concrete enough to act as the MCTS root node.
- `root_idea` must directly address at least one named `existing_problems`; it may also respond to `evaluation_gaps`, but evaluation gaps should usually support validation rather than define the main novelty.
- If `mature_idea` is provided, `root_idea` should usually be a minimally revised version of it, not a new paradigm.
- If `refinement_scope` is provided, `root_idea` must stay inside that scope. Do not relocate the novelty to a different subsystem just because it seems easier to improve.
- Preserve the same primary method axis unless experiment findings clearly invalidate a local component; even then, prefer local replacement over architecture reset.
- `root_idea` should explicitly read as a small-version improvement of the current idea. It should say what limitation in the current idea is being repaired and should avoid introducing a fresh unrelated mechanism as the new center of gravity.
- If the current idea does not already revolve around gating/routing/control logic, do not introduce a new gate, router, controller, monitor, or threshold policy as the main repair. Prefer expressing the patch as a correction to the existing mechanism itself.
- If the only available "small fix" would be a gate/router/controller/threshold/budget wrapper, or a patch whose main value is explainability, diagnosability, observability, auditing, monitoring, or validation tooling, set `preserve_current_idea.keep_original=true` and keep `root_idea` as a faithful restatement of the current mature_idea rather than inventing a weak patch.
- When `preserve_current_idea.keep_original=true`, do not introduce any new mechanism term into `root_idea`; explain in `preserve_current_idea.reason` why the original idea is being kept unchanged.
- If the current mature idea is training-free, do not add a new training stage, learned controller, auxiliary loss, or fine-tuning loop unless it is clearly indispensable. If such a training shift is not strongly justified, prefer preserving the original idea or making a training-free local repair instead.
- Every divergent_idea_seed MUST:
  (a) cite at least one SURVEY_ANCHOR and name which existing_problems it addresses; mention evaluation_gaps only when they affect validation,
  (b) propose a concrete local mechanism patch, not just instrumentation,
  (c) keep the main axis consistent with survey framing and close to the current idea,
  (d) remain a small contrastive variant rather than a replacement proposal.
- `divergent_idea_seeds` are optional supporting alternatives; return 0-1 only if they help contrast the chosen `root_idea`.
- Papers may contribute:
  - representative baselines and fair comparison protocol details,
  - feasibility constraints (latency/memory/compute),
  - evidence that a survey bottleneck or gap persists across recent works.
  Cited-paper capsules may NOT contribute:
  - a new main problem statement that is absent from survey,
  - a new core mechanism term as the central novelty unless it is only a local substitute for a weak component.
- Evaluation gaps alone should not define the root idea unless they expose a concrete missing mechanism that the root idea patches.
- If survey_contents lacks explicit anchors, create anchors by quoting a short distinctive phrase from the survey and prefix it with "SURVEY_QUOTE:".
- Keep the JSON valid and free of commentary.
"""

# Backward-compatible alias expected by prompt registry/imports.
ADVANCED_ANALYSIS_PROMPT = ADVANCED_ANALYSIS_PROMPT_SURVEY_LED_WITH_PAPERS
