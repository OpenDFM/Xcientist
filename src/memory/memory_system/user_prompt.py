from textwrap import dedent

EXPERIMENT_WORKING_SLOT_FILTER_USER_PROMPT = dedent("""
You guard ResearchAgent's long-term memory entrance. Decide if this WorkingSlot deserves promotion into FAISS storage.

Assess four dimensions:
1. Novelty – is this meaningfully new compared to typical research agent discoveries?
2. Utility – can future tasks reuse the insight, metric, procedure, or decision?
3. Stability – will the information stay valid for multiple iterations (i.e., not a transient log)?
4. Evidence – do attachments, metrics, or tags provide concrete support?

Return `yes` only when at least two dimensions are clearly satisfied or the slot closes a critical loop (e.g., root-causing a failure, finishing a checklist item). Otherwise return `no`.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")

EXPERIMENT_WORKING_SLOT_ROUTE_USER_PROMPT = dedent("""
Map this WorkingSlot to the correct ResearchAgent long-term memory family. Choose EXACTLY one label:

- semantic → enduring insights, generalized conclusions, reusable heuristics.
- episodic → Situation → Action → Result traces with metrics, timestamps, or narrative context.
- procedural → MUST provide "when to use", and OPTIONAL reproducible steps, commands, pipelines, or interface contracts that express "how to".

Tie-breaking rules:
- Prefer episodic if a chronological action/result trail exists, even if insights appear.
- Prefer procedural when explicit steps/tools/commands are primary.
- Otherwise output semantic.

Return only one of: "semantic", "episodic", "procedural".

<slot-dump>
{slot_dump}
</slot-dump>
""")

IDEA_WORKING_SLOT_FILTER_USER_PROMPT = dedent("""
You guard IdeaAgent's long-term memory entrance. Decide if this IdeaAgent WorkingSlot deserves promotion into FAISS storage.

Judge using four dimensions (return `yes` only if at least TWO are clearly satisfied):
1. Utility: Will this slot help future idea generation/evaluation (e.g., reusable defect→fix heuristic, evaluation protocol, operator outcome)?
2. Stability: Is it durable (not a transient log, not a one-off chat line)?
3. Specificity: Does it include concrete details (operator, targeted defects, metrics like novelty/feasibility/lift, failure modes, fairness protocol, or referenced memory IDs)?
4. Evidence: Does the slot provide supporting rationale/metrics/attachments that make it verifiable or reusable?

Hard YES rules (return `yes` even if brief) if the slot contains ANY of:
- a reusable defect→fix recipe (defect(s) + operator/action + why + expected lift/impact);
- a fairness/baseline/ablation protocol that can be reused;
- a clearly stated failure-mode surfacing plan (what fails, how to detect, what to log);
- a durable field insight distilled from multiple candidates (not just a single idea title).

Hard NO rules (return `no`) if the slot is:
- only a raw idea title with no method/experiments/risks;
- purely meta chatter, status updates, or repetition with no new info;
- missing any actionable content (no operator/defect/metric/protocol/insight).

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
""")

IDEA_WORKING_SLOT_ROUTE_USER_PROMPT = dedent("""
Map this IdeaAgent WorkingSlot to the correct ResearchAgent long-term memory family. Choose EXACTLY one label:

- semantic → durable field knowledge, generalized conclusions, anti-pattern constraints, stable heuristics (e.g., "always add baseline+ablation", "counterfactual tests reveal X").
- episodic → Situation→Action→Result traces for a specific MCTS iteration/path, with contextual narrative, metrics, and what happened (e.g., operator applied, evaluation scores, chosen best/pareto).
- procedural → reusable "how-to" guidance: when to use + optional reproducible steps/checklist/pipeline (e.g., "how to run fairness baseline upgrade", "how to harvest defect→fix experiences").

Routing rules tailored to MemoryGuidedMCTS:
1. Prefer procedural if the slot contains explicit steps/checklists/commands OR it reads like an instruction template
   (keywords: "steps", "checklist", "do X then Y", "protocol", "pipeline", "how to", "when to use").
2. Prefer episodic if the slot describes a specific run/path/outcome:
   - mentions a concrete idea title, node/path summary, "best/pareto", iteration, or a one-time evaluation outcome,
   - includes SAR narrative with metrics (novelty/feasibility/impact/risk/conciseness/confidence/lift).
3. Otherwise output semantic when it is an enduring insight/constraint:
   - anti-pattern constraints, guardrails, defect taxonomy insights,
   - field knowledge snippets distilled for reuse,
   - stable mapping from defect→operator with rationale (without step-by-step recipe).

Return only one of: "semantic", "episodic", "procedural". No explanations.

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_COMPRESS_USER_PROMPT = dedent("""
Merge the provided WorkingSlots into ONE distilled WorkingSlot suitable for the short-term queue.

Requirements:
- Remove duplicate facts while keeping supporting metrics or attachments that future agents might need.
- Surface causal links (Situation → Action → Result) whenever present.
- Normalize tags to 1–4 lowercase tokens.
- Keep summary ≤150 words; emphasize reusable, stable insights spanning research, execution, and follow-up actions.
- If attachments include command snippets, metrics, or notes, fold only the most representative subset into the compressed slot.

Input WorkingSlots (JSON):

<slots>
{slots_block}
</slots>

Output format (STRICTLY JSON):
<compressed-slot>
{{
    "stage": "compressed",
    "topic": "concise topic slug",
    "summary": "≤150 words describing the merged knowledge",
    "attachments": {{
        "notes": {{"items": ["bullet 1","bullet 2"]}},
        "metrics": {{"name": value}},
        "procedures": {{"steps": ["step1","step2"]}},
        "artifacts": {{"paths": ["..."]}}
    }},
    "tags": ["tag1","tag2"]
}}
</compressed-slot>
""")

ABSTRACT_EPISODIC_TO_SEMANTIC_PROMPT = dedent("""
You aggregate episodic traces into a single semantic memory entry. Capture the durable lesson that explains why the cluster exists.

Instructions:
- Highlight causal mechanisms, success/failure thresholds, and metrics that repeatedly appeared.
- Mention representative stages (e.g., experiment_execute) only if they add meaning.
- Provide tags that cover both domain concepts and process cues (e.g., ["vision","fog","stability"]).
- Return STRICT JSON containing `summary`, `detail`, `tags`.

Episodic cluster notes:
{episodic_notes}
""")

TRANSFER_SLOT_TO_TEXT_PROMPT = dedent("""
Convert the WorkingSlot JSON into a concise human-readable paragraph (no tags, no JSON). This summary feeds chat surfaces, not FAISS.

Guidance:
- Mention stage, topic, and the core outcome or decision.
- Cite standout metrics or attachments inline (e.g., "accuracy climbed to 0.73").
- Describe actionable next steps only if explicitly recorded.
- Limit to 2–4 sentences; avoid bulleting or markdown.

Input WorkingSlot (JSON):

{dump_slot_json}
""")

TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT = dedent("""
Convert the Experiment Agent workflow context into at most {max_slots} WorkingSlot entries ready for filtering/routing.

Context Snapshot:
<workflow-context>
{snapshot}
</workflow-context>

Authoring rules:
1. Each slot MUST capture a single reusable takeaway (decision, discovery, bottleneck, or command).
2. `stage` MUST be one of: pre_analysis, code_plan, code_implement, code_judge, experiment_execute, experiment_analysis, meta.
3. `summary` follows Situation → Action → Result whenever data exists; keep ≤130 words.
4. `topic` is a 3–6 word slug referencing the problem space.
5. `attachments` is optional but, when present, group similar info under keys such as
   - "notes": {{"items": []}}
   - "metrics": {{}}
   - "issues": {{"list": []}}
   - "actions": {{"list": []}}
6. `tags` is a list of lowercase keywords (≤5 items) mixing domain + workflow hints.
7. If the context lacks meaningful content, return `"slots": []` but keep the envelope.

**DO NOT wrap your JSON output in markdown code blocks (```json or ```). Output raw JSON only.**
Output STRICTLY as JSON:
{{
    "slots": [
    {{
        "stage": "code_plan",
        "topic": "coverage planning",
        "summary": "Situation/Action/Result narrative...",
        "attachments": {{
            "notes": {{"items": ["detail 1", "detail 2"]}},
            "metrics": {{"acc": 0.92}},
            "issues": {{"list": []}},
            "actions": {{"list": ["follow-up 1"]}}
        }},
        "tags": ["plan","coverage"]
    }}
    ]
}}
""")

TRANSFER_IDEA_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT = dedent("""
You convert Idea Agent traces into EXACTLY ONE WorkingSlot suitable for ResearchAgent's memory queue.
You MUST output a JSON object with a "slots" array.
Do NOT output 0 slots. Do NOT output more than {max_slots} slot.

Context snapshot:
<idea-agent-context>
{snapshot}
</idea-agent-context>

Authoring directives:
1. Stage MUST be one of: {stage_enums}. Match the dominant activity in the snippet.
2. Topic is a 3–6 word slug tied to the research focus (include modality/task when possible).
3. Summary (≤130 words) MUST follow Situation→Action→Result and MUST be specific, not generic:
   - Explicitly name which edit operator(s) fired, the targeted defects, and retrieved memory snippet IDs if available.
   - Capture structured idea details (title, abstract, core_contribution, method, experiments, risks) when summarizing candidates.
   - Highlight evaluation / Q updates (novelty/feasibility/impact/risk/conciseness/confidence), Pareto role (best/novel/feasible/concise),
     and fairness / failure-mode instrumentation.
4. Add a REPRODUCIBILITY SPEC inside attachments when relevant (strongly preferred whenever an idea proposes a mechanism).
   Put it under:
   - "procedures": {{"steps": [...]}}  (for minimal harness / protocol steps)
   - "notes": {{"items": [...]}}       (for concise mechanism specs and formulas)
   - "issues": {{"list": [...]}}       (for failure modes)
   - "actions": {{"list": [...]}}      (for mitigations / next actions)
   - "metrics": {{...}}               (for target metrics / thresholds / overhead budgets)

   The REPRODUCIBILITY SPEC MUST include (use short lines; be concrete):
   A) Mechanism spec (what exactly is gated/weighted):
      - e.g., whether attention/gating applies to each loss term vs each collocation-point residual, and which residual components.
   B) Minimal formula sketch:
      - e.g., sigmoid/softmax input definition, temperature, normalization, and any moving-average / clipping.
   C) Insertion point & overhead cap:
      - where it plugs into the PINN training loop and an upper bound on parameter/compute overhead (rough but explicit).
   D) Minimal verification harness:
      - 1–2 representative PDEs, baselines (vanilla PINN / dynamic weighting / ALM), and metrics (L2 error, PDE residual,
        stability variance, generalization gap).
   E) Failure modes & guards:
      - e.g., weight collapse/oscillation -> clip, entropy regularization, EMA smoothing; what to log.
   F) Trigger conditions (when to enable):
      - symptom rules such as “one loss term dominates > threshold for N steps” or “interface residual drift > threshold”.

5. Attachments are optional but, when helpful, group info under keys like:
   - "ideas": {{"items": ["title :: abstract"]}}
   - "operators": {{"items": ["operator → defect"]}}
   - "memories": {{"items": ["Field#1 summary"]}}
   - "metrics": {{"novelty": 4.3, "lift": 18, "overhead_params_pct": 3, "trigger_loss_dom_ratio": 0.8}}
6. Tags ≤5 items mixing domain + workflow cues, e.g., ["pinn","adaptive_weighting","mcts_evaluation","repro_protocol"].
7. Always emit exactly one slot even if the agent stalled; prefer grouping by the single most actionable, testable insight.

STRICT OUTPUT (no prose, no markdown code fences):
{{
  "slots": [
    {{
      "stage": "...",
      "topic": "...",
      "summary": "...",
      "attachments": {{
        "notes": {{"items": [
          "Mechanism: gate per-loss-term weights (not per-point residual) OR specify exact residual granularity.",
          "Formula: w_i = softmax((g_i)/T), g_i = f(stats); include normalization/EMA/clipping.",
          "Insertion: compute weights before backprop; overhead <= X% params / Y% time."
        ]}},
        "procedures": {{"steps": [
          "PDE1: ...; PDE2: ...",
          "Baselines: vanilla PINN; dynamic reweighting; ALM",
          "Metrics: L2; residual; stability var; gen gap",
          "Logging: weight entropy; max/min weight; residual drift"
        ]}},
        "issues": {{"list": ["failure mode 1", "failure mode 2"]}},
        "actions": {{"list": ["guard: clip weights", "guard: entropy reg", "guard: EMA smoothing"]}},
        "metrics": {{"trigger_rule": "loss_dom_ratio>0.8 for 500 steps", "overhead_params_pct": 3}}
      }},
      "tags": ["..."]
    }}
  ]
}}
""")

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_EXPEIRMENT = dedent("""
Transform the WorkingSlot into a semantic memory entry suitable for FAISS retrieval in HotpotQA-style multi-hop QA.

Expectations:
- The semantic record MUST capture **factual evidence** grounded in the WorkingSlot (e.g., retrieved passages, environment observations), not planning logic or agent-control flow.
- `summary` (≤80 words) is a compact, question-agnostic factual statement or tightly related fact cluster that can be reused as evidence (e.g., key relations, attributes, dates, locations).
- `detail` elaborates the supporting evidence: paraphrased or briefly quoted spans, source/page titles or IDs, and important caveats. Use "\\n" to separate logically distinct atomic facts or evidence items.
- Avoid speculation or heuristic advice; only include information that is directly supported by the WorkingSlot content.
- `tags` should mix entity names, domain hints, and relation/type hints (e.g., ["hotpotqa","wikipedia","albert-einstein","birthplace"]).

<working-slot>
{dump_slot_json}
</working-slot>

**DO NOT wrap your JSON output in markdown code blocks (```json or ```). Output raw JSON only.**
Output STRICTLY as JSON:
{{
    "summary": "semantic evidence summary",
    "detail": "expanded factual evidence and context",
    "tags": ["keyword1","keyword2"]
}}
""")

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT_IDEA = dedent("""
You curate MemoryGuidedMCTS knowledge. Convert the IdeaAgent WorkingSlot into a semantic record that captures durable guidance rather than run-specific chatter.

Authoring directives:
- Summaries (≤80 words) MUST describe reusable defect→fix insights, anti-pattern guardrails, or field knowledge that hold across future searches.
- `detail` should weave the causal reasoning: reference the edit operator(s), targeted defects, fairness or failure instrumentation, cited memory IDs, and any evaluation statistics (novelty/feasibility/impact/risk/conciseness/confidence/lift) that justify the claim.
- When the slot is episodic, abstract it into the lasting principle or heuristic that another traversal could reuse (e.g., “counterfactual-contrast lifted novelty when dataset bias was detected”).
- Tags blend domain concepts, operators, and workflow cues such as ["pinn","mechanism-commit-innovation","mcts_semantic"].

WorkingSlot excerpt:
<working-slot>
{dump_slot_json}
</working-slot>

**DO NOT wrap your JSON output in markdown code blocks (```json or ```). Output raw JSON only.**
Output STRICTLY as JSON:
<semantic-record>
{{
    "summary": "generalizable conclusion",
    "detail": "multi-sentence explanation with references to operators/defects/memory IDs/metrics",
    "tags": ["keyword1","keyword2"]
}}
</semantic-record>
""")

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_EXPRIMENT = dedent("""
Convert the WorkingSlot into an episodic memory record emphasizing Situation → Action → Result.

<working-slot>
{dump_slot_json}
</working-slot>

**DO NOT wrap your JSON output in markdown code blocks (```json or ```). Output raw JSON only.**
Output STRICTLY as JSON:
{{
    "stage": "{stage}",
    "summary": "≤80 word Situation → Action → Result overview",
    "detail": {{
        "situation": "Context and constraints",
        "actions": ["action 1","action 2"],
        "results": ["result 1","result 2"],
        "metrics": {{}},
        "artifacts": []
    }},
    "tags": ["keyword1","keyword2"]
}}
""")

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT_IDEA = dedent("""
Document this IdeaAgent WorkingSlot as an episodic memory focused on a specific MCTS traversal segment (stage = {stage}).

Guidance:
- Situation should capture topic, parent idea state, and why this stage ran (e.g., "mcts_expansion on counterfactual-contrast to fix dataset_bias").
- Actions must enumerate concrete operator applications, memory bundle usage, evaluation prompts, or guardrail enforcement. Reference idea title/abstract/method snippets when helpful.
- Results must state measurable outcomes: evaluation scores (novelty/feasibility/impact/risk/clarity/conciseness/confidence/lift), fairness or failure-mode findings, Pareto role (best/novel/feasible/concise), or memory persistence decisions.
- Populate `metrics` with numeric values whenever the slot contains them; include `"path"` or `"idea_signature"` entries inside `artifacts` when mentioned (e.g., path summary, memory_refs).

WorkingSlot excerpt:
<working-slot>
{dump_slot_json}
</working-slot>

**DO NOT wrap your JSON output in markdown code blocks (```json or ```). Output raw JSON only.**
Output STRICTLY as JSON:
<episodic-record>
{{
    "stage": "{stage}",
    "summary": "≤80 word Situation → Action → Result overview",
    "detail": {{
        "situation": "Context and targeted defects/operators",
        "actions": ["action 1","action 2"],
        "results": ["result 1","result 2"],
        "metrics": {{"novelty": 4.3, "lift": 12}},
        "artifacts": ["path: ...", "memory_refs: Field#1,Recipe#2"]
    }},
    "tags": ["keyword1","keyword2"]
}}
</episodic-record>
""")

TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_EXPERIMENT = dedent("""
Convert the WorkingSlot into a procedural memory entry that captures a reusable skill or checklist.

<working-slot>
{dump_slot_json}
</working-slot>

**DO NOT wrap your JSON output in markdown code blocks (```json or ```). Output raw JSON only.**
Output STRICTLY as JSON:
{{
    "name": "short skill name",
    "description": "≤60 words explaining when/why to apply it",
    "steps": ["step 1","step 2","step 3"],
    "code": "optional snippet or empty string",
    "tags": ["keyword1","keyword2"]
}}
""")

TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT_IDEA = dedent("""
Convert the IdeaAgent WorkingSlot into a procedural memory entry that describes how to reproduce the operator-driven workflow or evaluation harness it encodes.

Expectations:
- `name` should hint at when to apply the playbook (operator + targeted defects or evaluation purpose).
- The description (≤60 words) must state trigger conditions (e.g., "use when novelty stagnates and dataset_bias is flagged") and the intended impact (lift, fairness coverage, failure surfacing).
- `steps` should be actionable and sequential: include memory retrieval prep, edit-operator injection, reproducibility spec (mechanism, formula, insertion point, overhead), evaluation/ablation requirements, and persistence/guardrail steps.
- Use `code` for any command, pseudo-code, or schema snippets referenced in the slot; leave empty string if none.
- Tags mix domain cues and workflow hints such as ["idea_agent","mcts","counterfactual-contrast","fairness_protocol"].

WorkingSlot excerpt:
<working-slot>
{dump_slot_json}
</working-slot>

**DO NOT wrap your JSON output in markdown code blocks (```json or ```). Output raw JSON only.**
Output STRICTLY as JSON:
<procedural-record>
{{
    "name": "short skill name",
    "description": "≤60 words explaining when/why to apply it",
    "steps": ["step 1","step 2","step 3"],
    "code": "optional snippet or empty string",
    "tags": ["keyword1","keyword2"]
}}
</procedural-record>
""")