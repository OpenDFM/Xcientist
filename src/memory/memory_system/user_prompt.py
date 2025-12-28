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
You MUST output a JSON object with a "slots" array that contains exactly 1 element.
Do NOT output 0 slots. Do NOT output more than {max_slots} slot.

Context snapshot:
<idea-agent-context>
{snapshot}
</idea-agent-context>

Authoring directives:
1. Stage MUST be one of: {stage_enums}. Match the dominant activity recorded in the snippet.
2. Topic is a 3–6 word slug tying the slot to the research focus (include modality/task when possible).
3. Summary (≤130 words) follows Situation→Action→Result, explicitly referencing memory-guided MCTS behavior:
   - mention which edit operator(s) fired, the targeted defects, and retrieved memory snippet IDs if available;
   - capture structured idea details (title, abstract, core_contribution, method, experiments, risks) when summarizing candidates;
   - highlight evaluation/Q updates (novelty/feasibility/etc.), Pareto role (best/novel/feasible/concise), and fairness/failure-mode instrumentation.
4. Attachments are optional but, when helpful, group info under keys like
   - "ideas": {{"items": ["title :: abstract"]}}
   - "operators": {{"items": ["operator → defect"]}}
   - "metrics": {{"novelty": 4.3, "lift": 18}}
   - "memories": {{"items": ["Field#1 summary"]}}
   - "actions": {{"list": ["write ltm defect→fix"]}}
5. Tags ≤5 items mixing domain + workflow cues, e.g., ["diffusion","mcts_evaluation","fairness"].
6. Always emit at least one slot even if the agent stalled; prefer grouping by actionable insight rather than chronology.

STRICT OUTPUT (no prose, no markdown code fences):
{{
    "slots": [
        {{
            "stage": "...",
            "topic": "...",
            "summary": "...",
            "attachments": {{"notes": {{"items": []}}}},
            "tags": ["..."]
        }}
    ]
}}
""")

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT = dedent("""
Transform the WorkingSlot into a semantic memory entry suitable for FAISS retrieval.

Expectations:
- `summary` (≤80 words) expresses the enduring conclusion or heuristic.
- `detail` should elaborate supporting evidence, metrics, or caveats. Use "\\n" to separate logically distinct statements.
- `tags` mixes domain terms and method/process hints.

<working-slot>
{dump_slot_json}
</working-slot>

**DO NOT wrap your JSON output in markdown code blocks (```json or ```). Output raw JSON only.**
Output STRICTLY as JSON:
{{
    "summary": "semantic insight summary",
    "detail": "expanded reasoning and context",
    "tags": ["keyword1","keyword2"]
}}
"""
)

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT = dedent("""
Convert the WorkingSlot into an episodic memory record emphasizing Situation → Action → Result.

<working-slot>
{dump_slot_json}
</working-slot>

**DO NOT wrap your JSON output in markdown code blocks (```json or ```). Output raw JSON only.**
Output STRICTLY as JSON:
{{
    "stage": "{stage}",
    "summary": "≤80 word SAR overview",
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


TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT = dedent("""
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
