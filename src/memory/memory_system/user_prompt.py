from textwrap import dedent

WORKING_SLOT_FILTER_USER_PROMPT = dedent(
    """
You guard ResearchAgent's long-term memory entrance. Decide if this WorkingSlot deserves promotion into FAISS storage.

Primary goal:
- Prefer retaining actionable "what to do next" knowledge (playbooks/runbooks) over raw logs.

Promote (`yes`) when ANY of the following holds:
1) Actionability: the slot contains concrete next steps (e.g., attachments.next_actions / checks, or an explicit checklist/steps).
2) Procedural intent: tags include "scenario:run" or "scenario:debug" OR "procedure:runbook"/"procedure:playbook".
3) Debugging closure: the slot captures a clear trigger (key error line / condition) AND a follow-up action or verification.
4) Durable insight: a generalized rule/heuristic that will remain valid across tasks (semantic).

Reject (`no`) only when it is mostly noise, e.g.:
- raw stdout/stderr dumps without next actions,
- repetitive low-signal status messages ("done", "ok") without context,
- transient details that don't help decide the next step.

STRICT OUTPUT: respond with a single lowercase word: `yes` or `no`. Do not explain.

<slot-dump>
{slot_dump}
</slot-dump>
"""
)

WORKING_SLOT_ROUTE_USER_PROMPT = dedent(
    """
Map this WorkingSlot to the correct ResearchAgent long-term memory family. Choose EXACTLY one label:

- semantic → enduring insights, generalized conclusions, reusable heuristics.
- episodic → Situation → Action → Result traces with metrics, timestamps, or narrative context.
- procedural → MUST provide "when to use", and MUST include reproducible steps / commands / checklists that express "how to".
  Procedural is split by scenario:
  - scenario:run → runbook style (how to run / reproduce / execute)
  - scenario:debug → playbook style (how to diagnose / fix / verify)
  Encode scenario in tags using EXACTLY ONE of: "scenario:run" or "scenario:debug".
  Also encode type in tags using EXACTLY ONE of: "procedure:runbook" or "procedure:playbook".

Tie-breaking rules:
- Prefer episodic if a chronological action/result trail exists, even if insights appear.
- Prefer procedural when explicit steps/tools/commands are primary.
- Otherwise output semantic.

Return only one of: "semantic", "episodic", "procedural".

<slot-dump>
{slot_dump}
</slot-dump>
"""
)

WORKING_SLOT_COMPRESS_USER_PROMPT = dedent(
    """
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
"""
)

ABSTRACT_EPISODIC_TO_SEMANTIC_PROMPT = dedent(
    """
You aggregate episodic traces into a single semantic memory entry. Capture the durable lesson that explains why the cluster exists.

Instructions:
- Highlight causal mechanisms, success/failure thresholds, and metrics that repeatedly appeared.
- Mention representative stages (e.g., experiment_execute) only if they add meaning.
- Provide tags that cover both domain concepts and process cues (e.g., ["vision","fog","stability"]).
- Return STRICT JSON containing `summary`, `detail`, `tags`.

Episodic cluster notes:
{episodic_notes}
"""
)

TRANSFER_SLOT_TO_TEXT_PROMPT = dedent(
    """
Convert the WorkingSlot JSON into a concise human-readable paragraph (no tags, no JSON). This summary feeds chat surfaces, not FAISS.

Guidance:
- Mention stage, topic, and the core outcome or decision.
- Cite standout metrics or attachments inline (e.g., "accuracy climbed to 0.73").
- Describe actionable next steps only if explicitly recorded.
- Limit to 2–4 sentences; avoid bulleting or markdown.

Input WorkingSlot (JSON):

{dump_slot_json}
"""
)

TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT = dedent(
    """
Convert the Experiment Agent workflow context into at most {max_slots} WorkingSlot entries ready for filtering/routing.

Context Snapshot:
<workflow-context>
{snapshot}
</workflow-context>

Authoring rules:
1. Each slot MUST capture a single reusable takeaway (decision, discovery, bottleneck, or command).
2. `stage` MUST be a short string describing where this came from (e.g., agent_type, phase, or subsystem). Do NOT assume any fixed whitelist.
3. `summary` follows Situation → Action → Result whenever data exists; keep ≤130 words.
4. `topic` is a 3–6 word slug referencing the problem space.
5. `attachments` is optional but, when present, group similar info under keys such as
   - "notes": {{"items": []}}
   - "metrics": {{}}
   - "issues": {{"list": []}}
   - "actions": {{"list": []}}
6. `tags` is a list of lowercase keywords (≤5 items) mixing domain + workflow hints.
7. If the context lacks meaningful content, return `"slots": []` but keep the envelope.

Output STRICTLY as JSON within the tags below:
<working-slots>
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
</working-slots>
"""
)

TRANSFER_TRAJECTORY_TO_WORKING_SLOTS_PROMPT = dedent(
    """
Convert the full task trajectory (tool trace + outputs + errors) into at most {max_slots} WorkingSlot entries ready for filtering/routing.

Trajectory Snapshot:
<trajectory>
{snapshot}
</trajectory>

Authoring rules:
1. Output multiple slots: each slot captures ONE concrete, important takeaway (e.g., a fix, a command sequence, a failure root cause, a decision, a metric change).
2. Prefer specificity over generality. Avoid vague “did X successfully” unless it includes key details (paths, flags, error strings, metrics).
3. `stage` MUST be a short string describing where this came from (e.g., agent_type, phase, subsystem, or "writeback").
4. `topic` is a 3–6 word slug (e.g., "fix faiss threshold bug", "pip install numpy conflict").
5. `summary` MUST follow Situation → Action → Result → Next (SAR+N) whenever possible; keep ≤130 words.
   - "Next" is the MOST IMPORTANT part: what to do after observing the result/error.
6. `attachments` SHOULD prefer actionable follow-ups over raw logs. Use these keys when available:
   - "next_actions": {"list": ["do X", "run Y", "edit Z", "rerun W"]}
   - "checks": {"list": ["verify A", "assert B", "look for C"]}
   - "commands": {"list": ["..."]}     (only the commands needed for follow-up steps)
   - "trigger": {"text": "..."}        (minimal signature: key error line / condition)
   - "paths": {"list": ["..."]}
   - "metrics": {...}
   Raw outputs/errors are OPTIONAL; include only 1–2 lines if they define the trigger.
7. `tags` is a list of lowercase keywords (≤8 items). MUST include EXACTLY ONE scenario tag:
   - "scenario:run" for execution/reproduction/run instructions
   - "scenario:debug" for debugging/root-cause/fix flows
   For procedural-looking slots, also include EXACTLY ONE of: "procedure:runbook" or "procedure:playbook".
8. If the trajectory lacks meaningful content, return `"slots": []` but keep the envelope.

Output STRICTLY as JSON within the tags below:
<working-slots>
{{
    "slots": [
    {{
        "stage": "writeback",
        "topic": "root cause faiss query",
        "summary": "Situation/Action/Result narrative...",
        "attachments": {{
            "trigger": {{"text": "TypeError: ..."}},
            "next_actions": {{"list": ["apply fix ...", "rerun ..."]}},
            "checks": {{"list": ["verify ..."]}},
            "commands": {{"list": ["python -c ..."]}},
            "metrics": {{"count": 3}},
            "paths": {{"list": ["src/..."]}}
        }},
        "tags": ["scenario:debug","procedure:playbook","debug","memory"]
    }}
    ]
}}
</working-slots>
"""
)

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT = dedent(
    """
Transform the WorkingSlot into a semantic memory entry suitable for FAISS retrieval.

Expectations:
- `summary` (≤80 words) expresses the enduring conclusion or heuristic.
- `detail` should elaborate supporting evidence, metrics, or caveats. Use "\\n" to separate logically distinct statements.
- `tags` mixes domain terms and method/process hints.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<semantic-record>
{{
    "summary": "semantic insight summary",
    "detail": "expanded reasoning and context",
    "tags": ["keyword1","keyword2"]
}}
</semantic-record>
"""
)

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT = dedent(
    """
Convert the WorkingSlot into an episodic memory record emphasizing Situation → Action → Result.

Critical requirements:
- The MOST IMPORTANT output is "what we did next" and "what to try next time".
- Preserve only minimal trigger evidence from tools (1–2 lines) needed to recognize the situation again.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<episodic-record>
{{
    "stage": "{stage}",
    "summary": "≤80 word SAR overview",
    "detail": {{
        "situation": "Context and constraints",
        "actions": ["action 1","action 2"],
        "results": ["result 1","result 2"],
        "next_actions": ["what we did next", "what to do next time"],
        "checks": ["how to verify the fix/next step worked"],
        "metrics": {{}},
        "evidence": {{
            "trigger": ["one key error/output line"],
            "commands": ["only if needed"],
            "paths": ["only if needed"]
        }},
        "artifacts": []
    }},
    "tags": ["keyword1","keyword2"]
}}
</episodic-record>
"""
)


TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT = dedent(
    """
Convert the WorkingSlot into a procedural memory entry that captures a reusable skill or checklist.

Critical requirements:
- Tags MUST include EXACTLY ONE scenario tag: "scenario:run" or "scenario:debug".
- Tags MUST include EXACTLY ONE type tag: "procedure:runbook" or "procedure:playbook".
- Description MUST clearly state when to use (trigger/condition) AND what success looks like.
- Steps MUST follow: Trigger → Action → Verify (and optionally Rollback).
- Prefer "what to do next" over storing raw tool outputs. Tool outputs belong only as trigger signatures.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<procedural-record>
{{
    "name": "short skill name",
    "description": "≤60 words explaining when/why to apply it",
    "steps": ["step 1","step 2","step 3"],
    "code": "optional snippet or empty string",
    "tags": ["scenario:run","procedure:runbook","keyword1","keyword2"]
}}
</procedural-record>
"""
)
