from textwrap import dedent

WORKING_SLOT_FILTER_USER_PROMPT = dedent("""
Determine whether this slot should be converted to long-term memory (LTM).

Evaluation dimensions: novelty (new information), utility (reusable value), stability (whether it is not easily outdated).

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_ROUTE_USER_PROMPT = dedent("""
Classify this slot into one of the following categories:

-semantic: General conclusions/rules that can be reused across tasks

-episodic: A certain process (S→A→R), including indicators/results

-procedural: Practices/steps/commands/function calls that can be reused as skills

Only output a string, either "semantic", "procedural", or "episodic".

<slot-dump>
{slot_dump}
</slot-dump>
""")

WORKING_SLOT_COMPRESS_USER_PROMPT = dedent("""
Read the following WorkingSlots and compress them into ONE consolidated WorkingSlot.

Your goals:
- Deduplicate overlapping content while keeping key facts.
- Prefer information that is novel, useful across tasks, and stable (unlikely to expire soon).

Input WorkingSlots (JSON):

<slots>
{slots_block}
</slots>

Output format (STRICTLY JSON):
<compressed-slot>
{{
    "stage": "compressed",
    "topic": "a short topic slug",
    "summary": "≤150-word compact synthesis",
    "attachments": {{
        "notes": {{"items": ["bullet1","bullet2"]}},
        "metrics": {{"acc": 0.91}},
        "procedures": {{"steps": ["step1","step2"]}},
        "sources": {{"ids": ["src1","src2"]}}
    }},
    "tags": ["tag1","tag2"]
}}
</compressed-slot>
""")

ABSTRACT_EPISODIC_TO_SEMANTIC_PROMPT = dedent("""
Summarize the episodic records below into a single semantic memory entry.
Highlight enduring insights, causal links, and measurable outcomes.
Respond with JSON containing `summary`, `detail`, `tags`.

{episodic_notes}
""")

TRANSFER_SLOT_TO_TEXT_PROMPT = dedent("""
Convert the following WorkingSlot JSON into a concise text summary.

Input WorkingSlot (JSON):

{dump_slot_json}

Output format: A plain text, WITHOUT ANY WRAPTAG.
[Your concise text summary here]

SUMMARY GUIDELINES:
- Highlight key insights and important metrics.
- Include actionable items or next steps if present.
- Keep it clear, concise, and focused on utility.
- Avoid unnecessary details or jargon.

STRICT CONTRACT:
- Output ONLY the text summary wrapped in the specified tags.
""")

TRANSFER_EXPERIMENT_AGENT_CONTEXT_TO_WORKING_SLOTS_PROMPT = dedent("""
    Convert the following Experiment Agent workflow context into at most {max_slots} WorkingSlot entries.

    Context Snapshot:
    <workflow-context>
    {snapshot}
    </workflow-context>

    Guidelines:
    1. Identify distinct, reusable takeaways (e.g., agent decisions, execution outcomes, major findings).
    2. stage must be one of: pre_analysis, code_plan, code_implement, code_judge,
        experiment_execute, experiment_analysis, meta.
    3. topic is a 3-6 word slug describing the slot focus.
    4. attachments is a dict (you may include keys such as notes ({{"items": []}}),
        metrics ({{}}), issues ({{"list": []}}), actions ({{"list": []}}), artifacts ({{"paths": []}})).
        Omit keys that aren't applicable.
    5. tags is a short list of lowercase keywords.
    6. If there is insufficient information, return an empty slots list but still follow the specified JSON format.

    Output STRICTLY as JSON within the tags below:
    <working-slots>
    {{
        "slots": [
        {{
            "stage": "code_plan",
            "topic": "plan coverage",
            "summary": "Describe situation, action, result.",
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

TRANSFER_SLOT_TO_SEMANTIC_RECORD_PROMPT = dedent("""
    Convert the WorkingSlot below into a semantic memory record.

    <working-slot>
    {dump_slot_json}
    </working-slot>

    Output STRICTLY as JSON inside the tags:
    <semantic-record>
    {{
        "summary": "≤80 word overview of the key insight",
        "detail": "Detailed explanation (you may use '\\n' to separate points)",
        "tags": ["keyword1","keyword2"]
    }}
    </semantic-record>
    """
)

TRANSFER_SLOT_TO_EPISODIC_RECORD_PROMPT = dedent("""
    Convert the WorkingSlot below into an episodic memory record.

    <working-slot>
    {dump_slot_json}
    </working-slot>

    Output STRICTLY as JSON inside the tags:
    <episodic-record>
    {{
        "stage": "{stage}",
        "summary": "≤80 word SAR-style overview",
        "detail": {{
        "situation": "Context and constraints",
        "actions": ["action 1","action 2"],
        "results": ["result 1","result 2"],
        "metrics": {{}},
        "artifacts": []
        }},
        "tags": ["keyword1","keyword2"]
    }}
    </episodic-record>
    """
)


TRANSFER_SLOT_TO_PROCEDURAL_RECORD_PROMPT = dedent("""
Convert the WorkingSlot below into a procedural memory record.

<working-slot>
{dump_slot_json}
</working-slot>

Output STRICTLY as JSON inside the tags:
<procedural-record>
{{
    "name": "short skill name",
    "description": "≤60 word explanation of when/why to use it",
    "steps": ["step 1","step 2","step 3"],
    "code": "optional snippet or empty string",
    "tags": ["keyword1","keyword2"]
}}
</procedural-record>
"""
)