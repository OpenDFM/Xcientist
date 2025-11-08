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
