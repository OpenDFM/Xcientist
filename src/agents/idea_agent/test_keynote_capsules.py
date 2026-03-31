from src.agents.idea_agent.utils.prompting.prompt_views import format_paper_capsules_prompt_view
from src.agents.idea_agent.utils.workflow.ligagent_utils import collect_paper_context_entries


def test_format_paper_capsules_prompt_view_renders_paper_and_group_summaries():
    rendered = format_paper_capsules_prompt_view(
        [
            {
                "title": "Paper A",
                "paper_id": "paper_a",
                "reference_mode": "paper_summary",
                "summary": "Compressed summary for paper A.",
                "insight": "Compressed insight for paper A.",
                "score": 92,
            },
            {
                "title": "Remaining survey-cited papers (7)",
                "paper_id": "remaining_survey_cited_papers",
                "reference_mode": "group_summary",
                "summary": "Merged summary for the remaining papers.",
                "score": 61,
            },
        ]
    )

    assert "Type: paper_summary" in rendered
    assert "Compressed summary for paper A." in rendered
    assert "Compressed insight for paper A." in rendered
    assert "Type: group_summary" in rendered
    assert "Merged summary for the remaining papers." in rendered


def test_collect_paper_context_entries_uses_summary_and_insight():
    entries = collect_paper_context_entries(
        artifact={},
        reference_batches=[
            [
                {
                    "paper_id": "paper_a",
                    "title": "Paper A",
                    "reference_mode": "paper_summary",
                    "summary": "Compressed summary for paper A.",
                    "insight": "Detailed note.",
                }
            ]
        ],
    )

    assert len(entries) == 1
    assert entries[0]["title"] == "Paper A"
    assert "Compressed summary for paper A." in entries[0]["summary"]
    assert "Detailed note." in entries[0]["summary"]
