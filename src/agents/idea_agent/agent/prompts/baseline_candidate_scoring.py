BASELINE_CANDIDATE_SCORING_PROMPT = (
    "You are extracting baseline evidence. Given the idea card and candidate text, "
    "return JSON with: method_family, setting, reproducibility, evidence_snippets (list), "
    "match_score (0-5), representativeness_score (0-5), reproducibility_score (0-5)."
    "\nIdeaCard: {idea_card}\nCandidateTitle: {title}\nCandidateUrl: {url}\nText: {text}\n"
)
