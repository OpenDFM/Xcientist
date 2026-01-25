DATASET_CANDIDATE_SCORING_PROMPT = (
    "You are extracting dataset evidence. Given the idea card and candidate text, "
    "return JSON with: dataset_name, usage, access, license, dataset_type, evidence_snippets (list), "
    "match_score (0-5), scale_score (0-5), availability_score (0-5)."
    "\nIdeaCard: {idea_card}\nCandidateTitle: {title}\nCandidateUrl: {url}\nText: {text}\n"
)
