from utils.rich_logger import get_logger
from utils.api_call import SemanticScholarAPI, ChatAgent
import math
from modules.pe import (
    SURVEY_OUTLINE_GENERATION,
    SUBSECTION_DRAFT,
    SECTION_DRAFT,
    DRAFT_REFINEMENT,
)
from utils.utils import extract_json
import textwrap
from tqdm import tqdm


class SurveyGenerator:
    def __init__(self, config, work_analyzer):
        self.config = config
        self.logger = get_logger("SurveyGenerator")
        self.chat_agent = ChatAgent(config)
        self.work_analyzer = work_analyzer

    def format_papers_analysis(self, intra_analysis_results, inter_analysis_results):
        papers_analysis = ""
        for i, group in enumerate(intra_analysis_results):
            papers_analysis += f"Group {i+1} Analysis:\n"
            for j, q in enumerate(group):
                papers_analysis += f"Question {j+1}: {q['question']}\nAnswer {j + 1}: {q['answer']}\nRelated Papers {j + 1}: {q['related_papers']}\n\n---\n"

        papers_analysis += (
            f"\n\nHigh-level Inter-Group Analysis:\n{inter_analysis_results}\n"
        )
        return papers_analysis

    def generate_outline(
        self, intra_analysis_results, inter_analysis_results, papers, retry=1
    ):
        try:
            papers_analysis = self.format_papers_analysis(
                intra_analysis_results, inter_analysis_results
            )
            outline = {}

            # iteratively generate outline
            num_batches = math.ceil(
                len(papers)
                / self.config.ModuleInfo.SurveyGenerator.outline_generation_batch_size
            )
            for batch_idx in range(num_batches):
                self.logger.info(
                    f"Generating outline: processing batch {batch_idx + 1} of {num_batches}"
                )
                batch_papers = papers[
                    batch_idx
                    * self.config.ModuleInfo.SurveyGenerator.outline_generation_batch_size : (
                        batch_idx + 1
                    )
                    * self.config.ModuleInfo.SurveyGenerator.outline_generation_batch_size
                ]
                paper_keynotes = ""
                keynote_papers = []
                for paper_id in batch_papers:
                    keynote = self.work_analyzer.get_paper_keynote(paper_id)
                    paper_keynotes += f"Paper ID: {paper_id}\nKeynote: {keynote}\n\n"
                    # self.logger.info(f"keynote Paper ID in outline generation: {paper_id}\n") # YZY DEBUG
                    keynote_papers.append(paper_id)

                prompt = SURVEY_OUTLINE_GENERATION.format(
                    paper_keynotes=paper_keynotes,
                    current_outline=outline,
                    papers_analysis=papers_analysis,
                )

                valid = False
                for _ in range(
                    self.config.ModuleInfo.SurveyGenerator.outline_generation_max_retry_in_generation_loop
                ): ## YZY MODIFY :change retry mode to improve success
                    renew_outline = extract_json(
                        self.chat_agent.remote_chat(
                            prompt,
                            temperature=self.config.ModuleInfo.SurveyGenerator.outline_generation_temperature,
                        )
                    )
                    valid = self.validate_outline(intra_analysis_results, keynote_papers, renew_outline)
                    if valid:
                        outline = renew_outline
                        break
                    else:
                        self.logger.warning(f"Outline validation failed for batch {batch_idx + 1}. Retrying...")
                if not valid:
                    raise ValueError("Invalid paper ID after maximum retries in loop.")

            return outline
        except Exception as e:
            if (
                retry
                > self.config.ModuleInfo.SurveyGenerator.outline_generation_max_retry
            ):
                raise e
            return self.generate_outline(
                intra_analysis_results,
                inter_analysis_results,
                papers,
                retry=retry + 1,
            )

    def validate_outline(self, intra_analysis_results, keynote_papers, outline):
        valid_papers = set()
        for group in intra_analysis_results:
            for q in group:
                valid_papers.update(q["related_papers"])
        
        #YZY MODIFY: only for debug
        for paper_id in keynote_papers:
            valid_papers.add(paper_id)

        self.logger.info(f"ALL Validate paper ID: {valid_papers}") # YZY DEBUG

        for section in outline.get("sections", []):
            for paper_id in section.get("papers_to_use", []):
                self.logger.info(f"Validating paper ID: {paper_id}") # YZY DEBUG
                if paper_id not in valid_papers:
                    self.logger.error(
                        f"Paper ID {paper_id} in section '{section.get('title', '')}' is not in the valid papers set."
                    )
                    # raise ValueError("Invalid paper ID in outline section.")
                    return False

            for subsection in section.get("subsections", []):
                for paper_id in subsection.get("papers_to_use", []):
                    if paper_id not in valid_papers:
                        self.logger.error(
                            f"Paper ID {paper_id} in subsection '{subsection.get('title', '')}' is not in the valid papers set."
                        )
                        # raise ValueError("Invalid paper ID in outline subsection.")
                        return False

            return True

    def log_outline(
        self, outline, width=100, max_papers_display=100, desc_preview_len=None
    ):
        """
        Pretty-print outline to the logger.
        - width: wrap width for descriptions
        - max_papers_display: if many papers, only show this many then "..."
        - desc_preview_len: if set, truncate description to this many chars before wrapping
        """

        def format_desc(desc):
            if not desc:
                return ""
            if desc_preview_len is not None and len(desc) > desc_preview_len:
                desc = desc[:desc_preview_len].rstrip() + "..."
            return "\n".join(textwrap.wrap(desc, width=width))

        def format_papers(papers):
            if not papers:
                return ""
            if len(papers) > max_papers_display:
                shown = papers[:max_papers_display]
                return (
                    ", ".join(shown) + f", ...(+{len(papers)-max_papers_display} more)"
                )
            return ", ".join(papers)

        def print_section(sec, indent=0):
            prefix = " " * indent
            lines = []
            title = sec.get("title", "<no title>")
            lines.append(f"{prefix}- {title}")

            desc = sec.get("description", "")
            if desc:
                wrapped = format_desc(desc)
                # indent wrapped description one level further
                for dline in wrapped.splitlines():
                    lines.append(f"{prefix}  {dline}")

            papers = sec.get("papers_to_use", []) or sec.get("papers", [])
            if papers:
                lines.append(f"{prefix}  papers: {format_papers(papers)}")

            # recurse subsections
            for sub in sec.get("subsections", []):
                lines.append(print_section(sub, indent + 2))

            return "\n".join(lines)

        # build full text
        out_lines = []
        out_lines.append("=== Generated Survey Outline ===")
        out_lines.append(f"Survey Title: {outline.get('title', '<no title>')}\n")

        for sec in outline.get("sections", []):
            out_lines.append(print_section(sec))
            out_lines.append("")  # blank line between top-level sections

        pretty = "\n".join(out_lines)
        self.logger.info(pretty)

    def draft_survey(self, intra_analysis_results, inter_analysis_results, outline):
        relevant_analysis = self.format_papers_analysis(
            intra_analysis_results, inter_analysis_results
        )

        # step 1: subsection draft
        subsection_prompts = []
        for section in outline.get("sections", []):
            for subsection in section.get("subsections", []):
                papers = ""
                for paper_id in subsection.get("papers_to_use", []):
                    paper_raw_markdown = self.work_analyzer.get_paper_raw_markdown(
                        paper_id
                    )
                    papers += (
                        f"Paper ID: {paper_id}\nRaw markdown: {paper_raw_markdown}\n\n"
                    )
                prompt = SUBSECTION_DRAFT.format(
                    title=subsection.get("title", ""),
                    description=subsection.get("description", ""),
                    relevant_analysis=relevant_analysis,
                    papers=papers,
                )
                subsection_prompts.append(prompt)
        subsection_drafts = self.chat_agent.batch_remote_chat(
            subsection_prompts,
            desc="Drafting survey subsections...",
            temperature=self.config.ModuleInfo.SurveyGenerator.subsection_draft_temperature,
        )
        if self.config.BasicInfo.debug:
            self.logger.info(f"SUBSECTION DRAFTS: {subsection_drafts}")

        # step 2: section draft
        section_prompts = []
        idx = 0
        for section in outline.get("sections", []):
            subsection_drafts = "\n\n".join(
                subsection_drafts[idx : idx + len(section.get("subsections", []))]
            )
            idx += len(section.get("subsections", []))

            subsection_paper_ids = set()
            for subsection in section.get("subsections", []):
                subsection_paper_ids.update(subsection.get("papers_to_use", []))

            papers = ""
            for paper_id in section.get("papers_to_use", []):
                if paper_id in subsection_paper_ids:
                    continue  # already included in subsections
                paper_raw_markdown = self.work_analyzer.get_paper_raw_markdown(paper_id)
                papers += (
                    f"Paper ID: {paper_id}\nRaw markdown: {paper_raw_markdown}\n\n"
                )

            prompt = SECTION_DRAFT.format(
                title=section.get("title", ""),
                description=section.get("description", ""),
                subsection_drafts=subsection_drafts,
                papers=papers,
            )
            section_prompts.append(prompt)
        section_drafts = self.chat_agent.batch_remote_chat(
            section_prompts,
            desc="Drafting survey sections...",
            temperature=self.config.ModuleInfo.SurveyGenerator.section_draft_temperature,
        )
        if self.config.BasicInfo.debug:
            self.logger.info(f"SECTION DRAFTS: {section_drafts}")

        return (
            outline.get("title", "Untitled Survey")
            + "\n\n"
            + "\n\n".join(section_drafts)
        )

    def refine_draft(self, draft):
        prompt = DRAFT_REFINEMENT.format(draft_text=draft)
        output = self.chat_agent.remote_chat(
            prompt,
            temperature=self.config.ModuleInfo.SurveyGenerator.draft_refinement_temperature,
        )
        output = extract_json(output)

        survey = output.get("refined_survey", draft)
        references = output.get("references", [])

        self.logger.info(f"Total references found: {len(references)}. Generating...")
        reference = "References:\n"
        for index, paper_id in tqdm(enumerate(references)):
            self.logger.info(f" Generating reference for paper ID: {paper_id}") # YZY DEBUG
            reference += f"{index + 1}. {self.work_analyzer.generate_mla(paper_id)}\n"

        return survey + "\n\n" + reference

    def save_survey(self, final_survey):
        with open(self.config.BasicInfo.save_path, "w", encoding="utf-8") as f:
            f.write(final_survey)
