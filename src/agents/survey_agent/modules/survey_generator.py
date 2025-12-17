from utils.rich_logger import get_logger
from utils.api_call import SemanticScholarAPI, ChatAgent
import math
from modules.pe import (
    SURVEY_OUTLINE_GENERATION,
    SUBSECTION_DRAFT,
    SECTION_DRAFT,
    DRAFT_REFINEMENT,
    ERROR_FEEDBACK_PROMPT
)
from utils.utils import extract_json
import textwrap
from tqdm import tqdm

import json
import re

import os

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
                err_info = ""

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
                    keynote_papers.append(paper_id)

                prompt = SURVEY_OUTLINE_GENERATION.format(
                    paper_keynotes=paper_keynotes,
                    current_outline=outline,
                    papers_analysis=papers_analysis,
                )

                valid = False
                for retry_time in range(
                    self.config.ModuleInfo.SurveyGenerator.outline_generation_max_retry_in_generation_loop
                ): ## YZY MODIFY :change retry mode to improve success
                    if self.config.BasicInfo.debug:
                        if len(err_info) > self.config.ModuleInfo.SurveyGenerator.outline_max_error_info_length:
                            self.logger.warning(f"err information exceed max length truncate to {self.config.ModuleInfo.SurveyGenerator.outline_max_error_info_length}, full information:{err_info}")
                    renew_outline = extract_json(
                        self.chat_agent.remote_chat(
                            text_content = prompt if retry_time == 0 else prompt + ERROR_FEEDBACK_PROMPT.format(info = err_info[:self.config.ModuleInfo.SurveyGenerator.outline_max_error_info_length]),
                            temperature=self.config.ModuleInfo.SurveyGenerator.outline_generation_temperature,
                        )
                    )
                    valid, new_err = self.validate_outline(intra_analysis_results, keynote_papers, renew_outline)
                    if valid:
                        outline = renew_outline
                        break
                    else:
                        err_info += new_err
                        self.logger.warning(f"Outline validation failed for batch {batch_idx + 1}. Retrying this batch for {retry_time + 1}...")
                        if self.config.BasicInfo.debug:
                            self.logger.warning(f"Batch cummulative error: {err_info}")
                if not valid:
                    raise ValueError("Invalid paper ID after maximum retries in loop.")

            return outline
        except Exception as e:
            if (
                retry
                > self.config.ModuleInfo.SurveyGenerator.outline_generation_max_retry
            ):
                raise e
            if self.config.BasicInfo.debug:
                self.logger.error(
                    f"Outline generation failed on retry {retry} with error: {e}. Retrying..."
                )
            return self.generate_outline(
                intra_analysis_results,
                inter_analysis_results,
                papers,
                retry=retry + 1,
            )

    def validate_outline(self, intra_analysis_results, keynote_papers, outline):
        valid_papers = set()
        err_info = ""
        valid = True
        for group in intra_analysis_results:
            for q in group:
                valid_papers.update(q["related_papers"])
                if self.config.BasicInfo.debug:
                    self.logger.info(f"Valid paper IDs from intra-analysis: {q['related_papers']}") # YZY DEBUG
        
        #YZY MODIFY: only for debug
        for paper_id in keynote_papers:
            valid_papers.add(paper_id)
            if self.config.BasicInfo.debug:
                self.logger.info(f"Valid paper ID from keynote papers: {paper_id}") # YZY DEBUG

        # if self.config.BasicInfo.debug:
        #     self.logger.info(f"All valid paper ID: {valid_papers}") # YZY DEBUG

        for section in outline.get("sections", []):
            for paper_id in section.get("papers_to_use", []):
                if self.config.BasicInfo.debug:
                    self.logger.info(f"Validating paper ID: {paper_id} in OUTLINE GENERATION") # YZY DEBUG
                if paper_id not in valid_papers:
                    self.logger.error(
                        f"Paper ID {paper_id} in section '{section.get('title', '')}' is not in the valid papers set.\n"
                    )
                    # raise ValueError("Invalid paper ID in outline section.")
                    err_info += f"Paper ID {paper_id} in section '{section.get('title', '')}' is not in the valid papers set."
                    valid = False

            for subsection in section.get("subsections", []):
                for paper_id in subsection.get("papers_to_use", []):
                    if paper_id not in valid_papers:
                        self.logger.error(
                            f"Paper ID {paper_id} in subsection '{subsection.get('title', '')}' is not in the valid papers set."
                        )
                        err_info += f"Paper ID {paper_id} in subsection '{subsection.get('title', '')}' is not in the valid papers set.\n"
                        valid = False
                        # raise ValueError("Invalid paper ID in outline subsection.")
                        
        if not valid:
            return False, err_info
        return True, err_info

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

    def build_prompt_with_truncation(self, template: str, papers_list: list[str], params: dict):
        paper_num = len(papers_list)
        papers = ""

        params_no_papers = dict(params)
        params_no_papers["papers"] = ""
        valid_paper_ids = []

        estimated_prompt_tokens = self.chat_agent.estimate_tokens(template.format(**params_no_papers))

        per_paper_allowd = (self.config.APIInfo.llm_max_context_length 
                        - self.config.ModuleInfo.SurveyGenerator.llm_max_context_overhead_length_generation
                        - estimated_prompt_tokens) // max(paper_num, 1)

        for paper_id in papers_list:
            paper_raw_markdown = self.work_analyzer.get_paper_raw_markdown(
                paper_id
            )
            if paper_raw_markdown.strip() == "Fail to Get Content":
                self.logger.error(f"Failed to get content for paper ID: {paper_id} in SUBSECTION DRAFT. Skipping this paper in subsection draft.")
                continue
            if self.config.BasicInfo.debug:
                self.logger.info(f"Original length of paper {paper_id} raw markdown: {len(paper_raw_markdown)} in SUBSECTION DRAFT")
            paper_raw_markdown = self.chat_agent.truncate_text(paper_id, paper_raw_markdown, per_paper_allowd)
            papers += (
                f"Paper ID: {paper_id}\nRaw markdown: {paper_raw_markdown}\n\n"
            )
            valid_paper_ids.append(paper_id)

        params["papers"] = papers
        prompt = template.format(**params)

        return prompt, valid_paper_ids

    def draft_survey(self, intra_analysis_results, inter_analysis_results, outline):
        relevant_analysis = self.format_papers_analysis(
            intra_analysis_results, inter_analysis_results
        )

        # step 1: subsection draft
        subsection_prompts = []
        subsections_valid_paper_ids = []

        for section in outline.get("sections", []):
            for subsection in section.get("subsections", []):

                params_dict = {
                    "title": subsection.get("title", ""),
                    "description": subsection.get("description", ""),
                    "relevant_analysis": relevant_analysis,
                    "papers": "",
                }

                prompt, valid_paper_ids = self.build_prompt_with_truncation(
                                                                    template = SUBSECTION_DRAFT, 
                                                                    papers_list = subsection.get("papers_to_use", []), 
                                                                    params = params_dict
                                                                )
                subsections_valid_paper_ids.extend(valid_paper_ids)

                subsection_prompts.append(prompt)

        subsection_drafts = [""] * len(subsection_prompts)
        subsection_indices = list(range(len(subsection_prompts)))
        previous_err_infos = [""] * len(subsection_prompts)

        valid = False
        for try_time in range(self.config.ModuleInfo.SurveyGenerator.subsection_draft_max_retry):
            subsection_prompts_with_error = [subsection_prompts[i] + ERROR_FEEDBACK_PROMPT.format(info=previous_err_infos[i][:self.config.ModuleInfo.SurveyGenerator.subsection_draft_max_error_info_length]) 
                                                                        if len(previous_err_infos[i]) > 1 else subsection_prompts[i] for i in range(len(subsection_prompts))]
            response_drafts = self.chat_agent.batch_remote_chat(
                subsection_prompts_with_error,
                desc="Drafting survey subsections...",
                temperature=self.config.ModuleInfo.SurveyGenerator.subsection_draft_temperature,
            )
            err_prompts = []
            err_indices = []
            err_infos = []

            for i, drafts in enumerate(response_drafts):
                is_valid_subsection, info = self.validate_subsection_draft(drafts, subsections_valid_paper_ids)
                if not is_valid_subsection:
                    err_prompts.append(subsection_prompts[i])
                    err_indices.append(subsection_indices[i])
                    previous_err_infos[i] += "Previous Error:" + info + "\n"
                    err_infos.append(previous_err_infos[i])

                    if self.config.BasicInfo.debug:
                        self.logger.info(f'cumulative error info for subsection index {subsection_indices[i]}: {previous_err_infos[i]} in SUBSECTION DRAFT')
                    self.logger.warning(f"Subsection draft validation failed for subsection index {subsection_indices[i]}. Retrying this subsection for {try_time + 1}...")
                else:
                    subsection_drafts[subsection_indices[i]] = drafts
            
            if not err_prompts:
                valid = True
                break  # all valid
            else:
                subsection_prompts = err_prompts
                subsection_indices = err_indices
                if self.config.BasicInfo.debug:
                    self.logger.info(f"Retrying {len(err_prompts)} subsection drafts due to validation errors...")
                for i in range(len(subsection_prompts)):
                    subsection_prompts[i]
                previous_err_infos = err_infos

        if not valid:
            self.logger.error("Some subsection drafts failed validation after maximum retries.")
            raise ValueError("Invalid subsection draft after maximum retries.")

        if self.config.BasicInfo.debug:
            self.logger.info(f"SUBSECTION DRAFTS: {subsection_drafts}")

        # step 2: section draft
        section_prompts = []
        sections_valid_ids = []
        idx = 0
        for section in outline.get("sections", []):
            valid_ids = []
            subsection_drafts = "\n\n".join(
                subsection_drafts[idx : idx + len(section.get("subsections", []))]
            )
            idx += len(section.get("subsections", []))

            subsection_paper_ids = set()
            for subsection in section.get("subsections", []):
                subsection_paper_ids.update(subsection.get("papers_to_use", []))
                sections_valid_ids.extend(subsection.get("papers_to_use", []))

            papers = ""
            section_paper_ids = []
            for paper_id in section.get("papers_to_use", []):
                if paper_id in subsection_paper_ids:
                    continue  # already included in subsections
                section_paper_ids.append(paper_id)
                sections_valid_ids.append(paper_id)
                
            params_dict = {
                "title": section.get("title", ""),
                "description": section.get("description", ""),
                "subsection_drafts": subsection_drafts,
                "papers": "",
            }
            prompt, _ = self.build_prompt_with_truncation(
                template = SECTION_DRAFT,
                papers_list = section_paper_ids,
                params = params_dict,
            )

            section_prompts.append(prompt)
        
        section_drafts = [""] * len(section_prompts)
        section_indices = list(range(len(section_prompts)))
        previous_err_infos = [""] * len(section_prompts)    

        valid = True
        for try_time in range(self.config.ModuleInfo.SurveyGenerator.section_draft_max_retry):
            section_prompts_with_error = [section_prompts[i] + ERROR_FEEDBACK_PROMPT.format(info=previous_err_infos[i][:self.config.ModuleInfo.SurveyGenerator.section_draft_max_error_info_length]) 
                                        if len(previous_err_infos[i]) > 1 else section_prompts[i] for i in range(len(section_prompts))]

            response_drafts = self.chat_agent.batch_remote_chat(
                section_prompts_with_error,
                desc="Drafting survey sections...",
                temperature=self.config.ModuleInfo.SurveyGenerator.section_draft_temperature,
            )

            err_prompts = []
            err_indices = []
            err_infos = []
            for i, drafts in enumerate(response_drafts):
                is_valid_section, info = self.validate_section_draft(drafts, sections_valid_ids)
                if not is_valid_section:
                    self.logger.warning(f"Section draft validation failed for section index {section_indices[i]}. Retrying this section for {try_time + 1}...")
                    err_prompts.append(section_prompts[i])
                    err_indices.append(section_indices[i])
                    previous_err_infos[i] += "Previous Error:" + info + "\n"
                    if self.config.BasicInfo.debug:
                        self.logger.info(f'cumulative error info for section index {section_indices[i]}: {previous_err_infos[i]} in SECTION DRAFT')
                    err_infos.append(previous_err_infos[i])
                    valid =False
                else:
                    section_drafts[section_indices[i]] = drafts
            if not err_prompts:
                valid = True
                break  # all valid
            else:
                section_prompts = err_prompts
                section_indices = err_indices
                previous_err_infos = err_infos
                self.logger.info(f"Retrying {len(err_prompts)} section drafts due to validation errors...")
        if not valid:
            self.logger.error("Some section drafts failed validation after maximum retries.")
            raise ValueError("Invalid section draft after maximum retries.")
            
        if self.config.BasicInfo.debug:
            self.logger.info(f"SECTION DRAFTS: {section_drafts}")

        return (
            outline.get("title", "Untitled Survey")
            + "\n\n"
            + "\n\n".join(section_drafts)
        )
    
    def validate_section_draft(self, section_draft, papers):
            papers_set = set(papers)

            paper_id_pattern = r"\(Paper ID:\s*(\S+)\s*\)|\(Paper ID\s*(\S+)\s*\)|\(Paper\s*(\S+)\s*\)"

            raw = re.findall(paper_id_pattern, section_draft)
            paper_ids = [next((x for x in t if x), "") for t in raw]

            for paper_id in paper_ids:
                if paper_id not in papers_set:
                    self.logger.warning(f"Paper ID {paper_id} not found in papers set in SUBSECTION DRAFT VALIDATE.")
                    return False, f"Paper ID {paper_id} not found in papers set, probably a wrong paper_id."

            return True, ""

    def validate_subsection_draft(self, section_draft, papers):
        papers_set = set(papers)

        paper_id_pattern = r"\(Paper ID:\s*(\S+)\s*\)|\(Paper ID\s*(\S+)\s*\)|\(Paper\s*(\S+)\s*\)"

        raw = re.findall(paper_id_pattern, section_draft)
        paper_ids = [next((x for x in t if x), "") for t in raw]

        for paper_id in paper_ids:
            if paper_id not in papers_set:
                self.logger.warning(f"Paper ID {paper_id} not found in papers set in SUBSECTION DRAFT VALIDATE.")
                return False, f"Paper ID {paper_id} not found in papers set, probably a wrong paper_id."

        return True, ""

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

        return survey + "\n\n" + reference, references

    def save_survey(self, final_survey, references):
        save_path = self.config.BasicInfo.save_path
        save_json_path = self.config.BasicInfo.save_json_path

        # ensure parent dirs exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)

        # write md
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_survey)

        # write json
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump({"paper": final_survey, "references": references}, f, indent=2)
                
