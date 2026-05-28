from utils.rich_logger import get_logger
from utils.api_call import SemanticScholarAPI, ChatAgent
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.pe import (
    SURVEY_OUTLINE_GENERATION,
    SUBSECTION_DRAFT,
    SUBSECTION_DRAFT_WITH_CODE,
    CODE_REPORT_PROMPT,
    SECTION_DRAFT,
    DRAFT_REFINEMENT,
    ERROR_FEEDBACK_PROMPT,
    DRAFT_REFINEMENT_IN_PARTS,
    SURVEY_OUTLINE_GENERATION_PAPER_ASSIGNMENT,
    SURVEY_OUTLINE_GENERATION_OUTLINE_DRAFT,
    SECTION_REVISE,
    SECTION_REVIEW,
    SURVEY_REVIEW,
    SURVEY_REVISE,
    DRAFT_REFINEMENT_SUBSECTION_IN_PARTS,
    CODE_REPORT_PROMPT_FOR_SECTION_REVIEWER,
    CODE_REPORT_PROMPT_FOR_SECTION_REVISER,
    CODE_REPORT_PROMPT_FOR_SURVEY_REVIEWER,
    CODE_REPORT_PROMPT_FOR_SURVEY_REVISER
)
from modules.refine_agent import agentic_revise_survey_whole, agentic_revise_survey_in_parts
from typing import Dict, List, Union
from utils.err_info import CumulativeErrorInfo
from utils.utils import extract_json
import textwrap
from tqdm import tqdm

import json
import re
import copy
import os

class SurveyGenerator:
    def __init__(self, config, work_analyzer, database):
        self.config = config
        self.logger = get_logger("SurveyGenerator")
        self.chat_agent = ChatAgent(config)
        self.work_analyzer = work_analyzer
        self.database = database
        self.use_title_in_draft = config.ModuleInfo.SurveyGenerator.use_title_in_draft
        self.refine_in_parts = config.ModuleInfo.SurveyGenerator.draft_refinement_in_parts
        self.omit_error_preserve_retry_time = config.ModuleInfo.SurveyGenerator.omit_error_preserve_retry_time
        self.include_initial_analysis = config.ModuleInfo.SurveyGenerator.include_initial_analysis
        self.include_relation_graph = config.ModuleInfo.SurveyGenerator.include_relation_graph
        self.include_relation_table = config.ModuleInfo.SurveyGenerator.include_relation_table
        self.refine_in_parts_mode = self.config.ModuleInfo.SurveyGenerator.refine_in_parts_mode
        self.outline_fast_mode = self.config.ModuleInfo.SurveyGenerator.outline_assign_fast_mode
        self.agentic_refine_section = self.config.ModuleInfo.SurveyGenerator.agentic_refine_section
        self.agentic_refine_survey = self.config.ModuleInfo.SurveyGenerator.agentic_refine_survey
        self.always_omit_error = self.config.ModuleInfo.SurveyGenerator.always_omit_error_in_draft_validation

    def format_papers_analysis(self, intra_analysis_results, inter_analysis_results, concise_mode = False):
        papers_analysis = ""

        # debug
        if self.include_relation_graph and not concise_mode:
            papers_analysis += self.work_analyzer.format_analysis_graph()
            self.logger.info("add relation graph in analysis")

        if self.include_relation_table:
            table =  self.work_analyzer.format_analysis_table()
            papers_analysis += table
            # if self.config.BasicInfo.debug:
            #     self.logger.info(f"Relation Table: {table}")
            self.logger.info("add relation table in analysis")

        if self.include_initial_analysis:
            for i, group in enumerate(intra_analysis_results):
                papers_analysis += f"Group {i+1} Analysis:\n"
                for j, q in enumerate(group):
                    papers_analysis += f"Question {j+1}: {q['question']}\nAnswer {j + 1}: {q['answer']}\nRelated Papers {j + 1}: {q['related_papers']}\n\n---\n"

            papers_analysis += (
                f"\n\nHigh-level Inter-Group Analysis:\n{inter_analysis_results}\n"
            )
            papers_analysis = papers_analysis.replace('#', '*')
        

        return papers_analysis

    def format_intra_papers_analysis(self, intra_analysis_results, inter_analysis_results, cluster_index):
        papers_analysis = ""
        group = intra_analysis_results[cluster_index]
        
        papers_analysis += f"Group {cluster_index+1} Analysis:\n"
        for j, q in enumerate(group):
            papers_analysis += f"Question {j+1}: {q['question']}\nAnswer {j + 1}: {q['answer']}\nRelated Papers {j + 1}: {q['related_papers']}\n\n---\n"

        return papers_analysis

    def generate_outline(
        self, intra_analysis_results, inter_analysis_results, papers, retry=1
    ):
        if not self.config.ModuleInfo.SurveyGenerator.outline_generation_in_steps:
            return self.generate_outline_1_step(
                intra_analysis_results, inter_analysis_results, papers, retry
            )
        else:
            return self.generate_outline_in_steps(
                intra_analysis_results, inter_analysis_results, papers, retry
            )

    def generate_outline_in_steps(
        self, intra_analysis_results, inter_analysis_results, papers, retry=1
    ):
        outline = self.generate_outline_draft_outline(
            intra_analysis_results, inter_analysis_results, papers, retry
        )
        outline_with_paper_assignment = self.generate_outline_assign_papers(
            outline, intra_analysis_results, inter_analysis_results, papers, retry
        )
        return outline_with_paper_assignment

    def generate_outline_draft_outline(
        self, intra_analysis_results, inter_analysis_results, papers, retry=1
    ):
        max_retry_in_loop = self.config.ModuleInfo.SurveyGenerator.outline_generation_draft_max_retry
        max_iterations = getattr(self.config.ModuleInfo.SurveyGenerator, 'outline_generation_draft_max_iterations', 10)
        empty_iteration = getattr(self.config.ModuleInfo.SurveyGenerator, 'outline_generation_draft_empty_keynotes_iteration', 1)

        papers_analysis = self.format_papers_analysis(
            intra_analysis_results, inter_analysis_results
        )
        outline = {}

        # iteratively generate outline
        num_batches = math.ceil(
            len(papers)
            / self.config.ModuleInfo.SurveyGenerator.outline_generation_draft_batch_size
        )
        
        # Set max_iterations to num_batches + 1 (for final empty iteration)
        max_iterations = min(num_batches + empty_iteration, max_iterations)
        if max_iterations < num_batches + empty_iteration:
            self.logger.warning(f"max_iterations ({max_iterations}) < num_batches ({num_batches}) + empty_iteration ({empty_iteration}), truncating {num_batches + empty_iteration - max_iterations} batch(es)...")

        # stop debugging prompt
        debug = False
        for iteration_idx in range(max_iterations):
            # Determine if this is the final empty iteration
            is_empty_iteration = (iteration_idx + empty_iteration >= max_iterations)
            
            if is_empty_iteration:
                # Final iteration with empty paper_keynotes
                err_info = CumulativeErrorInfo()
                self.logger.info(
                    f"Generating outline: final refinement iteration (no new papers)"
                )
                paper_keynotes = ""
            else:
                # Normal batch processing
                batch_idx = iteration_idx
                err_info = CumulativeErrorInfo()

                self.logger.info(
                    f"Generating outline: processing batch {batch_idx + 1} of {num_batches} (iteration {iteration_idx + 1}/{max_iterations})"
                )
                batch_papers = papers[
                    batch_idx
                    * self.config.ModuleInfo.SurveyGenerator.outline_generation_draft_batch_size : (
                        batch_idx + 1
                    )
                    * self.config.ModuleInfo.SurveyGenerator.outline_generation_draft_batch_size
                ]
                paper_keynotes = ""
                for paper_id in batch_papers:
                    try:
                        keynote = self.work_analyzer.get_paper_keynote(paper_id)
                        paper_keynotes += f"Paper ID: {paper_id}\nKeynote: {keynote}\n\n"
                    except Exception as e:
                        self.logger.error(f"Failed to get keynote for paper ID: {paper_id} in OUTLINE GENERATION DRAFT STEP with error {e}. Skipping this paper in OUTLINE GENERATION.")
                        continue

            less_RAG_ratio = 1.0

            while True:
                query = f"Current Outline Title: {outline.get('title', '')}\n Current Outline Sections: {json.dumps(outline.get('sections', []))}"
                other_relevant_papers = self.database.query_and_text(query_text = query, 
                                                                    top_k = int(self.config.ModuleInfo.SurveyGenerator.outline_draft_RAG_topk*less_RAG_ratio), 
                                                                    include_paper_id = True) if self.config.ModuleInfo.SurveyGenerator.include_other_relevant_papers_RAG_in_outline else ""
                prompt = SURVEY_OUTLINE_GENERATION_OUTLINE_DRAFT.format(
                    paper_keynotes=paper_keynotes,
                    current_outline=outline,
                    papers_analysis=papers_analysis,
                    other_relevant_papers=other_relevant_papers
                )
                if self.chat_agent.estimate_tokens(prompt) < self.config.APIInfo.llm_max_context_length - self.config.ModuleInfo.SurveyGenerator.llm_max_context_overhead_length_outline_generation:
                    break
                else:
                    less_RAG_ratio = less_RAG_ratio * 0.5

                if less_RAG_ratio < 0.1:
                    self.logger.warning("Cannot fit prompt within context length even after reducing RAG papers. Proceeding with minimal RAG content.")
                    break

            if self.config.BasicInfo.debug and debug:
                self.logger.info(f"Outline draft generation prompt(debug mode, print the first prompt): {prompt}")
                debug = False

            for retry_time in range(
                max_retry_in_loop
            ): 
                valid = False
                try:
                    if self.config.BasicInfo.debug and retry_time > 0:
                        self.logger.info(f"Retry time {retry_time}, with cumulative error info: {err_info.get_errors_text()[:50]} in OUTLINE GENERATION-DRAFT")
                    renew_outline_raw = self.chat_agent.remote_chat(
                        text_content = prompt if retry_time == 0 else prompt + ERROR_FEEDBACK_PROMPT.format(info = err_info.get_errors_text()),
                        temperature=self.config.ModuleInfo.SurveyGenerator.outline_generation_temperature,
                    )
                    renew_outline = extract_json(renew_outline_raw)
                    valid, new_err, err_papers = self.validate_outline(intra_analysis_results, papers, renew_outline, format_only = True)
                    err_info.add_errors(new_err)
                    if not valid:
                        raise ValueError(f"{new_err}")
                except Exception as e:
                    self.logger.warning(f"Outline validation failed for iteration {iteration_idx + 1}: {e} in OUTLINE GENERATION-DRAFT. Retrying this iteration for {retry_time + 1}...")
                    valid = False
                
                if valid:
                    outline = renew_outline
                    break

        return outline


    def _validate_assignment(self, result, info_dict = None):
        error_conservatism_mode = info_dict.get("error_conservatism_mode", False)
        omit_error_preserve_retry_time = info_dict.get("omit_error_preserve_retry_time", 1)
        retry_time = info_dict.get("retry_time", 0)
        max_retry = info_dict.get("max_retry", 5)
        papers = info_dict.get("papers", [])

        omit_err = (retry_time + omit_error_preserve_retry_time >= max_retry and not error_conservatism_mode)

        papers_assignment = extract_json(result)
        err_papers = []

        for paper in papers_assignment:
            # Validate paper_id
            if "paper_id" not in paper:
                if omit_err:
                    continue
                else:
                    raise ValueError(f"Paper Dict lack paper_id key in current provided paper in outline paper assignment")
            
            paper_id = paper.get("paper_id")
            
            # Validate paper_id is in valid papers
            if paper_id not in papers and paper_id not in self.work_analyzer.work_collector.graph_paper_ids:
                if omit_err:
                    err_papers.append(paper_id)
                else:
                    raise ValueError(f"Paper ID {paper_id} not in current provided papers in outline paper assignment")
            
            # Validate assignment key exists and is a dict
            if "assignment" not in paper:
                if omit_err:
                    err_papers.append(paper_id)
                    self.logger.warning(f"Omit Err mode. Paper {paper_id} lacks 'assignment' key, skipping this paper in OUTLINE GENERATION-ASSIGN")
                    continue
                else:
                    raise ValueError(f"Paper {paper_id} lacks 'assignment' key in outline paper assignment")
            
            assignment = paper.get("assignment")
            if not isinstance(assignment, dict):
                if omit_err:
                    err_papers.append(paper_id)
                    self.logger.warning(f"Omit Err mode. Paper {paper_id} 'assignment' is not a dict, skipping this paper in OUTLINE GENERATION-ASSIGN")
                    continue
                else:
                    raise ValueError(f"Paper {paper_id} 'assignment' is not a dict in outline paper assignment")
            
            # Validate assignment values are lists
            for assign_section, assign_subsections in assignment.items():
                if not isinstance(assign_subsections, list):
                    if omit_err:
                        err_papers.append(paper_id)
                        self.logger.warning(f"Omit Err mode. Paper {paper_id} assignment for section '{assign_section}' is not a list, skipping this paper in OUTLINE GENERATION-ASSIGN")
                        continue
                    else:
                        raise ValueError(f"Paper {paper_id} assignment for section '{assign_section}' is not a list in outline paper assignment")

        ## still err in last try, just delete errors rather than retry
        if omit_err and len(err_papers) > 0:
            self.logger.warning(f"max retry reached, deleting error papers{err_papers} directly and returning in OUTLINE GENERATION-ASSIGN")
            papers_assignment = [paper for paper in papers_assignment if paper.get("paper_id", "unknown") not in err_papers]


        result = json.dumps(papers_assignment)
        # Fixed: return (val, result) not (result, val)
        return True, result

    def generate_outline_assign_papers(
        self, outline, intra_analysis_results, inter_analysis_results, papers, retry=1
    ):
        max_retry_in_loop = self.config.ModuleInfo.SurveyGenerator.outline_generation_assign_max_retry
        self.logger.info(f"OUTLINE GENERATION ASSIGNMENT need to assign {len(papers)} papers")
        papers_analysis = self.format_papers_analysis(
            intra_analysis_results, inter_analysis_results
        )

        # step 1: initialize
        for section in outline.get("sections"):
            section["papers_to_use"] = []
            for subsection in section.get("subsections", []):
                subsection["papers_to_use"] = []

        
        outline_with_paper_assignment = copy.deepcopy(outline)

        # step 2: cal batch
        num_batches = math.ceil(
            len(papers)
            / self.config.ModuleInfo.SurveyGenerator.outline_generation_assign_batch_size
        )

        prompts = []
        for batch_idx in range(num_batches):
            err_info = CumulativeErrorInfo()

            self.logger.info(
                f"Generating outline: processing batch {batch_idx + 1} of {num_batches}"
            )
            # step 3: chunk batch
            batch_papers = papers[
                batch_idx
                * self.config.ModuleInfo.SurveyGenerator.outline_generation_assign_batch_size : (
                    batch_idx + 1
                )
                * self.config.ModuleInfo.SurveyGenerator.outline_generation_assign_batch_size
            ]

            # step 4: build prompt
            paper_keynotes = ""
            for paper_id in batch_papers:
                try:
                    keynote = self.work_analyzer.get_paper_keynote(paper_id)
                    title = self.work_analyzer.work_collector.get_paper_title(paper_id)
                    paper_keynotes += f"Paper ID: {paper_id}\nTitle: {title}\nKeynote: {keynote}\n\n"
                except Exception as e:
                    self.logger.error(f"Failed to get keynote for paper ID: {paper_id} in OUTLINE GENERATION ASSIGNMENT with error {e}. Skipping this paper in OUTLINE GENERATION ASSIGNMENT.")

            less_RAG_ratio = 1.0

            while True:
                query = f"Current Outline Title: {outline.get('title', '')}\n Current Outline Sections: {json.dumps(outline.get('sections', []))}"
                other_relevant_papers = self.database.query_and_text(query_text = query, 
                                                                    top_k = int(self.config.ModuleInfo.SurveyGenerator.outline_assign_RAG_topk*less_RAG_ratio), 
                                                                    include_paper_id = True) if self.config.ModuleInfo.SurveyGenerator.include_other_relevant_papers_RAG_in_outline else ""
                prompt = SURVEY_OUTLINE_GENERATION_PAPER_ASSIGNMENT.format(
                    paper_keynotes=paper_keynotes,
                    current_outline=outline,
                    papers_analysis="",
                    other_relevant_papers=other_relevant_papers
                )
                if self.chat_agent.estimate_tokens(prompt) < self.config.APIInfo.llm_max_context_length - self.config.ModuleInfo.SurveyGenerator.llm_max_context_overhead_length_outline_generation:
                    break
                else:
                    less_RAG_ratio = less_RAG_ratio * 0.5

                if less_RAG_ratio < 0.1:
                    self.logger.warning("Cannot fit prompt within context length even after reducing RAG papers. Proceeding with minimal RAG content.")
                    raise ValueError("Prompt too long even after reducing RAG content in OUTLINE GENERATION.")
                    break

            prompts.append(prompt)

            # if self.config.BasicInfo.debug:
            #     self.logger.info(f"Outline generation prompt : {prompt}")

            if self.outline_fast_mode:
                break
            

            valid = False
            # step 5: chat(fast mode will use batch chat out of loop)
            for retry_time in range(
                max_retry_in_loop
            ): 
                omit_err = (retry_time + self.omit_error_preserve_retry_time >= max_retry_in_loop and not self.config.BasicInfo.error_conservatism_mode)
                try:
                    if self.config.BasicInfo.debug and retry_time > 0:
                        self.logger.info(f"Retry time {retry_time} with cumulative error info: {err_info.get_errors_text()} in OUTLINE GENERATION-ASSIGN")

                    paper_assignment_raw = self.chat_agent.remote_chat(
                        text_content = prompt if retry_time == 0 else prompt + ERROR_FEEDBACK_PROMPT.format(info = err_info.get_errors_text()),
                        temperature=self.config.ModuleInfo.SurveyGenerator.outline_generation_temperature,
                    )
                    papers_assignment = extract_json(paper_assignment_raw)

                    err_papers = []
                    # if self.config.BasicInfo.debug:
                    #     self.logger.info(f"Retry_condition: {retry_time + self.omit_error_preserve_retry_time >= max_retry_in_loop}, omit_enable_condition: {not self.config.BasicInfo.error_conservatism_mode}. Omit Err mode: {omit_err} in OUTLINE GENERATION-ASSIGN")


                    for paper in papers_assignment:
                        if "paper_id" not in paper:
                            if omit_err:
                                continue
                            else:
                                raise ValueError(f"Paper Dict lack paper_id key in current provided paper in outline paper assignment")
                        if paper["paper_id"] not in papers and paper["paper_id"] not in self.work_analyzer.work_collector.graph_paper_ids:
                            err_info.add_error(f"Paper ID {paper['paper_id']} not in current provided papers in paper assignment")
                            if omit_err:
                                err_papers.append(paper["paper_id"])
                            else:
                                raise ValueError(f"Paper ID {paper['paper_id']} not in current provided papers in outline paper assignment")

                    ## still err in last try, just delete errors rather than retry
                    if omit_err and len(err_papers) > 0:
                        self.logger.warning(f"max retry reached, deleting error papers{err_papers} directly and returning in OUTLINE GENERATION-ASSIGN")
                        papers_assignment = [paper for paper in papers_assignment if paper.get("paper_id", "unknown") not in err_papers]

                    outline_with_paper_assignment = self._assign_paper(papers_assignment, outline_with_paper_assignment, omit_err = omit_err)
                    
                    valid, new_err, err_papers = self.validate_outline(intra_analysis_results, papers, outline_with_paper_assignment, omit_err = omit_err)
                    err_info.add_errors(new_err)
                    
                    if omit_err:
                        if self.config.BasicInfo.debug:
                            self.logger.info(f"Omit Err mode, proceeding with valid outline even if errors exist in OUTLINE GENERATION-ASSIGN.")
                        valid = True
                    if not valid:
                        raise ValueError(f"{new_err}")
                        
                except Exception as e:
                    if omit_err:
                        valid = True
                    self.logger.warning(f"Outline validation failed for batch {batch_idx + 1}: {e} in OUTLINE GENERATION-ASSIGN. Retrying this batch for {retry_time + 1}...")
                    continue

                if valid:
                    break

            if not valid:
                raise ValueError("Invalid paper ID after maximum retries in loop.")
            
        if self.outline_fast_mode:
            info_dict = {
                "papers": papers,
                "error_conservatism_mode": self.config.BasicInfo.error_conservatism_mode,
                "omit_error_preserve_retry_time": self.omit_error_preserve_retry_time,
                "max_retry": self.config.ModuleInfo.SurveyGenerator.outline_generation_assign_max_retry,
            }
            self.logger.info("[OUTLINE FAST MODE DEBUG]Fast mode enabled, use batch_remote_chat to process outline assignement")
            results = self.chat_agent.batch_remote_chat_with_retry(prompts, 
                                                                   self._validate_assignment, 
                                                                   max_retry = self.config.ModuleInfo.SurveyGenerator.outline_generation_assign_max_retry,
                                                                   desc = "Generating batch chat result for outline assignment",
                                                                   # Fixed: removed negative sign from temperature
                                                                   temperature=self.config.ModuleInfo.SurveyGenerator.outline_generation_temperature,
                                                                   info_dict=info_dict,)

            omit_err = (self.omit_error_preserve_retry_time >= 1 and not self.config.BasicInfo.error_conservatism_mode)
            for result in results:
                papers_assignment = extract_json(result)
                outline_with_paper_assignment = self._assign_paper(papers_assignment, outline_with_paper_assignment, omit_err = omit_err)

        return outline_with_paper_assignment

    def _assign_paper(self, paper_assignment, outline, omit_err = False):
        outline_with_paper_assignment = outline

        for paper in paper_assignment:
            paper_id = paper.get("paper_id")
            for assign_section, assign_subsections in paper.get("assignment").items():
                matched_section = False
                for section in outline_with_paper_assignment.get("sections"):
                    if assign_section == section.get("title"):
                        matched_section = True
                        if paper_id not in section["papers_to_use"]:
                            section["papers_to_use"].append(paper_id)
                        for subsection_title in assign_subsections:
                            matched_subsection = False
                            for subsection in section.get("subsections", []):
                                if subsection_title == subsection.get("title"):
                                    matched_subsection = True
                                    if paper_id not in subsection["papers_to_use"]:
                                        subsection["papers_to_use"].append(paper_id)
                            if not matched_subsection:
                                if omit_err:
                                    self.logger.info(f"Omit Err mode. Subsection title {subsection_title} not found during paper assignment for paper ID {paper_id}, skipping this subsection assignment.")
                                    continue
                                raise ValueError(f"Subsection title {subsection_title} not found during paper assignment.")
                if not matched_section:
                    if omit_err:
                        self.logger.info(f"Omit Err mode. Section title {assign_section} not found during paper assignment for paper ID {paper_id}, skipping this section assignment.")
                        continue
                    raise ValueError(f"Section title {assign_section} not found during paper assignment.")

        return outline_with_paper_assignment

    def generate_outline_1_step(
        self, intra_analysis_results, inter_analysis_results, papers, retry=1
    ):
        max_retry_in_loop = self.config.ModuleInfo.SurveyGenerator.outline_generation_max_retry_in_generation_loop
        max_retry_out_loop = self.config.ModuleInfo.SurveyGenerator.outline_generation_max_retry
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
                err_info = CumulativeErrorInfo()

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
                for paper_id in batch_papers:
                    try:
                        keynote = self.work_analyzer.get_paper_keynote(paper_id)
                        paper_keynotes += f"Paper ID: {paper_id}\nKeynote: {keynote}\n\n"
                    except Exception as e:
                        self.logger.error(f"Failed to get keynote for paper ID: {paper_id} in OUTLINE GENERATION with error {e}. Skipping this paper in OUTLINE GENERATION.")
                        continue

                less_RAG_ratio = 1.0

                while True:
                    query = f"Current Outline Title: {outline.get('title', '')}\n Current Outline Sections: {json.dumps(outline.get('sections', []))}"
                    other_relevant_papers = self.database.query_and_text(query, int(self.config.ModuleInfo.SurveyGenerator.outline_RAG_topk*less_RAG_ratio)) if self.config.ModuleInfo.SurveyGenerator.include_other_relevant_papers_RAG_in_outline else ""
                    prompt = SURVEY_OUTLINE_GENERATION.format(
                        paper_keynotes=paper_keynotes,
                        current_outline=outline,
                        papers_analysis=papers_analysis,
                        other_relevant_papers=other_relevant_papers
                    )
                    if self.chat_agent.estimate_tokens(prompt) < self.config.APIInfo.llm_max_context_length - self.config.ModuleInfo.SurveyGenerator.llm_max_context_overhead_length_outline_generation:
                        break
                    else:
                        less_RAG_ratio = less_RAG_ratio * 0.5

                    if less_RAG_ratio < 0.1:
                        self.logger.warning("Cannot fit prompt within context length even after reducing RAG papers. Proceeding with minimal RAG content.")
                        raise ValueError("Prompt too long even after reducing RAG content in OUTLINE GENERATION.")
                        break



                # if self.config.BasicInfo.debug:
                #     self.logger.info(f"Outline generation prompt : {prompt}")

                valid = False
                for retry_time in range(
                    max_retry_in_loop
                ): 

                    renew_outline_raw = self.chat_agent.remote_chat(
                        text_content = prompt if retry_time == 0 else prompt + ERROR_FEEDBACK_PROMPT.format(info = err_info.get_errors_text()),
                        temperature=self.config.ModuleInfo.SurveyGenerator.outline_generation_temperature,
                    )

                    renew_outline = extract_json(renew_outline_raw)

                    omit_error = (retry_time + self.omit_error_preserve_retry_time >= max_retry_in_loop and retry + 1 == max_retry_out_loop and not self.config.BasicInfo.error_conservatism_mode)
                    
                    valid, new_err, err_papers = self.validate_outline(intra_analysis_results, papers, renew_outline, omit_err = omit_error)
                    if valid:
                        outline = renew_outline
                        break
                    else:
                        # if omit_error:
                        #     self.logger.warning("max retry reached, deleting error papers directly and returning in OUTLINE GENERATION")
                        #     renew_outline = self._remove_papers_from_outline(renew_outline, err_papers)
                        #     valid_after, new_err, _ = self.validate_outline(intra_analysis_results, papers, renew_outline, omit_err = omit_error)
                        #     if not valid_after:
                        #         raise ValueError(f"Invalid outline after removing hallucinations: {new_err}")
                        #     outline = renew_outline
                        #     valid = True
                        #     break

                        err_info.add_errors(new_err)
                        self.logger.warning(f"Outline validation failed for batch {batch_idx + 1}. Retrying this batch for {retry_time + 1}...")
                        if self.config.BasicInfo.debug:
                            self.logger.warning(f"Batch cummulative error: {err_info.get_errors_text()}")
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

    def validate_outline_format(self, outline):
        if not isinstance(outline, dict):
            return False, "Outline must be a dictionary."
        # if "title" not in outline or not isinstance(outline["title"], str):
        #     return False, "Outline must have a 'title' field of type string."
        if "sections" not in outline or not isinstance(outline["sections"], list):
            return False, "Outline must have a 'sections' field of type list."
        for section in outline["sections"]:
            if not isinstance(section, dict):
                return False, f"Each section must be a dictionary, got {type(section)}."
            if "title" not in section or not isinstance(section["title"], str):
                return False, f"Each section must have a 'title' field of type string."
            if "description" not in section or not isinstance(section["description"], str):
                return False, f"Each section must have a 'description' field of type string."
            if "subsections" not in section or not isinstance(section["subsections"], list):
                return False, f"Each section must have a 'subsections' field of type list."
            for subsection in section["subsections"]:
                if not isinstance(subsection, dict):
                    return False, f"Each subsection must be a dictionary, got {type(subsection)}."
                if "title" not in subsection or not isinstance(subsection["title"], str):
                    return False, f"Each subsection must have a 'title' field of type string."
                if "description" not in subsection or not isinstance(subsection["description"], str):
                    return False, f"Each subsection must have a 'description' field of type string."
        return True, ""

    def validate_outline(self, intra_analysis_results, papers, outline, format_only = False, omit_err = False):
        valid_papers = set()
        if self.config.ModuleInfo.SurveyGenerator.include_other_relevant_papers_RAG_in_outline:
            valid_papers.update(self.database.valid_paper_ids)
        err_info = []
        valid = True
        err_papers = []

        for group in intra_analysis_results:
            for q in group:
                valid_papers.update(q["related_papers"])
                # if self.config.BasicInfo.debug:
                #     self.logger.info(f"Valid paper IDs from intra-analysis: {q['related_papers']}") # YZY DEBUG
        
        for paper_id in papers:
            valid_papers.add(paper_id)
        
        valid_format, err_format =  self.validate_outline_format(outline)

        if not valid_format:
            err_info.append(f"Outline format error: {err_format}\n")
            valid = False
            return valid, err_info, err_papers
        
        if format_only:
            return valid, err_info, err_papers

        paper_set = set()
        for section in outline.get("sections", []):
            paper_set.update(section.get("papers_to_use", []))

            err_papers_to_remove = []
            for paper_id in section.get("papers_to_use", []):
                # if self.config.BasicInfo.debug:
                #     self.logger.info(f"Validating paper ID: {paper_id} in OUTLINE GENERATION") # YZY DEBUG
                if paper_id not in valid_papers:
                    self.logger.error(
                        f"Paper ID {paper_id} in section '{section.get('title', '')}' is not in the valid papers set.\n"
                    )
                    if omit_err:
                        err_papers_to_remove.append(paper_id)
                        self.logger.info(f"Omit Err mode. Removed invalid paper ID {paper_id} from section '{section.get('title', '')}' due to omit_err=True.")
                    else:
                        err_papers.append(paper_id)
                        # raise ValueError("Invalid paper ID in outline section.")
                        err_info.append(f"Paper ID {paper_id} in section '{section.get('title', '')}' is not in the valid papers set.")
                        valid = False
            section["papers_to_use"] = [pid for pid in section.get("papers_to_use", []) if pid not in err_papers_to_remove]

            for subsection in section.get("subsections", []):
                paper_set.update(subsection.get("papers_to_use", []))

                err_papers_to_remove = []
                for paper_id in subsection.get("papers_to_use", []):
                    if paper_id not in valid_papers:
                        self.logger.error(
                            f"Paper ID {paper_id} in subsection '{subsection.get('title', '')}' is not in the valid papers set."
                        )
                        if omit_err:
                            err_papers_to_remove.append(paper_id)
                            self.logger.info(f"Omit Err mode. Removed invalid paper ID {paper_id} from subsection '{subsection.get('title', '')}' due to omit_err=True.")
                        else:
                            err_info.append(f"Paper ID {paper_id} in subsection '{subsection.get('title', '')}' is not in the valid papers set.\n")
                            err_papers.append(paper_id)
                            valid = False
                        # raise ValueError("Invalid paper ID in outline subsection.")
                subsection["papers_to_use"] = [pid for pid in subsection.get("papers_to_use", []) if pid not in err_papers_to_remove]
                        
        self.logger.info(f"A {valid} batch, reference paper num: {len(paper_set)}")
        if self.config.BasicInfo.debug:
            self.logger.info(f"Use papers: {paper_set}")

        if not valid:
            return False, err_info, err_papers

        return True, err_info, []

    def log_outline(
        self, outline, width=100, max_papers_display=100, desc_preview_len=None
    ):
        used_paper_ids = set()
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
                used_paper_ids.update(papers)

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
        self.logger.info(f"Total unique papers used in outline: {len(used_paper_ids)}")

    def build_prompt_with_truncation(self, template: str, papers_list: list[str], params: dict):
        paper_num = len(papers_list)
        papers = ""
        use_full_text = self.config.ModuleInfo.SurveyGenerator.use_full_text_in_survey_generation

        params_no_papers = dict(params)
        params_no_papers["papers"] = ""
        valid_paper_ids = []

        estimated_prompt_tokens = self.chat_agent.estimate_tokens(template.format(**params_no_papers))

        per_paper_allowd = (self.config.APIInfo.llm_max_context_length 
                        - self.config.ModuleInfo.SurveyGenerator.llm_max_context_overhead_length_generation
                        - estimated_prompt_tokens) // max(paper_num, 1)

        for paper_id in papers_list:
            try:
                title, abstract = self.work_analyzer.work_collector.get_paper_title_abstract(paper_id)
            except Exception as e:
                title, abstract = "", ""

            if use_full_text:
                try:
                    paper_raw_markdown = self.work_analyzer.work_collector.get_paper_raw_markdown(
                        paper_id
                    )
                except Exception as e:
                    if self.config.ModuleInfo.WorkAnalyzer.abstract_when_full_text_fail:
                        self.logger.info(f"Full text fetch failed for paper ID: {paper_id}: {e} in SURVEY DRAFT, using abstract instead.")
                        try:
                            paper_raw_markdown, _ = self.work_analyzer.work_collector.get_paper_title_abstract(
                                paper_id
                            )
                            paper_raw_markdown = str(paper_raw_markdown)
                        except Exception as e:
                            self.logger.error(f"Failed to get abstract for paper ID: {paper_id} in SURVEY DRAFT with error {e}. Skipping this paper in SURVEY DRAFT.")
                            continue
                    else:
                        self.logger.error(f"Failed to get content for paper ID: {paper_id} in SURVEY DRAFT. Skipping this paper in SURVEY DRAFT.")
                        continue
                if self.config.BasicInfo.debug:
                    self.logger.info(f"Original length of paper {paper_id} raw markdown: {len(paper_raw_markdown)} in SURVEY DRAFT")
                paper_raw_markdown = self.chat_agent.truncate_text(paper_id, paper_raw_markdown, per_paper_allowd)

                if self.use_title_in_draft:
                    self.logger.info(f"Using title for paper ID: {paper_id} in SURVEY DRAFT")
                    papers += (
                        f"Title: {title}\nRaw markdown: {paper_raw_markdown}\n\n"
                    )
                else:
                    papers += (
                        f"Paper ID: {paper_id}\nRaw markdown: {paper_raw_markdown}\n\n"
                    )
                valid_paper_ids.append(paper_id)

            else:
                if self.config.BasicInfo.debug:
                    self.logger.info(f"Using keynote for paper ID: {paper_id} in SURVEY DRAFT")
                try:
                    paper_keynote = self.work_analyzer.get_paper_keynote(
                        paper_id
                    )
                    paper_keynote = str(paper_keynote)
                except Exception as e:
                    self.logger.error(f"Failed to get keynote for paper ID: {paper_id} in SURVEY DRAFT with error {e}. Skipping this paper in SURVEY DRAFT.")
                    continue
                if self.config.BasicInfo.debug:
                    self.logger.info(f"Original length of paper {paper_id} paper keynote: {len(paper_keynote)} in SURVEY DRAFT")
                paper_keynote = self.chat_agent.truncate_text(paper_id, paper_keynote, per_paper_allowd)
                if self.use_title_in_draft:
                    papers += f"Title: {title}\nKeynote: {paper_keynote}\n\n"
                else:
                    papers += f"Paper ID: {paper_id}\nKeynote: {paper_keynote}\n\n"
                valid_paper_ids.append(paper_id)

        params["papers"] = papers
        prompt = template.format(**params)

        if self.config.BasicInfo.debug:
            self.logger.info(f"Built prompt with length {len(prompt)}")

        return prompt, valid_paper_ids

    def draft_survey(self, intra_analysis_results, inter_analysis_results, outline, code_report=None):
        relevant_analysis = self.format_papers_analysis(
            intra_analysis_results, inter_analysis_results
        )

        # Determine which template to use based on whether code_report is provided
        use_code_template = code_report is not None
        if use_code_template:
            self.logger.info("Code report provided. Using code-aware prompt template for subsection drafting.")
            code_report_prompt = CODE_REPORT_PROMPT.format(code_report=code_report)
        else:
            code_report_prompt = ""
            self.logger.info("No code report provided. Using standard prompt template for subsection drafting.")

        # Minimal word count for any draft validation (0 means disabled).

        # step 1: subsection draft
        subsection_prompts = []
        subsections_valid_paper_ids = []
        subsection_least_words = self.config.ModuleInfo.SurveyGenerator.subsection_least_words or "no limit"
        subsection_least_citations = self.config.ModuleInfo.SurveyGenerator.subsection_least_citations or "no limit"
        

        for section_index, section in enumerate(outline.get("sections", [])):
            for subsection_index, subsection in enumerate(section.get("subsections", [])):
                
                query = f"Title: {subsection.get('title', '')}\n Description: {subsection.get('description', '')}"
                other_paper_RAG_text = self.database.query_and_text(query, self.config.ModuleInfo.SurveyGenerator.subsection_RAG_topk) if self.config.ModuleInfo.SurveyGenerator.include_other_relevant_papers_RAG else ""
                params_dict = {
                    "title": subsection.get("title", ""),
                    "description": subsection.get("description", ""),
                    "relevant_analysis": relevant_analysis,
                    "papers": "",
                    "other_relevant_papers": other_paper_RAG_text,
                    "subsection_least_words": subsection_least_words,
                    "subsection_least_citations": subsection_least_citations,
                    "survey_outline": json.dumps(outline, ensure_ascii=False),
                    "code_report_prompt": code_report_prompt,
                    # "section_index": section_index + 1,
                    # "subsection_index": subsection_index + 1
                }

                # Use code-aware template if code_report is provided, otherwise use standard template
                template = SUBSECTION_DRAFT_WITH_CODE if use_code_template else SUBSECTION_DRAFT
                
                prompt, valid_paper_ids = self.build_prompt_with_truncation(
                                                                    template = template, 
                                                                    papers_list = subsection.get("papers_to_use", []), 
                                                                    params = params_dict
                                                                )
                subsections_valid_paper_ids.extend(valid_paper_ids)

                subsection_prompts.append(prompt)

                # self.logger.info(f"OUTLINEDEBUG: Built subsection draft prompt for section {section_index + 1} subsection {subsection_index + 1}: \n\n {prompt}")

        # if self.config.BasicInfo.debug:
        #     self.logger.info(f"subsection prompt example: {subsection_prompts[0] if subsection_prompts else 'No prompts generated'}")

        subsection_drafts = [""] * len(subsection_prompts)
        subsection_indices = list(range(len(subsection_prompts)))
        previous_err_infos = [CumulativeErrorInfo() for _ in range(len(subsection_prompts))]

        valid = False
        for try_time in range(self.config.ModuleInfo.SurveyGenerator.subsection_draft_max_retry):
            subsection_prompts_with_error = [subsection_prompts[i] + ERROR_FEEDBACK_PROMPT.format(info=previous_err_infos[i].get_errors_text()) 
                                                                        if len(previous_err_infos[i].get_errors_text()) > 1 else subsection_prompts[i] for i in range(len(subsection_prompts))]
            try:    
                response_drafts = self.chat_agent.batch_remote_chat(
                    subsection_prompts_with_error,
                    desc="Drafting survey subsections...",
                    temperature=self.config.ModuleInfo.SurveyGenerator.subsection_draft_temperature,
                )
            except Exception as e:
                self.logger.error(f"Failed to get subsection drafts from chat agent: {e} in SUBSECTION DRAFT. Retrying all subsections for {try_time + 1}...")
                continue
            err_prompts = []
            err_indices = []
            err_infos = []

            for i, drafts in enumerate(response_drafts):
                allow_omit = (not self.config.BasicInfo.error_conservatism_mode) and (try_time + self.omit_error_preserve_retry_time >= self.config.ModuleInfo.SurveyGenerator.subsection_draft_max_retry)
                if self.use_title_in_draft:
                    is_valid_subsection, info, cleaned_draft = self.validate_title_citation_draft(
                        drafts, subsections_valid_paper_ids, self.config.ModuleInfo.SurveyGenerator.subsection_least_words, omit_error=allow_omit
                    )
                else:
                    is_valid_subsection, info, cleaned_draft = self.validate_id_citation_draft(
                        drafts, subsections_valid_paper_ids, self.config.ModuleInfo.SurveyGenerator.subsection_least_words, omit_error=allow_omit
                    )

                if not is_valid_subsection:
                    err_prompts.append(subsection_prompts[i])
                    err_indices.append(subsection_indices[i])
                    previous_err_infos[i].add_errors(info)
                    err_infos.append(previous_err_infos[i])

                    if self.config.BasicInfo.debug:
                        self.logger.info(f'cumulative error info for subsection index {subsection_indices[i]}: {previous_err_infos[i].get_errors_text()} in SUBSECTION DRAFT')
                    self.logger.warning(f"Subsection draft validation failed for subsection index {subsection_indices[i]}. Retrying this subsection for {try_time + 1}...")
                else:
                    subsection_drafts[subsection_indices[i]] = cleaned_draft
            
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
            # self.logger.info(f"SUBSECTION DRAFTS: {subsection_drafts}")
            total_cites = 0
            for i, subsection_draft in enumerate(subsection_drafts):
                cites = self.count_unique_titles(subsection_draft)
                total_cites += cites
                self.logger.info(f"Subsection {i} citation num: {cites}")
            self.logger.info(f"Total citation num in subsection drafts: {total_cites}")

        # step 2: section draft
        section_preamble_least_words = self.config.ModuleInfo.SurveyGenerator.section_preamble_least_words or "no limit"
        section_preamble_least_citations = self.config.ModuleInfo.SurveyGenerator.section_preamble_least_citations or "no limit"
        
        section_prompts = []
        sections_valid_ids = []
        sections_subsection_drafts = []
        idx = 0
        for section_index, section in enumerate(outline.get("sections", [])):
            valid_ids = []
            current_subsection_drafts = "\n\n"
            for i in range(len(section.get("subsections", []))):
                current_subsection_drafts += "#### " +section.get("subsections", [])[i].get("title", "") + "\n\n" + subsection_drafts[idx + i] + "\n\n"
            sections_subsection_drafts.append(current_subsection_drafts)
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
            
            query = f"Title: {section.get('title', '')}\n Description: {section.get('description', '')}"
            other_paper_RAG_text = self.database.query_and_text(query, self.config.ModuleInfo.SurveyGenerator.section_RAG_topk) if self.config.ModuleInfo.SurveyGenerator.include_other_relevant_papers_RAG else ""
            params_dict = {
                "title": section.get("title", ""),
                "description": section.get("description", ""),
                "subsection_drafts": current_subsection_drafts,
                "papers": "",
                "other_relevant_papers": other_paper_RAG_text,
                "section_least_words": section_preamble_least_words,
                "section_least_citations": section_preamble_least_citations,
                # "section_index": section_index + 1,
                "survey_outline": json.dumps(outline, ensure_ascii=False),
            }
            prompt, _ = self.build_prompt_with_truncation(
                template = SECTION_DRAFT,
                papers_list = section_paper_ids,
                params = params_dict,
            )
            # self.logger.info(f"OUTLINEDEBUG: Section draft prompt for section index {section_index} before error feedback: \n\n {prompt}")

            section_prompts.append(prompt)

        # if self.config.BasicInfo.debug:
        #     self.logger.info(f"section prompt example: {section_prompts[0] if section_prompts else 'No prompts generated'}")
        
        section_drafts = [""] * len(section_prompts)
        section_indices = list(range(len(section_prompts)))
        previous_err_infos = [CumulativeErrorInfo()] * len(section_prompts) 

        valid = True
        self.logger.info(f"Starting section draft with max_retry: {self.config.ModuleInfo.SurveyGenerator.section_draft_max_retry}, error_conservatism_mode: {self.config.BasicInfo.error_conservatism_mode}")
        for try_time in range(self.config.ModuleInfo.SurveyGenerator.section_draft_max_retry):
            section_prompts_with_error = [section_prompts[i] + ERROR_FEEDBACK_PROMPT.format(info=previous_err_infos[i].get_errors_text()) 
                                        if len(previous_err_infos[i].get_errors_text()) > 1 else section_prompts[i] for i in range(len(section_prompts))]

            try:
                response_drafts = self.chat_agent.batch_remote_chat(
                    section_prompts_with_error,
                    desc="Drafting survey sections...",
                    temperature=self.config.ModuleInfo.SurveyGenerator.section_draft_temperature,
                )
            except Exception as e:
                self.logger.error(f"Failed to get section drafts from chat agent: {e} in SECTION DRAFT. Retrying all sections for {try_time + 1}...")
                continue

            err_prompts = []
            err_indices = []
            err_infos = []
            for i, drafts in enumerate(response_drafts):
                allow_omit = (not self.config.BasicInfo.error_conservatism_mode) and (try_time + self.omit_error_preserve_retry_time >= self.config.ModuleInfo.SurveyGenerator.section_draft_max_retry)
                if self.use_title_in_draft:
                    is_valid_section, info, cleaned_draft = self.validate_title_citation_draft(
                        drafts, sections_valid_ids, self.config.ModuleInfo.SurveyGenerator.section_preamble_least_words, omit_error=allow_omit
                    )
                else:
                    is_valid_section, info,  cleaned_draft = self.validate_id_citation_draft(
                        drafts, sections_valid_ids, self.config.ModuleInfo.SurveyGenerator.section_preamble_least_words, omit_error=allow_omit
                    )

                if not is_valid_section:
                    self.logger.warning(f"Section draft validation failed for section index {section_indices[i]}. Retrying this section for {try_time + 1}...")
                    err_prompts.append(section_prompts[i])
                    err_indices.append(section_indices[i])
                    previous_err_infos[i].add_errors(info)
                    if self.config.BasicInfo.debug:
                        self.logger.info(f'cumulative error info for section index {section_indices[i]}: {previous_err_infos[i].get_errors_text()} in SECTION DRAFT')
                    err_infos.append(previous_err_infos[i])
                    valid =False
                else:
                    section_drafts[section_indices[i]] = cleaned_draft + sections_subsection_drafts[section_indices[i]]
                    self.logger.info(f"OUTLINEDEBUG: Section draft for index {section_indices[i]} valid, and updated with subsection drafts.")

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
            
        # if self.config.BasicInfo.debug:
        #     self.logger.info(f"SECTION DRAFTS: {section_drafts}")

        outcome_draft = outline.get("title", self.config.BasicInfo.topic + " Survey")+ "\n\n"+ "\n\n".join(section_drafts)
        self.logger.info(f"OUTLINEDEBUG: Full draft before cleaning and refine: \n\n {outcome_draft}")

        if self.use_title_in_draft:
            self.logger.info(f"Total unique paper titles in correct format in Draft before cleaning and refine: {self.count_unique_titles(outcome_draft)}")
        else:
            self.logger.info(f"Total paper references in correct format in Draft before cleaning: {self.count_unique_paper_ids(outcome_draft)}")

        drafts = {
            "section_drafts": section_drafts,
            "full_draft": outcome_draft,
            "title": outline.get("title", self.config.BasicInfo.topic + " Survey"),
            "outline": outline
        }
        return drafts
    
    def validate_id_citation_draft(self, section_draft, papers, least_words=0, omit_error=False):
        if self.always_omit_error:
            omit_error = True
        if not isinstance(section_draft, str):
            return False, [f"draft context is {type(section_draft)}, not str"], section_draft

        valid = True
        citation_valid = True
        err_info = []

        if least_words and len(section_draft.split()) < least_words*self.config.ModuleInfo.SurveyGenerator.draft_length_relax_ratio:
            self.logger.info(
                f"Section draft too short: {len(section_draft.split())} words < {least_words}."
            )
            if omit_error:
                self.logger.info("Omitting draft length being too short error...")
            else:
                valid = False
                err_info.append(f"Section draft too short: {len(section_draft.split())} words < {least_words}.")

        papers_set = set(papers)
        if self.config.ModuleInfo.SurveyGenerator.include_other_relevant_papers_RAG:
            papers_set.update(self.database.valid_paper_ids)
        err_papers = []
        paper_ids = list(self.get_unique_paper_ids_from_raw(section_draft))

        if '#' in section_draft:
            valid = False
            err_info.append("Draft contains '#' character, probably a wrong format.")
            self.logger.error(f"Draft contains '#' character, probably a wrong format in SUBSECTION DRAFT.")
            if omit_error:
                self.logger.info("Omitting draft containing '#' character error...")
                section_draft = section_draft.replace('#', '')

        for paper_id in paper_ids:
            if paper_id not in papers_set:
                self.logger.warning(f"Paper ID {paper_id} not found in papers set in SUBSECTION DRAFT VALIDATE.")
                err_info.append(f"Paper ID {paper_id} not found in papers set, probably a wrong paper_id.")
                valid = False
                citation_valid = False
                err_papers.append(paper_id)

        cleaned = section_draft
        if not valid and omit_error:
            if not citation_valid:
                cleaned = self._remove_err_paper_ids_from_text(section_draft, err_papers)
            valid = True

        return valid, err_info, cleaned

    def validate_title_citation_draft(self, section_draft, papers, least_words=0, omit_error=False):
        if self.always_omit_error:
            omit_error = True
        if self.config.BasicInfo.debug:
            self.logger.info(f"Validating draft titles in DRAFT TITLE VALIDATE.")
        if not isinstance(section_draft, str):
            return False, [f"draft context is {type(section_draft)}, not str"], section_draft

        citation_valid = True
        err_info = []
        if least_words and len(section_draft.split()) < least_words*self.config.ModuleInfo.SurveyGenerator.draft_length_relax_ratio:
            self.logger.info(
                f"Draft too short for title validation: {len(section_draft.split())} words < {least_words}."
            )
            err_info.append(f"Draft too short for title validation: {len(section_draft.split())} words < {least_words}.")

        papers_set = set(papers)
        if self.config.ModuleInfo.SurveyGenerator.include_other_relevant_papers_RAG:
            papers_set.update(self.database.valid_paper_ids)

        valid, paper_ids, titles, err_titles = self.extract_and_validate_titles_in_text(section_draft)
        if not valid:
            citation_valid = False

        for err_paper_title in err_titles:
            err_info.append(f"Paper title '{err_paper_title}' not found in database, probably a wrong or incomplete title, or the paper is not in valid citation range.\n")

        if '#' in section_draft:
            valid = False
            err_info.append("Draft contains '#' character, probably a wrong format.")
            self.logger.error(f"Draft contains '#' character, probably a wrong format in SUBSECTION DRAFT.")
            if omit_error:
                self.logger.info("Omitting draft containing '#' character error...")
                section_draft = section_draft.replace('#', '')

        cleaned = section_draft
        if not valid and omit_error:
            if not citation_valid:
                cleaned = self._remove_err_paper_titles_from_text(section_draft, err_titles)
            valid = True

        return valid, err_info, cleaned

    def _remove_err_paper_ids_from_text(self, text: str, err_ids: list[str]) -> str:
        cleaned = text
        for pid in err_ids:
            pat = rf"<Paper ID:\s*{re.escape(pid)}\s*>|\(Paper ID:\s*{re.escape(pid)}\s*\)|<Paper\s*ID\s*:\s*{re.escape(pid)}\s*>|\(Paper\s*ID\s*{re.escape(pid)}\s*\)|<Paper\s*{re.escape(pid)}\s*>|\(Paper\s*{re.escape(pid)}\s*\)|{re.escape(pid)}|<Paper\s*<{re.escape(pid)}>\s*>"
            cleaned = re.sub(pat, "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _remove_err_paper_titles_from_text(self, text: str, err_titles: list[str]) -> str:
        cleaned = text
        for title in err_titles:
            # Remove the bracketed title citation, and any standalone title leftovers
            pat = rf"<\s*{re.escape(title)}\s*>|{re.escape(title)}"
            cleaned = re.sub(pat, "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _remove_papers_from_outline(self, outline: dict, err_ids: list[str]) -> dict:
        err_set = set(err_ids)
        for section in outline.get("sections", []) or []:
            section["papers_to_use"] = [p for p in section.get("papers_to_use", []) if p not in err_set]
            for subsection in section.get("subsections", []) or []:
                subsection["papers_to_use"] = [p for p in subsection.get("papers_to_use", []) if p not in err_set]
        return outline

    def review_section(self, section_text, previous_section_text=None, next_section_text=None, section_title = None, section_outline = None, code_report = None, env_report = None):
        self.logger.info("\n--- [Reviewer] analyzing text... ---")
        
        if self.config.ModuleInfo.SurveyGenerator.include_env_report and env_report:
            final_report = f"CODE REPORT: {code_report}\n\nENV REPORT: {env_report}\n\n"
        else:
            final_report = code_report

        if self.config.ModuleInfo.SurveyGenerator.include_code_report and final_report:
            additional_context = CODE_REPORT_PROMPT_FOR_SECTION_REVIEWER.format(code_report = final_report)
            self.logger.info(f"injecting code report to section reviewer: {additional_context[:1500]}")
        else:
            additional_context = "None"

        review_prompt = SECTION_REVIEW.format(
            topic = self.config.BasicInfo.topic,
            section_text=section_text,
            section_least_words = self.config.ModuleInfo.SurveyGenerator.section_least_words or "no limit",
            previous_section_text=previous_section_text or "",
            next_section_text=next_section_text or "",
            current_section_length = len(section_text.split()),
            section_title = section_title or "",
            section_outline = section_outline or "",
            additional_context = additional_context,
        )
        valid = False
        for _ in range(self.config.ModuleInfo.SurveyGenerator.section_review_retry):
            try:
                suggestions = extract_json(
                    self.chat_agent.remote_chat(
                        review_prompt,
                        temperature=self.config.ModuleInfo.SurveyGenerator.section_review_temperature,
                    )
                )
                valid = True
                break
            except Exception as e:
                self.logger.error(f"Section review failed with error {e}. Retrying...")
                continue

        if valid:
            print(f"   [Reviewer] Generated {len(suggestions)} suggestions.")
            return suggestions
        else:
            print("   [Reviewer] Failed to generate valid list.")
            raise ValueError("Invalid section review output.")

    def _apply_revision_to_text(self, original_text: str, revision_json: dict) -> str:
        """
        apply revision to original text based on revision json
        """
        if revision_json.get("action") == "done":
            return original_text

        old_str = revision_json.get("originalText", "")
        new_str = revision_json.get("newText", "")

        if not old_str:
            self.logger.warning("'originalText' is empty or do not have the key.")
            return original_text

        idx = original_text.find(old_str)
        if idx == -1:
            self.logger.error(f"Could not find exact substring to replace.\nTarget: {old_str[:50]}...")
            # TODO: fuzy match?
            return original_text
        
        new_text = original_text[:idx] + new_str + original_text[idx + len(old_str):]
        if self.config.BasicInfo.debug:
            self.logger.info(f"OUTLINEREVISEDEBUG: text to be replaced: {old_str}\n")
            self.logger.info(f"OUTLINEREVISEDEBUG: new text: {new_str}\n")
            self.logger.info(f"   >>> Applied revision: Replaced {len(old_str)} chars with {len(new_str)} chars.")
        return new_text

    def revise_section(self, section_text, suggestion, section_title = None, section_outline = None, code_report = None, env_report = None):
        self.logger.info(f"\n--- [Reviser] processing suggestion: {suggestion}... ---")
        # TODO: refine the query text
        valid = False
        for _ in range(self.config.ModuleInfo.SurveyGenerator.section_revise_retry):
            try:
                if self.config.ModuleInfo.SurveyGenerator.include_env_report and env_report:
                    final_report = f"CODE REPORT: {code_report}\n\nENV REPORT: {env_report}\n\n"
                else:
                    final_report = code_report

                if self.config.ModuleInfo.SurveyGenerator.include_code_report and final_report:
                    additional_context = CODE_REPORT_PROMPT_FOR_SECTION_REVISER.format(code_report = final_report)
                    self.logger.info(f"injecting code report to section revieser: {additional_context[:1500]}")
                else:
                    additional_context = "None"
                prompt = SECTION_REVISE.format(
                    section_title=section_title,
                    section_outline = section_outline or "",
                    topic=self.config.BasicInfo.topic, 
                    text=section_text,
                    citations=self.database.query_and_text(section_title or self.config.BasicInfo.topic, self.config.ModuleInfo.SurveyGenerator.section_revision_RAG_topk),
                    additional_context = additional_context,
                    reviewer_suggestion=suggestion
                )

                parsed_json = extract_json(
                    self.chat_agent.remote_chat(
                        prompt,
                        temperature=self.config.ModuleInfo.SurveyGenerator.section_revise_temperature,
                    )
                )
                if not isinstance(parsed_json, dict):
                    raise ValueError("Parsed JSON is not a dict.")
                if "###" in parsed_json.get("newText", "") or "###" in parsed_json.get("originalText", ""):
                    self.logger.warning("OUTLINEREVISEDEBUG: Revision contains title: '###', which may break markdown structure. This is not allowed.\n")
                    self.logger.info(f"orginal_text:{parsed_json.get('originalText', '')}\n")
                    self.logger.info(f"new_text:{parsed_json.get('newText', '')}\n")
                    raise ValueError("Revision contains '###' which may break markdown structure.")
                valid = True
                break
            except Exception as e:
                self.logger.error(f"Section revision attempt failed with error {e}. Retrying...")
        
        if not valid:
            self.logger.error("Failed to generate valid revision after retries.")
            raise ValueError("Invalid section revision output.")

        if isinstance(parsed_json, dict):
            if parsed_json.get("action") == "done":
                if self.config.BasicInfo.debug:
                    self.logger.info("   [Reviser] Decided no change needed.")
                return section_text
            elif parsed_json.get("action") == "replace":
                if self.config.BasicInfo.debug:
                    self.logger.info("   [Reviser] Applying revision...")
                return self._apply_revision_to_text(section_text, parsed_json)
        else:
            raise ValueError("Parsed revision JSON is not a dict.")
        return section_text
    
    def _format_section_outline(self, section_outline, include_description = True):
        text = f"- Section title: {section_outline.get('title', '')}\n"
        if include_description:
            text += f"  Section description: {section_outline.get('description', '')}\n"
        for subsection in section_outline.get("subsections", []):
            text += f"  - Subsection title: {subsection.get('title', '')}\n"
            if include_description:
                text += f"    Subsection description: {subsection.get('description', '')}\n"
        return text

    def review_and_revise_section(self, current_text, previous_section_text=None, next_section_text=None, section_outline=None, code_report = None, env_report = None):
        MAX_OUTER_ITERATIONS = self.config.ModuleInfo.SurveyGenerator.max_review_revise_iterations
        for i in range(MAX_OUTER_ITERATIONS):
            self.logger.info(f"\n\n====== step 1.1 revise in parts: OUTER LOOP {i+1}/{MAX_OUTER_ITERATIONS} ======")
            
            try:
                suggestions = self.review_section(current_text, previous_section_text, next_section_text, section_outline.get("title", ""), self._format_section_outline(section_outline), code_report = code_report, env_report = env_report)
            except Exception as e:
                self.logger.error(f"Section review failed in OUTER LOOP {i+1} with error {e}. Exiting review and revise loop.")
                continue
            
            if not suggestions or len(suggestions) == 0:
                if (isinstance(suggestions, list) and len(suggestions) == 1 and suggestions[0].lower() == "done"):
                    self.logger.info("Reviewer indicates completion. Exiting.")
                    break
                elif isinstance(suggestions, str) and suggestions.lower() == "done":
                    self.logger.info("Reviewer indicates completion. Exiting.")
                    break
                else:
                    self.logger.info(f"Unexpected reviewer output type: {type(suggestions)} or empty suggestions.")
                    self.logger.info("No suggestions generated. Exiting.")
                    break

            self.logger.info(f"Starting to apply {len(suggestions)} suggestions...")
            for idx, sug in enumerate(suggestions):
                try:
                    new_text = self.revise_section(current_text, sug, section_title=section_outline.get("title", ""), section_outline = self._format_section_outline(section_outline, False), code_report = code_report, env_report = env_report)
                except Exception as e:
                    self.logger.error(f"Section revision failed for suggestion {idx+1} in OUTER LOOP {i+1} with error {e}. Skipping this suggestion.")
                    continue
                
                if new_text != current_text:
                    current_text = new_text
                else:
                    self.logger.info(f"   (Suggestion {idx+1} resulted in no change)")

            for _ in range(self.config.ModuleInfo.SurveyGenerator.no_suggestion_run_each_iteration):
                self.logger.info(f"\n--- step 1.2 NO SUGGESTION LOOP ---")
                try:
                    new_text = self.revise_section(current_text, "", section_title=section_outline.get("title", ""), code_report = code_report, env_report = env_report)
                except Exception as e:
                    self.logger.error(f"Section revision failed for suggestion {idx+1} in OUTER LOOP empty-suggest-modify with error {e}. Skipping this suggestion.")
                    continue
            
            self.logger.info(f"\n--- LOOP {i+1} REVISED FINISH --- \n")
            if self.config.BasicInfo.debug and (i + 1) % 5 == 0:
                self.logger.info(current_text)
            self.logger.info(f"\n--- End of OUTER LOOP {i+1} --- \n")
        if self.config.BasicInfo.debug:
            self.logger.info("\n\n=== Final Result ===")
            self.logger.info(current_text[:30])
        return current_text

    def review_and_revise_survey_in_parts(self, draft, outline, code_report = None, env_report = None):
        if not self.config.ModuleInfo.SurveyGenerator.enable_review_and_revise:
            self.logger.info("Review and revise module is disabled. Skipping...")
            return draft
        if self.agentic_refine_section:
            return agentic_revise_survey_in_parts(self, draft, outline, self.config.ModuleInfo.SurveyGenerator.max_review_revise_iterations, code_report)
        sections = draft.get("section_drafts", []) or []
        if len(sections) == 0:
            self.logger.error("No sections found in draft for review and revise.")
            raise ValueError("No sections in draft.")
        
        max_parallel = getattr(self.config.ModuleInfo.SurveyGenerator, 'revise_section_in_parallel', 1)
        
        if max_parallel <= 1:
            revised_sections = []
            for idx, section_text in enumerate(sections):
                self.logger.info(f"\n\n***** Reviewing and Revising Section {idx + 1}/{len(sections)}: {outline.get('sections', [])[idx].get('title', 'No Title')} *****")
                previous_section_text = sections[idx - 1] if idx > 0 else ""
                next_section_text = sections[idx + 1] if idx + 1 < len(sections) else ""
                revised_text = self.review_and_revise_section(
                    section_text,
                    previous_section_text=previous_section_text,
                    next_section_text=next_section_text,
                    section_outline=outline.get('sections', [])[idx],
                    code_report = code_report,
                    env_report = env_report
                )
                revised_sections.append(revised_text)
        else:
            max_parallel = min(max_parallel, len(sections))
            self.logger.info(f"\n\n***** Processing {len(sections)} sections in parallel (max {max_parallel} workers) *****")
            
            def revise_section_with_context(idx):
                section_text = sections[idx]
                self.logger.info(f"Reviewing and Revising Section {idx + 1}/{len(sections)}: {outline.get('sections', [])[idx].get('title', 'No Title')}")
                previous_section_text = sections[idx - 1] if idx > 0 else ""
                next_section_text = sections[idx + 1] if idx + 1 < len(sections) else ""
                
                return self.review_and_revise_section(
                    section_text,
                    previous_section_text=previous_section_text,
                    next_section_text=next_section_text,
                    section_outline=outline.get('sections', [])[idx],
                    code_report = code_report,
                    env_report = env_report
                )
            
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                future_to_idx = {executor.submit(revise_section_with_context, idx): idx for idx in range(len(sections))}
                revised_sections = [None] * len(sections)
                
                for future in tqdm(as_completed(future_to_idx), total=len(sections), desc="Revising sections in parallel", unit="section"):
                    idx = future_to_idx[future]
                    try:
                        revised_sections[idx] = future.result()
                    except Exception as exc:
                        self.logger.error(f"Section {idx} generated an exception: {exc}")
                        revised_sections[idx] = sections[idx]

        # if self.config.BasicInfo.debug:
        #     self.logger.info("\n\n=== Revised Sections ===")
        #     with open("./revised_sections_debug.txt", "w", encoding="utf-8") as f:
        #         for idx, sec in enumerate(revised_sections):
        #             f.write("\n")
        #             f.write(sec)

        draft['section_drafts'] = revised_sections
        draft["full_draft"] = outline.get("title", self.config.BasicInfo.topic + " Survey") + "\n\n" + "\n\n".join(revised_sections)

        return draft

    def revise_survey(self, survey, outline, suggestion, code_report=None, env_report = None):
        self.logger.info(f"\n--- [Reviser] processing suggestion: {suggestion}... ---")
        # TODO: refine the query text
        valid = False
        for _ in range(self.config.ModuleInfo.SurveyGenerator.section_revise_retry):
            try:
                if self.config.ModuleInfo.SurveyGenerator.include_env_report and env_report:
                    final_report = f"CODE REPORT: {code_report}\n\nENV REPORT: {env_report}\n\n"
                else:
                    final_report = code_report

                if self.config.ModuleInfo.SurveyGenerator.include_code_report and final_report:
                    additional_context = CODE_REPORT_PROMPT_FOR_SURVEY_REVISER.format(code_report = final_report)
                    self.logger.info(f"injecting code report to survey reviser: {additional_context[:500]}")
                else:
                    additional_context = "None"
                    
                prompt = SURVEY_REVISE.format(
                    topic=self.config.BasicInfo.topic,
                    survey_outline = self._format_survey_outline(outline, False) or "",
                    survey=survey,
                    reviewer_suggestion=suggestion,
                    additional_context = additional_context,
                )

                parsed_json = extract_json(
                    self.chat_agent.remote_chat(
                        prompt,
                        temperature=self.config.ModuleInfo.SurveyGenerator.section_revise_temperature,
                    )
                )
                if not isinstance(parsed_json, dict):
                    raise ValueError("Parsed JSON is not a dict.")
                if "###" in parsed_json.get("newText", "") or "###" in parsed_json.get("originalText", ""):
                    self.logger.warning("OUTLINEREVISEDEBUG: Revision contains title: '###', which may break markdown structure. This is not allowed.\n")
                    self.logger.info(f"orginal_text: {parsed_json.get('originalText', '')}\n")
                    self.logger.info(f"new_text: {parsed_json.get('newText', '')}\n")
                    raise ValueError("Revision contains '###' which may break markdown structure.")
                valid = True
                break
            except Exception as e:
                self.logger.error(f"Section revision attempt failed with error {e}. Retrying...")
        
        if not valid:
            self.logger.error("Failed to generate valid revision after retries.")
            raise ValueError("Invalid section revision output.")

        if isinstance(parsed_json, dict):
            if parsed_json.get("action") == "done":
                if self.config.BasicInfo.debug:
                    self.logger.info("   [Reviser] Decided no change needed.")
                return survey
            elif parsed_json.get("action") == "replace":
                if self.config.BasicInfo.debug:
                    self.logger.info("   [Reviser] Applying revision...")
                return self._apply_revision_to_text(survey, parsed_json)
        else:
            raise ValueError("Parsed revision JSON is not a dict.")
        return survey
    
    def _format_survey_outline(self, outline, include_description = True):
        text = ""
        for section_outline in outline.get("sections"):
            text += f"- Section title: {section_outline.get('title', '')}\n"
            if include_description:
                text += f"  Section description: {section_outline.get('description', '')}\n"
            for subsection in section_outline.get("subsections", []):
                text += f"  - Subsection title: {subsection.get('title', '')}\n"
                if include_description:
                    text += f"    Subsection description: {subsection.get('description', '')}\n"
        return text

    def review_survey(self, survey, survey_outline = None, code_report=None, env_report = None):
        self.logger.info("\n--- [Reviewer] analyzing text... ---")
        if self.config.ModuleInfo.SurveyGenerator.include_env_report and env_report:
            final_report = f"CODE REPORT: {code_report}\n\nENV REPORT: {env_report}\n\n"
        else:
            final_report = code_report
        if self.config.ModuleInfo.SurveyGenerator.include_code_report and final_report:
            additional_context = CODE_REPORT_PROMPT_FOR_SURVEY_REVIEWER.format(code_report = final_report)
            self.logger.info(f"injecting code report to survey reviewer: {additional_context[:1500]}")
        else:
            additional_context = "None"

        review_prompt = SURVEY_REVIEW.format(
            topic = self.config.BasicInfo.topic,
            survey=survey,
            survey_outline = self._format_survey_outline(survey_outline, False),
            additional_context = additional_context,
        )
        valid = False
        for _ in range(self.config.ModuleInfo.SurveyGenerator.section_review_retry):
            try:
                suggestions = extract_json(
                    self.chat_agent.remote_chat(
                        review_prompt,
                        temperature=self.config.ModuleInfo.SurveyGenerator.section_review_temperature,
                    )
                )
                valid = True
                break
            except Exception as e:
                self.logger.error(f"Section review failed with error {e}. Retrying...")
                continue

        if valid:
            print(f"   [Reviewer] Generated {len(suggestions)} suggestions.")
            return suggestions
        else:
            print("   [Reviewer] Failed to generate valid list.")
            raise ValueError("Invalid section review output.")

    def review_and_revise_survey(self, survey, outline, code_report=None, env_report = None):
        MAX_WHOLE_SURVEY_ITERATION = self.config.ModuleInfo.SurveyGenerator.review_and_revise_whole_survey_max_iteration
        if self.agentic_refine_survey:
            return agentic_revise_survey_whole(self, survey, outline, MAX_WHOLE_SURVEY_ITERATION, code_report)
        for i in range(MAX_WHOLE_SURVEY_ITERATION):
            self.logger.info(f"\n\n====== step 2 revise whole survey: OUTER LOOP {i+1}/{MAX_WHOLE_SURVEY_ITERATION} ======")
            
            try:
                suggestions = self.review_survey(survey, outline, code_report=code_report, env_report = env_report)
            except Exception as e:
                self.logger.error(f"Survey review failed in OUTER LOOP {i+1} with error {e}. Exiting review and revise loop.")
                continue
            
            suggestions = suggestions[:self.config.ModuleInfo.SurveyGenerator.reviewer_max_suggestions]
            if not suggestions or len(suggestions) == 0:
                if (isinstance(suggestions, list) and len(suggestions) == 1 and suggestions[0].lower() == "done"):
                    self.logger.info("Reviewer indicates completion. Exiting.")
                    break
                elif isinstance(suggestions, str) and suggestions.lower() == "done":
                    self.logger.info("Reviewer indicates completion. Exiting.")
                    break
                else:
                    self.logger.info(f"Unexpected reviewer output type: {type(suggestions)} or empty suggestions.")
                    self.logger.info("No suggestions generated. Exiting.")
                    break

            self.logger.info(f"Starting to apply {len(suggestions)} suggestions...")
            for idx, sug in enumerate(suggestions):
                try:
                    new_survey = self.revise_survey(survey, outline, sug, code_report=code_report, env_report = env_report)
                except Exception as e:
                    self.logger.error(f"Section revision failed for suggestion {idx+1} in OUTER LOOP {i+1} with error {e}. Skipping this suggestion.")
                    continue
                
                if new_survey != survey:
                    survey = new_survey
                else:
                    self.logger.info(f"   (Suggestion {idx+1} resulted in no change)")
            
            self.logger.info(f"\n--- LOOP {i+1} REVISED FINISH --- \n")
            # if self.config.BasicInfo.debug and (i + 1) % 5 == 0:
            #     self.logger.info(survey)
            self.logger.info(f"\n--- End of OUTER LOOP {i+1} --- \n")
        # if self.config.BasicInfo.debug:
        #     self.logger.info("\n\n=== Final Result ===")
        #     self.logger.info(survey)
        return survey



    def _format_full_survey_text_from_drafts(self, draft):
        sections = draft.get("section_drafts", []) or []

        def _parse_heading(line: str):
            """Return cleaned title if the line looks like a markdown heading."""
            match = re.match(r"^\s*(#+)\s*(.*?)\s*(#*)\s*$", line)
            if not match:
                return None
            # Drop trailing # and surrounding whitespace to isolate the title text.
            title = re.sub(r"#+$", "", match.group(2) or "").strip()
            # Remove any leading numeric index like "1." or "1.2" that may already be present.
            title = re.sub(r"^\d+(?:\.\d+)*\s*[\.|\-|\)]?\s*", "", title)
            return title if title else None

        formatted_sections = []

        for sec_idx, raw_section in enumerate(sections, start=1):
            if self.config.BasicInfo.debug:
                self.logger.info(f"------RAW SECTION {sec_idx}------")
                ## assign section titles
                raw_section = "##" + draft["outline"].get("sections", [])[sec_idx - 1].get("title", "") + "\n" + raw_section
                self.logger.info(raw_section)
                self.logger.info("----------------------------------")
            lines = (raw_section or "").splitlines()
            new_lines: list[str] = []
            saw_section_heading = False
            sub_idx = 0
            last_added_heading = None  # Track the last added heading to detect duplicates

            for line in lines:
                # if self.config.BasicInfo.debug:
                #     self.logger.info(f"ORIGINAL LINE: {line}")
                title = _parse_heading(line)
                if title:
                    if not saw_section_heading:
                        saw_section_heading = True
                        outline_title = draft["outline"].get("sections", [])[sec_idx - 1].get("title", "")
                        if self.config.BasicInfo.debug:
                            self.logger.info(f"outline title: {outline_title} - title: {title}")
                        if outline_title != "" and outline_title != title:
                            self.logger.warning("section title not same! use outline result")
                            title = outline_title
                        new_lines.append(f"## {sec_idx}. {title}")
                        last_added_heading = ("section", sec_idx, title)
                        if self.config.BasicInfo.debug:
                            self.logger.info(f"Extract Section Title: ## {sec_idx}. {title}")
                    else:
                        # Check for duplicate subsection title (exact match)
                        is_duplicate = False
                        
                        if last_added_heading and last_added_heading[0] == "subsection":
                            if title == last_added_heading[2]:
                                is_duplicate = True
                                if self.config.BasicInfo.debug:
                                    self.logger.info(f"Skipping duplicate subsection title: {title}")
                        
                        if not is_duplicate:
                            sub_idx += 1
                            new_lines.append(f"### {sec_idx}.{sub_idx}. {title}")
                            last_added_heading = ("subsection", sec_idx, title)
                            if self.config.BasicInfo.debug:
                                self.logger.info(f"Extract Subsection {sub_idx} Title: ### {sec_idx}.{sub_idx}. {title}")
                        # If duplicate, skip adding this heading but still track it
                    continue

                new_lines.append(line)
                # Reset last_added_heading when we add content (not a heading)
                last_added_heading = None

            # Post-process to add blank lines before subsection headings for better readability
            final_lines: list[str] = []
            for i, line in enumerate(new_lines):
                # Check if this line is a subsection heading (starts with ###)
                if re.match(r'^\s*###\s+\d+\.\d+', line):
                    # Add a blank line before subsection heading if not already present
                    if final_lines and final_lines[-1].strip() != '':
                        final_lines.append('')
                final_lines.append(line)
            new_lines = final_lines

            if not saw_section_heading:
                # Fallback: treat the first non-empty line as the section title; everything else stays as content.
                self.logger.warning(f"No section heading found in section {sec_idx}, using fallback title.")
                fallback_title = f"Section {sec_idx}"
                for idx, line in enumerate(new_lines):
                    if line.strip():
                        fallback_title = re.sub(r"#+$", "", line).strip("# ") or fallback_title
                        new_lines.pop(idx)
                        break
                new_lines.insert(0, f"## {sec_idx}. {fallback_title}\n")

            formatted_sections.append("\n".join(new_lines).strip())

        return f"# {draft.get('title', self.config.BasicInfo.topic + ' Survey')}" + "\n\n" +"\n\n".join(formatted_sections)

    def _validate_refinement_result(self, result: str, info_dict: dict = None) -> tuple:
        """Validation function for refinement results. Checks if result is non-empty and not an error message."""
        if result is None:
            raise ValueError("Result is None")
        if not isinstance(result, str):
            raise ValueError(f"Result is not a string, got {type(result)}")
        if len(result.strip()) == 0:
            raise ValueError("Result is empty")
        # Check if the result looks like an error/explanation instead of refined content
        result_lower = result.strip().lower()
        if result_lower.startswith("the draft text is empty") or result_lower.startswith("the user wants me to refine"):
            raise ValueError(f"Result appears to be an explanation rather than refined content: {result[:100]}...")
        return True, result

    def refine_draft(self, draft, code_report = None, env_report = None):
        # Optional: first refine each section independently with local context, keeping <title> citations.
        draft_text = draft["full_draft"]
        for section in draft.get("section_drafts", []) or []:
            if self.config.BasicInfo.debug:
                self.logger.info(f"OUTLINEDEBUG: Section draft before refinement: \n\n {section}\n")
                self.logger.info(f"Section draft word count: {len(section.split())}\n")
        if self.refine_in_parts:
            sections = draft.get("section_drafts", []) or []
            if self.refine_in_parts_mode == 0: # no refinement
                refined_sections = sections
            if self.refine_in_parts_mode == 1: # trivial refinement: can modify subsection title
                refined_sections = []
                for idx, section_text in enumerate(sections):
                    prev_text = sections[idx - 1] if idx > 0 else ""
                    next_text = sections[idx + 1] if idx + 1 < len(sections) else ""
                    part_prompt = DRAFT_REFINEMENT_IN_PARTS.format(
                        title = draft["outline"]["sections"][idx].get("title", ""),
                        previous_text=prev_text,
                        next_text=next_text,
                        draft_text=section_text
                    )
                    part_raw = None
                    for retry_time in range(5):
                        try:
                            part_raw = self.chat_agent.remote_chat(
                                part_prompt,
                                temperature=self.config.ModuleInfo.SurveyGenerator.draft_refinement_temperature,
                            )
                            break
                        except Exception as e:
                            self.logger.error(
                                f"Section {idx} refinement failed on retry {retry_time} with error: {e}. Retrying..."
                            )
                            continue
                    if part_raw is None:
                        raise RuntimeError(
                            f"Section {idx} refinement failed after all retries."
                        )
                    refined_sections.append(
                        part_raw
                    )
            if self.refine_in_parts_mode == 2: # advanced refinement: refinement in subsection unit
                subsection_pattern = re.compile(r"(####\s*(.+?))\n", re.IGNORECASE)

                # Split each section into (preamble, [(heading, body), ...])
                # preamble = text before the first #### — kept as-is, NOT refined
                all_section_preambles: list[str] = []
                all_section_chunks: list[list[tuple[str, str]]] = []  # per section: [(heading, body), ...]
                for sec_idx, section_text in enumerate(sections):
                    self.logger.info(f"[REFINE SPLIT DEBUG]: orginal text: \n{section_text}\n")
                    parts = subsection_pattern.split(section_text)
                    # parts layout from re.split with 2 groups:
                    # [pre_text, full_match_0, title_0, body_0, full_match_1, title_1, body_1, ...]
                    preamble = parts[0]  # text before first #### — not refined
                    self.logger.info(f"[REFINE SPLIT DEBUG]: preamble: \n{preamble}")
                    chunks: list[tuple[str, str]] = []
                    i = 1
                    while i + 2 <= len(parts):
                        heading = parts[i].rstrip("\n")   # e.g. "#### 2.1 Background"
                        body    = parts[i + 2] if i + 2 < len(parts) else ""
                        self.logger.info(f"[REFINE SPLIT DEBUG]: heading: \n{heading}")
                        self.logger.info(f"[REFINE SPLIT DEBUG]: body: \n{body}\n\n")
                        
                        # Additional check: if body is only whitespace or very short, log it
                        body_stripped = body.strip()
                        if len(body_stripped) < 10:  # Very short body might be problematic
                            self.logger.warning(f"[SPLIT DEBUG] Section {sec_idx} subsection '{heading}' has short body (len={len(body_stripped)}): '{body_stripped[:100]}...'")
                        
                        chunks.append((heading, body))
                        i += 3
                    all_section_preambles.append(preamble)
                    all_section_chunks.append(chunks)

                # Check if fast mode is enabled
                refine_fast_mode = getattr(self.config.ModuleInfo.SurveyGenerator, 'refine_in_parts_fast_mode', True)
                
                if refine_fast_mode:
                    # Fast mode: collect all prompts, batch them together, then reassemble by section
                    self.logger.info(f"[mode=2 fast] Collecting all subsection prompts across all sections...")
                    
                    # Collect all prompts with their section/subsection indices
                    all_prompts: list[str] = []
                    prompt_info: list[tuple[int, int, str, str]] = []  # (sec_idx, chunk_idx, heading, original_body)
                    
                    for sec_idx, chunks in enumerate(all_section_chunks):
                        section_title = draft["outline"]["sections"][sec_idx].get("title", "")
                        prev_section_text = sections[sec_idx - 1] if sec_idx > 0 else all_section_preambles[sec_idx]
                        next_section_text = sections[sec_idx + 1] if sec_idx + 1 < len(sections) else ""
                        
                        for chunk_idx, (heading, body) in enumerate(chunks):
                            # Skip empty body - use original content directly
                            if not body.strip():
                                self.logger.warning(f"[mode=2 fast] Section {sec_idx} subsection '{heading}' has empty body, using original content.")
                                prompt_info.append((sec_idx, chunk_idx, heading, body))
                                all_prompts.append(None)  # Placeholder - will use original body
                                continue
                            
                            prev_body = chunks[chunk_idx - 1][1] if chunk_idx > 0 else prev_section_text
                            next_body = chunks[chunk_idx + 1][1] if chunk_idx + 1 < len(chunks) else next_section_text
                            prompt = DRAFT_REFINEMENT_SUBSECTION_IN_PARTS.format(
                                topic=self.config.BasicInfo.topic,
                                subsection_title=heading,
                                section_title=section_title,
                                previous_text=prev_body,
                                next_text=next_body,
                                draft_text=body,
                            )
                            prompt_info.append((sec_idx, chunk_idx, heading, body))
                            all_prompts.append(prompt)
                    
                    # Separate valid prompts from placeholders
                    valid_prompt_indices = [i for i, p in enumerate(all_prompts) if p is not None]
                    valid_prompts = [all_prompts[i] for i in valid_prompt_indices]
                    
                    if valid_prompts:
                        self.logger.info(f"[mode=2 fast] Batching {len(valid_prompts)} prompts for batch_remote_chat_with_retry...")
                        
                        try:
                            results = self.chat_agent.batch_remote_chat_with_retry(
                                valid_prompts,
                                validate_fn=self._validate_refinement_result,
                                max_retry=self.config.ModuleInfo.SurveyGenerator.draft_refinement_max_retry or 7,
                                desc="[mode=2 fast] Refining all subsections...",
                                temperature=self.config.ModuleInfo.SurveyGenerator.draft_refinement_temperature,
                            )
                        except Exception as e:
                            self.logger.error(f"[mode=2 fast] batch_remote_chat_with_retry failed: {e}. Falling back to original content.")
                            results = [None] * len(valid_prompts)
                        
                        # Map results back to all_prompts indices
                        refined_results = [None] * len(all_prompts)
                        for local_idx, global_idx in enumerate(valid_prompt_indices):
                            refined_results[global_idx] = results[local_idx]
                    else:
                        refined_results = [None] * len(all_prompts)
                    
                    # Reassemble sections from results
                    refined_sections = []
                    for sec_idx, chunks in enumerate(all_section_chunks):
                        if not chunks:
                            self.logger.info(f"[mode=2 fast] Section {sec_idx} has no subsections, keeping as-is.")
                            # for debug
                            self.logger.info(f"[mode=2 fast] Section {sec_idx} has no subsections: {sections[sec_idx]}")
                            refined_sections.append(sections[sec_idx])
                            continue
                        
                        subsections: list[str] = []
                        preamble = all_section_preambles[sec_idx]
                        if preamble:
                            subsections.append(preamble)
                        
                        for chunk_idx, (heading, original_body) in enumerate(chunks):
                            # Find this subsection's result
                            global_idx = None
                            for i, (s_idx, c_idx, h, b) in enumerate(prompt_info):
                                if s_idx == sec_idx and c_idx == chunk_idx:
                                    global_idx = i
                                    break
                            
                            if global_idx is not None and refined_results[global_idx] is not None:
                                refined_body = refined_results[global_idx]
                            else:
                                # Use original body if refinement failed or was skipped
                                refined_body = original_body
                                self.logger.info(f"[mode=2 fast] Section {sec_idx} subsection {chunk_idx} using original body (refinement failed/skipped).")
                            
                            subsections.append(f"{heading}\n{refined_body}")
                        
                        refined_sections.append("\n".join(subsections))
                        self.logger.info(f"[mode=2 fast] Section {sec_idx} reassembled with {len(chunks)} subsections.")
                
                else:
                    # Normal mode: process section by section (original behavior)
                    refined_sections = []
                    for sec_idx, chunks in enumerate(all_section_chunks):
                        section_title = draft["outline"]["sections"][sec_idx].get("title", "")
                        prev_section_text = sections[sec_idx - 1] if sec_idx > 0 else all_section_preambles[sec_idx]
                        next_section_text = sections[sec_idx + 1] if sec_idx + 1 < len(sections) else ""

                        if not chunks:
                            # No subsections found — keep section as-is
                            self.logger.info(f"[mode=2] Section {sec_idx} has no subsections, skipping refinement.")
                            refined_sections.append(sections[sec_idx])
                            continue

                        # Build prompts only for the named subsection chunks (preamble excluded)
                        sec_prompts: list[str] = []
                        sec_chunk_info: list[tuple[str, str]] = []  # (heading, original_body)
                        
                        for chunk_idx, (heading, body) in enumerate(chunks):
                            # Skip empty body - use original content directly
                            if not body.strip():
                                self.logger.info(f"[mode=2] Section {sec_idx} subsection {chunk_idx} has empty body, skipping refinement.")
                                sec_prompts.append(None)  # Placeholder
                                sec_chunk_info.append((heading, body))
                                continue
                            
                            prev_body = chunks[chunk_idx - 1][1] if chunk_idx > 0 else prev_section_text
                            next_body = chunks[chunk_idx + 1][1] if chunk_idx + 1 < len(chunks) else next_section_text
                            prompt = DRAFT_REFINEMENT_SUBSECTION_IN_PARTS.format(
                                topic=self.config.BasicInfo.topic,
                                subsection_title=heading,
                                section_title=section_title,
                                previous_text=prev_body,
                                next_text=next_body,
                                draft_text=body,
                            )
                            sec_prompts.append(prompt)
                            sec_chunk_info.append((heading, body))

                        # Separate valid prompts from placeholders
                        valid_prompt_indices = [i for i, p in enumerate(sec_prompts) if p is not None]
                        valid_prompts = [sec_prompts[i] for i in valid_prompt_indices]
                        
                        if not valid_prompts:
                            # All chunks were empty, use original section
                            self.logger.info(f"[mode=2] Section {sec_idx} all subsections empty, keeping as-is.")
                            refined_sections.append(sections[sec_idx])
                            continue

                        self.logger.info(f"[mode=2] Section {sec_idx}: refining {len(valid_prompts)} subsections via batch_remote_chat_with_retry...")
                        
                        try:
                            results = self.chat_agent.batch_remote_chat_with_retry(
                                valid_prompts,
                                validate_fn=self._validate_refinement_result,
                                max_retry=self.config.ModuleInfo.SurveyGenerator.draft_refinement_max_retry or 5,
                                desc=f"[Section {sec_idx}] Refining subsections...",
                                temperature=self.config.ModuleInfo.SurveyGenerator.draft_refinement_temperature,
                            )
                        except Exception as e:
                            self.logger.error(f"[mode=2] Section {sec_idx} batch_remote_chat_with_retry failed: {e}. Using original content.")
                            results = [None] * len(valid_prompts)
                        
                        # Map results back to all chunk indices
                        refined_bodies: list[str | None] = [None] * len(sec_prompts)
                        for local_idx, global_idx in enumerate(valid_prompt_indices):
                            refined_bodies[global_idx] = results[local_idx]
                        
                        # Fall back to original body for any that failed
                        for idx, (heading, original_body) in enumerate(sec_chunk_info):
                            if refined_bodies[idx] is None:
                                self.logger.warning(f"[mode=2] Section {sec_idx} subsection {idx} refinement failed. Using original body.")
                                refined_bodies[idx] = original_body

                        # Reassemble: preamble (unchanged) + refined subsections
                        subsections: list[str] = []
                        preamble = all_section_preambles[sec_idx]
                        if preamble:
                            subsections.append(preamble)
                        for idx, (heading, _original_body) in enumerate(sec_chunk_info):
                            subsections.append(f"{heading}\n{refined_bodies[idx]}")
                            self.logger.info(f"======= refinement debug for Section {sec_idx} Subsection {idx} #### {heading} =======")
                        refined_sections.append("\n".join(subsections))

            draft['section_drafts'] = refined_sections

            survey = self._format_full_survey_text_from_drafts(draft)

            if self.config.ModuleInfo.SurveyGenerator.review_and_revise_whole_survey_in_refinement:
                survey = self.review_and_revise_survey(survey, draft["outline"], code_report=code_report, env_report = env_report)

            self._test_valid_citation_threshold(survey)

            survey, references, correct_titles, err_titles = self.extract_and_process_citations(survey)

            if self.config.BasicInfo.debug:
                self.logger.info(f"correct citationnum {len(correct_titles)}")
                self.logger.info(f"err citationnum {len(err_titles)}")
                self.logger.info(f"valid ratio: {len(correct_titles)/(len(correct_titles)+len(err_titles)) if (len(correct_titles)+len(err_titles))>0 else 0}")

            if self.config.BasicInfo.debug:
                self.logger.info(f"Total unique paper titles in correct format in Draft after refining in parts: {len(set(correct_titles))}")

        else:
            prompt = DRAFT_REFINEMENT.format(draft_text=draft_text)
            output = None
            for retry_time in range(5):
                try:
                    output_raw = self.chat_agent.remote_chat(
                        prompt,
                        temperature=self.config.ModuleInfo.SurveyGenerator.draft_refinement_temperature,
                    )
                    output = extract_json(output_raw)
                    break
                except Exception as e:
                    self.logger.error(
                        f"Draft refinement failed on retry {retry_time} with error: {e}. Retrying..."
                    )
                    continue

            if output is None:
                raise RuntimeError("Draft refinement failed after all retries.")

            survey = output.get("refined_survey", draft_text)

            if self.config.ModuleInfo.SurveyGenerator.review_and_revise_whole_survey_in_refinement:
                survey = self.review_and_revise_survey(survey, draft["outline"], code_report=code_report, env_report = env_report)

            references = output.get("references", [])

        self.logger.info(f"Total references found: {len(references)}. Generating...")
        reference = "References:\n"
        for index, paper_id in tqdm(enumerate(references)):
            # self.logger.info(f" Generating reference for paper ID: {paper_id}") # YZY DEBUG
            # if self.use_title_in_draft:
            #     # resolve title to paper id
            #     try:
            #         resolved_paper_id, _, _ = self.database.resolve_title_to_paper_id(paper_id)
            #         paper_id = resolved_paper_id
            #         if self.config.BasicInfo.debug:
            #             self.logger.info(f"Resolved title '{paper_id}' to paper ID '{resolved_paper_id}' in REFERENCE GENERATION.")
            #     except ValueError:
            #         self.logger.error(f"Title '{paper_id}' could not be resolved to a paper ID in REFERENCE GENERATION. Skipping this reference.")
            #         continue
            try:
                reference += f"{index + 1}. {self.work_analyzer.generate_mla(paper_id)}\n"
            except Exception as e:
                self.logger.error(f"Failed to generate reference for paper ID: {paper_id} with error {e}. Skipping this reference.")
                reference += f"{index + 1}. unknown citation\n"

        return survey + "\n\n" + reference, references

    def save_survey(self, final_survey, references):
        save_path = self.config.BasicInfo.save_path
        save_json_path = self.config.BasicInfo.save_json_path
        topic = getattr(self.config.BasicInfo, "topic", "survey")

        # ensure parent dirs exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)

        if self.config.BasicInfo.debug:
            self.logger.info(f"\n--------------------------")
            self.logger.info(f"final survey:{final_survey}")
            self.logger.info(f"--------------------------\n")
        # write md
        if self.config.BasicInfo.debug:
            self.logger.info(f"Saving final survey to {save_path}...")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(final_survey)

        # write json
        if self.config.BasicInfo.debug:
            self.logger.info(f"Saving final survey JSON to {save_json_path}...")
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "topic": topic,
                    "topic_slug": os.path.basename(os.path.dirname(save_json_path)),
                    "paper": final_survey,
                    "references": references,
                },
                f,
                indent=2,
            )
                
    def count_unique_paper_ids(self, text: str) -> int:
        return len(self.get_unique_paper_ids_from_raw(text))

    def count_unique_titles(self, text: str) -> int:
        _, _, titles, _ = self.extract_and_validate_titles_in_text(text)
        unique_titles = set(titles)
        return len(unique_titles)

    def get_unique_paper_ids_from_raw(self, text: str):
        pattern = re.compile(
            r"<Paper ID:\s*([^\s>]+)\s*>"
            r"|\(Paper ID:\s*([^\s\)]+)\s*\)"
            r"|<Paper\s*ID\s*:\s*([^\s>]+)\s*>"
            r"|\(Paper\s*ID\s*([^\s\)]+)\s*\)"
            r"|<Paper\s*([^\s>]+)\s*>"
            r"|\(Paper\s*([^\s\)]+)\s*\)"
            r"|<Paper\s*<\s*([^\s>]+)\s*>\s*>"
            r"|<Paper ID:\s*([^>]+?)\s*>",  # e.g., '<Paper ID: 2408.08464, Paper ID: 2406.09324>'",
            flags=re.IGNORECASE,
        )

        matches = pattern.findall(text or "")
        ids = set()
        ordered_ids = []
        for m in matches:
            raw = next((grp for grp in m if grp), "").strip()
            if not raw:
                continue
            # handle combined forms like "2408.08464, Paper ID: 2406.09324"
            parts = [p.strip() for p in re.split(r",|and", raw) if p.strip()]
            for p in parts:
                # remove leading 'Paper ID:' if present
                p = re.sub(r"^(?i:paper\s*id:?)\s*", "", p).strip()
                if p not in ids:
                    ids.add(p)
                    ordered_ids.append(p)
        return ordered_ids

    def extract_and_validate_titles_in_text(self, text: str):
        """Extract titles inside '<...>' and validate each whole bracketed chunk as one citation.
        Splitting by comma risks breaking titles that contain commas, so treat the entire content as a single title.
        """
        if not isinstance(text, str):
            return False, []

        matches = re.findall(r"<([^<>]+)>", text or "")
        err_titles = []
        paper_ids = []
        titles = []

        for raw in matches:
            title = raw.strip()
            if not title:
                continue
            try:
                paper_id, matched_title, _ = self.database.resolve_title_to_paper_id(
                                                                title_text = title,
                                                                min_title_similarity = self.config.ModuleInfo.SurveyGenerator.valid_title_min_similarity)
            except ValueError:
                err_titles.append(title)
                if self.config.BasicInfo.debug:
                    self.logger.warning(
                        f"Title '{title}' could not be resolved to a paper ID in VALIDATE TITLES."
                    )
                continue

            # if self.config.BasicInfo.debug:
            #     self.logger.info(
            #         f"Title '{title}' resolved to paper ID '{paper_id}' with matched title '{matched_title}' in VALIDATE TITLES."
            #     )
            paper_ids.append(paper_id)
            titles.append(matched_title)

        return len(err_titles) == 0, paper_ids, titles, err_titles

    def extract_and_process_citations(self, text: str):
        """Normalize citations, order references, and convert to numbered brackets."""
        if not isinstance(text, str):
            self.logger.error("Input text to EXTRACT AND PROCESS CITATIONS is not a string in EXTRACT AND PROCESS CITATIONS.")
            return "", [], [], []

        ordered_paper_ids: list[str] = []
        paper_id_to_index: dict[str, int] = {}
        normalized_parts: list[str] = []
        last_idx = 0

        valid_titles: list[str] = []
        err_titles: list[str] = []

        for match in re.finditer(r"<([^<>]+)>", text):
            normalized_parts.append(text[last_idx:match.start()])
            title = match.group(1).strip()
            last_idx = match.end()
            if not title:
                continue

            try:
                paper_id, matched_title, _ = self.database.resolve_title_to_paper_id(
                    title_text=title,
                    min_title_similarity=self.config.ModuleInfo.SurveyGenerator.valid_title_min_similarity,
                )
                valid_titles.append(title)
            except ValueError:
                if getattr(self.config, 'AblationInfo', None) and getattr(self.config.AblationInfo, 'survey_generator_disabled', False):
                    self.logger.info(f"Ablation mode: Using title as paper_id for '{title}'")
                    paper_id = title
                else:
                    err_titles.append(title)
                    if self.config.BasicInfo.debug:
                        self.logger.warning(f"Title '{title}' could not be resolved. Removing citation.")
                    continue

            if paper_id not in paper_id_to_index:
                ordered_paper_ids.append(paper_id)
                paper_id_to_index[paper_id] = len(ordered_paper_ids)

            normalized_parts.append(f"[{paper_id_to_index[paper_id]}]")

        normalized_parts.append(text[last_idx:])
        processed_text = "".join(normalized_parts)

        return processed_text, ordered_paper_ids, valid_titles, err_titles

    def _test_valid_citation_threshold(self, text: str):
        self.logger.info("Testing valid citation thresholds from 0.1-0.9 and test valid ratio...")
        self.logger.info(f"Total extracted titles: {len(re.findall(r'<([^<>]+)>', text))}")
        self.logger.info(f"------------------------------------------------------------")
        threshold = 0.0
        while threshold < 1.0:

            valid_titles: list[str] = []
            err_titles: list[str] = []
            
            for match in re.finditer(r"<([^<>]+)>", text):
                title = match.group(1).strip()
                if not title:
                    continue

                try:
                    paper_id, matched_title, _ = self.database.resolve_title_to_paper_id(
                        title_text=title,
                        min_title_similarity=threshold,
                    )
                    valid_titles.append(title)
                except ValueError:
                    err_titles.append(title)
                    if self.config.BasicInfo.debug:
                        self.logger.warning(f"Title '{title}' could not be resolved. Removing citation.")
                    continue
            valid_ratio = len(valid_titles) / (len(valid_titles) + len(err_titles)) if (len(valid_titles) + len(err_titles)) > 0 else 0
            self.logger.info(f"At threshold {threshold}, valid titles: {len(valid_titles)}, err titles: {len(err_titles)}, valid ratio: {valid_ratio}")
            threshold += 0.1

        self.logger.info(f"------------------------------------------------------------")

        return 
