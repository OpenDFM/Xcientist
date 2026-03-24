import os
import numpy as np
import tiktoken
import re
import json
from tqdm import trange,tqdm
import time
import threading
import sys

from utils.api_call import ChatAgent
from utils.utils import get_hash, extract_json

from modules.pe import (
    EVAL_CRITERIA,
    JUDGE_WITH_CRITERIA_PROMPT,
    JUDGE_WITH_CRITERIA_PROMPT_10_DIMENSIONS,
    NLI_PROMPT,
    JUDGE_WITH_CRITERIA_PROMPT_10_DIMENSIONS_NO_EXP,
    JUDGE_WITH_CRITERIA_PROMPT_NO_EXP
)
from utils.rich_logger import get_logger

class Judge():
    def __init__(self, config, work_analyzer) -> None:
        self.config = config
        self.topic = config.BasicInfo.topic
        self.chat_agent = ChatAgent(config, self.config.ModuleInfo.Judge.use_different_api_for_judge)
        self.logger = get_logger("Judge")

        self.judge_model = config.ModuleInfo.Judge.model

        self.work_analyzer = work_analyzer

        self.rubric_judge_max_retry = 3

        self.explanation_in_rubric = self.config.ModuleInfo.Judge.explanation_in_rubric

    def __criteria_based_judging(self, topic, survey, criterion, res_l, reason_l, idx, retry=1):
        criterion_paras = EVAL_CRITERIA[criterion]
        survey = self.chat_agent.truncate_prompt(survey, self.config.APIInfo.llm_max_context_length - 10000, self.config.APIInfo.llm_model_name)
        if self.config.ModuleInfo.Judge.rubrics_eval_4_dimensions:
            if self.explanation_in_rubric:
                prompt_template = JUDGE_WITH_CRITERIA_PROMPT
            else:
                prompt_template = JUDGE_WITH_CRITERIA_PROMPT_NO_EXP
            prompt = prompt_template.format(
                TOPIC = topic, SURVEY = survey, Criterion_Description = criterion_paras['description'],
                Score_1_Description = criterion_paras['score 1'], 
                Score_2_Description = criterion_paras['score 2'],
                Score_3_Description = criterion_paras['score 3'],
                Score_4_Description = criterion_paras['score 4'], 
                Score_5_Description = criterion_paras['score 5']
            )
        else:
            if self.explanation_in_rubric:
                prompt_template = JUDGE_WITH_CRITERIA_PROMPT_10_DIMENSIONS
            else:
                prompt_template = JUDGE_WITH_CRITERIA_PROMPT_10_DIMENSIONS_NO_EXP
            prompt = prompt_template.format(
                TOPIC = topic, SURVEY = survey, Criterion_Description = criterion_paras['description'],
                Score_1_Description = criterion_paras['score 1'], 
                Score_2_Description = criterion_paras['score 2'],
                Score_3_Description = criterion_paras['score 3'],
                Score_4_Description = criterion_paras['score 4'], 
                Score_5_Description = criterion_paras['score 5'],
                Score_6_Description = criterion_paras['score 6'],
                Score_7_Description = criterion_paras['score 7'],
                Score_8_Description = criterion_paras['score 8'],
                Score_9_Description = criterion_paras['score 9'],
                Score_10_Description = criterion_paras['score 10']
            )
        # self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        result = {"score": 0, "reason": "init"}
        try:
            result = self.chat_agent.remote_chat(text_content = prompt, temperature=0)
            if self.explanation_in_rubric:
                result = extract_json(result)
                score = result["score"]
                reason = result["reason"]
            else:
                score = self.extract_num(result)
                reason = "No explanation provided."
        except Exception as e:
            self.logger.error(f"Error in criteria judging for criterion {criterion} on retry {retry}: {e}")
            if retry <= self.rubric_judge_max_retry:
                return self.__criteria_based_judging(topic, survey, criterion, res_l, reason_l, idx, retry=retry+1)
            else:
                self.logger.error(f"Failed to judge criterion {criterion} after 3 retries. Assigning score 0.")
                score = 0
                reason = "Failed to judge after retries."
        res_l[idx] = float(score)
        reason_l[idx] = reason
        return result
    
    def extract_num(self, string):
        numbers = re.findall(r'\d+', string)
        if len(numbers) == 0:
            return ''
        return eval(numbers[0])

    def batch_criteria_based_judging(self, survey, criteria):
        thread_l = []
        scores = [0] * len(criteria)
        reasons = [""] * len(criteria)
        for i in range(len(criteria)):
            thread = threading.Thread(target=self.__criteria_based_judging, args=(self.topic, survey, criteria[i], scores, reasons, i))
            thread_l.append(thread)
            thread.start()
        for thread in thread_l:
            thread.join()
        return scores, reasons
    
    def __nli(self, sources, claim, res_l, idx):
        prompt = NLI_PROMPT.format(
            SOURCE='\n'.join(sources),
            CLAIM=claim
        )

        res = self.chat_agent.remote_chat(text_content = prompt, 
                                            temperature=self.config.ModuleInfo.Judge.nli_temperature, 
                                            model=self.judge_model)

        if 'yes' in res.lower():
            res_l[idx] += 1
            return 1
        else:
            res_l[idx] += 0
            return 0
        
    def __relevant(self, sources, com_sources, claim, res_l, idx):
        prompt = NLI_PROMPT.format(
            SOURCE='\n'.join(sources),
            CLAIM=claim
        )
        # self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)

        res = self.chat_agent.remote_chat(text_content = prompt, 
                                            temperature=self.config.ModuleInfo.Judge.nli_temperature, 
                                            model=self.judge_model)
        if sources == []:
            res_l[idx] += 0
            return 0

        if 'yes' in res.lower():
            res_l[idx] += 1
            return 1
        else:
            prompt = NLI_PROMPT.format(
                SOURCE='\n'.join(com_sources),
                CLAIM=claim
            )
            # self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
            res = self.chat_agent.remote_chat(text_content = prompt, 
                                                temperature=self.config.ModuleInfo.Judge.nli_temperature,
                                                model=self.judge_model)
            if 'yes' in res.lower():
                res_l[idx] += 0
                return 0
            else:
                res_l[idx] += 1
                return 1
      
    def citation_quality(self, survey_with_reference, references):
        survey = survey_with_reference.split('## References')[0]
        survey_sections = survey.split('###')
        citation_pattern = re.compile(r'[^.!?]*\[[^\]]+\][^.!?]*[.!?]')
        sentences = []
        for content in survey_sections:
            sentences += citation_pattern.findall(content)
        claims = []
        sources_ids = []
        index_to_keynotes = {}
        for s in sentences:
            sources = re.findall(pattern=r'\[(.*?)\]', string=s)
            if len(sources) > 0:
                source_ids = set()
                for ref in sources:
                    for num in ref.split(';'):
                        number = self.extract_num(num)
                        if number != '':
                            source_ids.add(number)
                if len(source_ids) >0:
                    claims.append(re.sub(pattern=r'\[(.*?)\]', repl='',string=s))
                    sources_ids.append(list(source_ids))
        
        # build index to keynote mapping
        fail_index = []
        for source_ids in sources_ids:
            for index in list(source_ids):
                if index not in index_to_keynotes:
                    if index-1 < 0 or index-1 >= len(references):
                        self.logger.error(f"Reference index {index} out of range.")
                        # index_to_keynotes[index] = ""
                        fail_index.append(index)
                        continue
                    try:
                        keynote_dict = self.work_analyzer.get_paper_keynote(references[index-1])
                        index_to_keynotes[index] = f'paper {index} keynotes: {json.dumps(keynote_dict, ensure_ascii=False)}\n\n'  ## paper id start from 1 in reference list
                    except Exception as e:
                        self.logger.error(f"Error getting keynote for paper id {references[index-1]} for CITATION eval: {e}")
                        # index_to_keynotes[index] = ""
                        fail_index.append(index)
        
        self.logger.warning(f"Failed to get keynotes for {len(fail_index)} references during citation evaluation.")
        if self.config.ModuleInfo.Judge.remove_failed_citation_in_eval:
            for index in fail_index:
                for source_ids in sources_ids:
                    if index in source_ids:
                        source_ids.remove(index)
        fail_index = set(fail_index)

        thread_l = []
        scores = [0] * len(claims)
        for i in range(len(claims)):
            keynotes = [index_to_keynotes[index] for index in sources_ids[i] if index not in fail_index]
            thread = threading.Thread(target=self.__nli, args=(keynotes, claims[i], scores, i))
            thread_l.append(thread)
            thread.start()
        for thread in tqdm(thread_l):
            thread.join()
        citation_num = 0
        thread_l = []
        precisions = [0] * len(claims)
        for j, claim, source_ids in zip(range(len(claims)), claims, sources_ids):
            citation_num += len(source_ids)
            if scores[j] == 1:
                for index in source_ids:
                    keynotes = [index_to_keynotes[index]] if index not in fail_index else []
                    com_keynotes = [index_to_keynotes[_] for _ in source_ids if not _ == index]
                    thread = threading.Thread(target=self.__relevant, args=(keynotes, com_keynotes, claim, precisions, j))
                    thread_l.append(thread)
                    thread.start()
        for thread in tqdm(thread_l):
            thread.join()

        precisions = np.array(precisions)

        if len(claims) == 0 or citation_num == 0:
            return 0.0, 0.0
        return np.array(scores).mean(), precisions.sum()/citation_num

    def count_valid_citation(self, references):
        count = 0
        for reference in references:
            try:
                mla = self.work_analyzer.generate_mla(paper_id = reference)
            except Exception as e:
                self.logger.error(f"Failed to generate MLA for paper id {reference} with error: {e}")
                continue
            count += 1
        return count

    def evaluate(self, survey, references): ## references is a list of paper ids
        recall = 0.0
        precision = 0.0
        Coverage = 0.0
        Structure = 0.0
        Relevance = 0.0
        Rigor = 0.0
        Depth = 0.0
        valid = 0
        eval_log = ""
        return_dict = {}
        reason_dict = {}
        
        if self.config.ModuleInfo.Judge.rubrics_eval_4_dimensions:
            criterion = ['Coverage', 'Structure','Relevance','Depth', 'Rigor&Authenticity']

            scores, reasons = self.batch_criteria_based_judging(survey, criterion)

            dimension_score = {}
            
            Avg_score = 0.0
            for c, s, r in zip(criterion, scores, reasons):
                print(f'{c} = {s}\n')
                eval_log += f'{c} = {s}\n'
                eval_log += f'Reason: {r}\n'
                dimension_score[c] = s
                reason_dict[c] = r
                Avg_score += s
            
            Avg_score /= len(criterion)

            return_dict.update(dimension_score)

            return_dict.update(
                {
                "Total_Score": Avg_score
                }
            )

            self.logger.info(f"Score_dict: {dimension_score}\n")

            eval_log += f"Coverage: {dimension_score.get('Coverage')}\n"
            eval_log += f"Structure: {dimension_score.get('Structure')}\n"
            eval_log += f"Relevance: {dimension_score.get('Relevance')}\n"   
            eval_log += f"Depth: {dimension_score.get('Depth')}\n"
            eval_log += f"Rigor&Authenticity: {dimension_score.get('Rigor&Authenticity')}\n"


        elif self.config.ModuleInfo.Judge.rubrics_eval_10_dimensions:
            criterion = ['Synthesis Quality', 'Organization', 'Readability','Academic Rigor','Clarity', 'Coherence', 'Comprehensiveness', 'Critical Analysis', 'Novelty and Insights', 'Future Directions']
            Core_Quality = 0
            Writing_Quality = 0
            Content_Depth = 0

            scores, reasons = self.batch_criteria_based_judging(survey, criterion)

            dimension_score = {}
            for c, s, r in zip(criterion, scores, reasons):
                print(f'{c} = {s}\n')
                eval_log += f'{c} = {s}\n'
                eval_log += f'Reason: {r}\n'
                dimension_score[c] = s
                reason_dict[c] = r
            
            return_dict.update(dimension_score)

            Core_Quality = (dimension_score['Synthesis Quality'] + dimension_score['Organization'])/2
            Writing_Quality = (dimension_score['Readability'] + dimension_score['Academic Rigor'] + dimension_score['Clarity'] + dimension_score['Coherence'])/4
            Content_Depth = (dimension_score['Comprehensiveness'] + dimension_score['Critical Analysis'] + dimension_score['Novelty and Insights'] + dimension_score['Future Directions'])/4
            Total_score = (Core_Quality*0.6+Writing_Quality*0.2+Content_Depth*0.2)

            return_dict.update({
                'Core_Quality': Core_Quality,
                'Writing_Quality': Writing_Quality,
                'Content_Depth': Content_Depth,
                "Total_Score": Total_score
            })
            self.logger.info(f"Score_dict: {dimension_score}\n")

            eval_log += f"Core_Quality: {Core_Quality}\n"
            eval_log += f"Writing_Quality: {Writing_Quality}\n"
            eval_log += f"Content_Quality: {Content_Depth}\n"   
            eval_log += f'Score: {Core_Quality*0.6+Writing_Quality*0.2+Content_Depth*0.2}\n'

        if self.config.ModuleInfo.Judge.citation_eval:
            recall, precision = self.citation_quality(survey, references)
            valid = self.count_valid_citation(references)
            self.logger.info(f"recall: {recall} precision: {precision}")


            eval_log += f'Judged by {self.judge_model}:\n'
            eval_log += f'Citation Recall = {recall}\nCitation Precision = {precision}\n'
            eval_log += f'Citation Number = {len(references)}\n'
            eval_log += f'Valid Citation Number = {valid}\n'
            eval_log += f'Valid Citation Ratio = {valid / len(references) if len(references) > 0 else 0.0}\n'
        self.logger.info(eval_log)

        eval_log += '----------------------------------------\n'

        if not os.path.exists(self.config.BasicInfo.evaluation_save_path):
            os.makedirs(os.path.dirname(self.config.BasicInfo.evaluation_save_path), exist_ok=True)
            
        # with open(self.config.BasicInfo.evaluation_save_path, 'a') as f:
        #     f.write(eval_log)

        return_dict.update({
            'Citation_Recall': recall,
            'Citation_Precision': precision,
            'Citation_Number': len(references),
            'Valid_Citation_Number': valid,
            'Valid_Citation_Ratio': valid / len(references) if len(references) > 0 else 0.0
        })

        return return_dict, reason_dict

    def save_evaluation(self, return_dict):
        if not os.path.exists(self.config.BasicInfo.evaluation_save_path):
            os.makedirs(os.path.dirname(self.config.BasicInfo.evaluation_save_path), exist_ok=True)

        with open(self.config.BasicInfo.evaluation_save_path, 'a') as f:
            f.write(f"===== Judging at {time.asctime()} ======")
            f.write(f"===== Judging by {self.chat_agent.self.model_name} ======")

        for key, value in return_dict.items():
            with open(self.config.BasicInfo.evaluation_save_path, 'a') as f:
                f.write(f'{key}: {value}\n')