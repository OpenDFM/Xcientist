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

from modules.pe import (
    EVAL_CRITERIA,
    JUDGE_WITH_CRITERIA_PROMPT,
    NLI_PROMPT,
)
from utils.rich_logger import get_logger

class Judge():
    def __init__(self, config, work_analyzer) -> None:
        self.config = config
        self.topic = config.BasicInfo.topic
        self.chat_agent = ChatAgent(config)
        self.logger = get_logger("Judge")

        self.judge_model = config.ModuleInfo.Judge.model

        self.work_analyzer = work_analyzer

    def __criteria_based_judging(self, topic, survey, criterion, res_l, idx):
        criterion_paras = EVAL_CRITERIA[criterion]

        prompt = JUDGE_WITH_CRITERIA_PROMPT.format(
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
        scores = self.chat_agent.remote_chat(text_content = prompt, temperature=0)
        res_l[idx] = self.extract_num(scores)
        return scores
    
    def extract_num(self, string):
        numbers = re.findall(r'\d+', string)
        if len(numbers) == 0:
            return ''
        return eval(numbers[0])

    def batch_criteria_based_judging(self, survey, criteria):
        thread_l = []
        scores = [0] * len(criteria)
        for i in range(len(criteria)):
            thread = threading.Thread(target=self.__criteria_based_judging, args=(self.topic, survey, criteria[i], scores, i))
            thread_l.append(thread)
            thread.start()
        for thread in thread_l:
            thread.join()
        return scores
    
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
                        index_to_keynotes[index] = ""
                        continue
                    try:
                        keynote_dict = self.work_analyzer.get_paper_keynote(references[index-1])
                        index_to_keynotes[index] = f'paper {index} keynotes: {json.dumps(keynote_dict, ensure_ascii=False)}\n\n'  ## paper id start from 1 in reference list
                    except Exception as e:
                        self.logger.error(f"Error getting keynote for paper id {references[index-1]} for cCITATION eval: {e}")
                        index_to_keynotes[index] = ""
                        fail_index.append(index)
        
        self.logger.warning(f"Failed to get keynotes for {len(fail_index)} references during citation evaluation.")
        for index in fail_index:
            for source_ids in sources_ids:
                if index in source_ids:
                    source_ids.remove(index)

        thread_l = []
        scores = [0] * len(claims)
        for i in range(len(claims)):
            keynotes = [index_to_keynotes[index] for index in sources_ids[i]]
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
                    keynotes = [index_to_keynotes[index]]
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

    def evaluate(self, survey, references): ## references is a list of paper ids
        Core_Quality = 0
        Writing_Quality = 0
        Content_Depth = 0
        recall = 0.0
        precision = 0.0
        eval_log = ""
        
        if self.config.ModuleInfo.Judge.rubrics_eval:
            criterion = ['Synthesis Quality', 'Organization', 'Readability','Academic Rigor','Clarity', 'Coherence', 'Comprehensiveness', 'Critical Analysis', 'Novelty and Insights', 'Future Directions']

            scores = self.batch_criteria_based_judging(survey, criterion)

            dimension_score = {}
            for c, s in zip(criterion, scores):
                print(f'{c} = {s}\n')
                eval_log += f'{c} = {s}\n'
                dimension_score[c] = s

            Core_Quality = (dimension_score['Synthesis Quality'] + dimension_score['Organization'])/2
            Writing_Quality = (dimension_score['Readability'] + dimension_score['Academic Rigor'] + dimension_score['Clarity'] + dimension_score['Coherence'])/4
            Content_Depth = (dimension_score['Comprehensiveness'] + dimension_score['Critical Analysis'] + dimension_score['Novelty and Insights'] + dimension_score['Future Directions'])/4
            
            self.logger.info(f"Core_Quality: {Core_Quality}")
            self.logger.info(f"Writing_Quality: {Writing_Quality}")
            self.logger.info(f"Content_Quality: {Content_Depth}")
            self.logger.info(f'Score: {Core_Quality*0.6+Writing_Quality*0.2+Content_Depth*0.2}')

            eval_log += f"Core_Quality: {Core_Quality}\n"
            eval_log += f"Writing_Quality: {Writing_Quality}\n"
            eval_log += f"Content_Quality: {Content_Depth}\n"   
            eval_log += f'Score: {Core_Quality*0.6+Writing_Quality*0.2+Content_Depth*0.2}\n'

        if self.config.ModuleInfo.Judge.citation_eval:
            recall, precision = self.citation_quality(survey, references)
            self.logger.info(f"recall: {recall} precision: {precision}")

            eval_log += f"citation recall: {recall}\n"
            eval_log += f"citation precision: {precision}\n"

            result = f'Judged by {self.judge_model}:\n'
            result += f'Citation Recall = {recall:.4f}\nCitation Precision = {precision:.4f}\n'
            self.logger.info(result)

            eval_log += result

        eval_log += '----------------------------------------\n'

        if not os.path.exists(self.config.BasicInfo.evaluation_save_path):
            os.makedirs(os.path.dirname(self.config.BasicInfo.evaluation_save_path), exist_ok=True)
            
        with open(self.config.BasicInfo.evaluation_save_path, 'a') as f:
            f.write(eval_log)

        return {
            'Core_Quality': Core_Quality,
            'Writing_Quality': Writing_Quality,
            'Content_Depth': Content_Depth,
            'Citation_Recall': recall,
            'Citation_Precision': precision
        }