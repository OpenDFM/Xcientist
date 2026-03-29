import hydra
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.rich_logger import get_logger
from omegaconf import OmegaConf
from modules.work_collector import WorkCollector
from modules.work_analyzer import WorkAnalyzer
from modules.survey_generator import SurveyGenerator
from modules.survey_generator import SurveyGenerator
from modules.judge import Judge
from topics.Benchmark_topics import SURVEYGEN_TOPICS, AUTOSURVEY_TOPICS
from utils.config_utils import merge_with_default_survey_config
from utils.file_utils import write_domain_header, write_topic_header, write_result, write_domain_result, write_header

import json
from pathlib import Path

logger = get_logger("baseline Eval")
        

def get_AutoSurvey_cfg(config, domain, topic, save_path, paper_path):
    file_path = Path(paper_path) / topic / "exp_1" / f"{topic}.json"
    with open(file_path , "r") as f:
        survey_data = json.load(f)
        survey = survey_data.get("survey", "")
        references = survey_data.get("reference", {})

        references_list = []
        for ref in references.values():
            references_list.append(ref)
    
        logger.info(f"Loaded survey and {len(references_list)} references for topic: {topic}")

    cfg_for_topic = OmegaConf.merge(config, {
                                    "BasicInfo": {
                                        "topic": topic, 
                                        "evaluation_save_path": save_path
                                        }
                                    })
    return cfg_for_topic, survey, references_list

def get_DeepSurvey_cfg(config, domain, topic, save_path, paper_path):
    file_path = Path(paper_path) / f"{topic.replace(' ', '_')}.json"
    logger.info(f"Loading survey from {file_path}")
    with open(file_path , "r") as f:
        survey_data = json.load(f)
        survey = survey_data.get("paper")
        references = survey_data.get("references")
    
        logger.info(f"Loaded survey and {len(references)} references for topic: {topic}")

    cfg_for_topic = OmegaConf.merge(config, {
                                    "BasicInfo": {
                                        "topic": topic, 
                                        "evaluation_save_path": save_path
                                        }
                                    })
    return cfg_for_topic, survey, references

def get_Human_cfg(config, domain, topic, save_path, paper_path):
    file_path = Path(paper_path) / f"{topic}" / "auto" /f"{topic}.md"
    logger.info(f"Loading human survey from {file_path}")
    with open(file_path , "r") as f:
        survey = f.read()
        references = []
    
        logger.info(f"Loaded survey and {len(references)} references for topic: {topic}")

    cfg_for_topic = OmegaConf.merge(config, {
                                    "BasicInfo": {
                                        "topic": topic, 
                                        "evaluation_save_path": save_path
                                        }
                                    })
    return cfg_for_topic, survey, references
                            

@hydra.main(config_path="../config", config_name="evaluate_baseline", version_base=None)
def main(config):
    config = merge_with_default_survey_config(config)
    benchmarks = {}

    if config.BasicInfo.user_defiend_benchmarks:
        with open(config.BasicInfo.topic_path, "r", encoding="utf-8") as f:
            benchmarks["user_defined"] = [line.strip() for line in f if line.strip()]

    if config.BasicInfo.AutoSurvey_benchmark:
        benchmarks["AutoSurvey_LLM"] = AUTOSURVEY_TOPICS["AutoSurvey-LLM"]

    if config.BasicInfo.SurveyGen_benchmark:
        for domain, topics in SURVEYGEN_TOPICS.items():
            benchmarks[domain] = topics

    # logger.info(f"total topics: {len(topics)}")
    # logger.info(f"all topics: {topics}")

    baselines = list(config.EvalInfo.baselines)
    save_paths = list(config.EvalInfo.save_paths)
    paper_paths = list(config.EvalInfo.paper_paths)

    if len(baselines) != len(save_paths) or len(baselines) != len(paper_paths):
        logger.error("The number of baselines and save paths and paper_paths must be the same.")
        raise ValueError("The number of baselines and save paths and paper_paths must be the same.")

    logger.info(f"baseline num: {len(baselines)}")
    for baseline, save_path, paper_path in zip(baselines, save_paths, paper_paths):
        logger.info(f"Evaluating baseline: {baseline}")

        write_header(save_path)
        for domain, topics in benchmarks.items():
            logger.info(f"Evaluating domain: {domain}")
            write_domain_header(save_path, domain)

            domain_results = []

            for topic in topics:
                logger.info(f"Evaluating {baseline} output for topic: {topic}")
                
                try:
                    if "autosurvey" in baseline.lower():
                        cfg_for_topic, survey, references_list = get_AutoSurvey_cfg(config, domain, topic, save_path, paper_path)
                    elif "deepsurvey" in baseline.lower():
                        cfg_for_topic, survey, references_list = get_DeepSurvey_cfg(config, domain, topic, save_path, paper_path)
                    elif "human" in baseline.lower():
                        cfg_for_topic, survey, references_list = get_Human_cfg(config, domain, topic, save_path, paper_path)
                    else:
                        logger.error(f"Baseline {baseline} not recognized.")
                        raise ValueError(f"Baseline {baseline} not recognized.")
                except FileNotFoundError:
                    logger.error(f"Survey file for topic {topic} not found. Skipping.")
                    with open(save_path, 'a') as f:
                        f.write(f"ERROR: topic {topic} file not found.\n")
                    continue

                logger.info("Starting Deep Survey Pipeline")
                logger.yaml(OmegaConf.to_container(cfg_for_topic, resolve=True))

                logger.info("Initializing Work Collector...")
                work_collector = WorkCollector(cfg_for_topic)

                # initialize the WorkAnalyzer
                logger.info("Initializing Work Analyzer...")
                work_analyzer = WorkAnalyzer(cfg_for_topic, work_collector)

                logger.info("Initializing Survey Judge...")
                survey_judge = Judge(cfg_for_topic, work_analyzer)

                eval_result, eval_reason = survey_judge.evaluate(survey, references_list)
                write_result(save_path, topic, eval_result, eval_reason)
                domain_results.append(eval_result)
                
            logger.info("Calculating Average Evaluation Results Across All Topics:")
            write_domain_result(save_path, domain, domain_results)



    

if __name__ == "__main__":
    main()
