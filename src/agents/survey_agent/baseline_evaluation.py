import hydra
from utils.rich_logger import get_logger
from omegaconf import OmegaConf
from modules.work_collector import WorkCollector
from modules.work_analyzer import WorkAnalyzer
from modules.survey_generator import SurveyGenerator
from modules.survey_generator import SurveyGenerator
from modules.judge import Judge

import json

logger = get_logger("baseline Eval")
        

def get_AutoSurvey_cfg(config, topic, save_path, paper_path):
    with open(paper_path + f"/{topic}/exp_1/{topic}.json" , "r") as f:
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
                            

@hydra.main(config_path="config", config_name="evaluate_baseline", version_base=None)
def main(config):
    with open(config.BasicInfo.topic_path, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]

    scores = {
        "Core_Quality": 0,
        "Writing_Quality": 0,
        "Content_Depth": 0,
        "Citation_Recall": 0,
        "Citation_Precision": 0
    }

    if len(config.EvalInfo.baselines) != len(config.EvalInfo.save_paths):
        logger.error("The number of baselines and save paths must be the same.")
        raise ValueError("The number of baselines and save paths must be the same.")

    for baseline, save_path, paper_path in zip(config.EvalInfo.baselines, config.EvalInfo.save_paths, config.EvalInfo.paper_paths):
        for topic in topics:
            logger.info(f"Evaluating AutoSurvey output for topic: {topic}")
            
            if baseline == "AutoSurvey":
                cfg_for_topic, survey, references_list = get_AutoSurvey_cfg(config, topic, save_path, paper_path)
            else:
                logger.error(f"Baseline {baseline} not recognized.")
                raise ValueError(f"Baseline {baseline} not recognized.")

            logger.info("Starting Deep Survey Pipeline")
            logger.yaml(OmegaConf.to_container(cfg_for_topic, resolve=True))

            logger.info("Initializing Work Collector...")
            work_collector = WorkCollector(cfg_for_topic)

            # initialize the WorkAnalyzer
            logger.info("Initializing Work Analyzer...")
            work_analyzer = WorkAnalyzer(cfg_for_topic, work_collector)

            logger.info("Initializing Survey Judge...")
            survey_judge = Judge(cfg_for_topic, work_analyzer)

            with open(save_path, 'a') as f:
                f.write(f"Evaluation Result for Topic: {topic}\n")
                f.write('----------------------------------------\n')

            eval_results = survey_judge.evaluate(survey, references_list)
            for key in scores.keys():
                scores[key] += eval_results[key]

        num_topics = len(topics)
        logger.info("Average Evaluation Results Across All Topics:")
        for key in scores.keys():
            avg_score = scores[key] / num_topics
            logger.info(f"{key}: {avg_score:.4f}")
        
        with open(save_path, 'a') as f:
            f.write("Average Evaluation Results Across All Topics:\n")
            for key in scores.keys():
                avg_score = scores[key] / num_topics
                f.write(f"{key}: {avg_score:.4f}\n")



    

if __name__ == "__main__":
    main()
