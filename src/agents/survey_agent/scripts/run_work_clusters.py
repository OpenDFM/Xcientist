import hydra
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.rich_logger import get_logger
from omegaconf import OmegaConf
from modules.work_collector import WorkCollector
from modules.work_analyzer import WorkAnalyzer
from modules.database import Database
from modules.survey_generator import SurveyGenerator
from modules.judge import Judge

import re

logger = get_logger("Deep Survey")


def run_pipeline(config, work_collector, database, work_analyzer, survey_generator, judge):
    # step 1: related work collection
    logger.info("Collecting related work...")

    # collect seed papers
    seed_paper_ids = work_collector.collect_seed_papers(config.BasicInfo.topic)
    # seed_paper_ids = work_collector.collect_seed_papers_debug()
    logger.info(f"Collected seed paper IDs: {seed_paper_ids}")

    # expand seed papers by reference and citation
    logger.info("Expanding seed papers by reference and citation...")
    expanded_paper_ids = work_collector.expand_seed_papers_by_reference_and_citation(
        seed_paper_ids
    )
    if config.BasicInfo.debug:
        logger.info(f"Expanded paper IDs: {expanded_paper_ids}")
        
    logger.info("Building paper embedding database...")
    database.build_with_graph()
    logger.info("Paper embedding database built.")

    # step 2: comprehend papers
    logger.info("Comprehending papers...")

    # deep reading for papers
    collected_papers = seed_paper_ids + expanded_paper_ids
    logger.info(f"Total papers to read: {len(collected_papers)}")
    err_papers = work_analyzer.read_papers_and_write_keynotes(collected_papers)
    logger.info("Deep reading completed.")
    if err_papers:
        logger.warning(f"Some papers failed to read {len(err_papers)} papers after retries")
        collected_papers = [pid for pid in collected_papers if pid not in err_papers]
        logger.info(f"Proceeding with {len(collected_papers)} successfully read papers.")
    

    # clustering
    logger.info("Clustering papers...")
    clustering_result = work_analyzer.cluster_papers(collected_papers)
    logger.info(f"Clustering completed. {len(clustering_result)} clusters formed.")
    work_analyzer.log_clusters(clustering_result)

    # intra-cluster analysis
    logger.info(f"Starting intra-cluster analysis...")
    intra_analysis_results = work_analyzer.intra_cluster_analysis(clustering_result)
    work_analyzer.log_intra_cluster_analysis(intra_analysis_results)
    logger.info("Intra-cluster analysis completed.")

    # inter-cluster analysis
    logger.info(f"Starting inter-cluster analysis...")
    inter_analysis_results = work_analyzer.inter_cluster_analysis(
        intra_analysis_results
    )
    work_analyzer.log_inter_cluster_analysis(inter_analysis_results)
    logger.info("Inter-cluster analysis completed.")

    # survey generation
    logger.info("Generating survey...")
    # generate outline
    outline = survey_generator.generate_outline(
        intra_analysis_results, inter_analysis_results, collected_papers
    )
    survey_generator.log_outline(outline)
    logger.info("Survey generation completed.")
    logger.info("Drafting survey content...")

    # generate draft
    draft = survey_generator.draft_survey(
        intra_analysis_results, inter_analysis_results, outline
    )

    # if config.BasicInfo.debug:
    #     logger.info(f'DRAFT: {draft}')

    # print(draft)
    logger.info("Survey drafting completed.")
    logger.info("Refining survey draft...")
    survey, references = survey_generator.refine_draft(draft)
    survey_generator.save_survey(survey, references)
    logger.info("Survey refinement completed.")

    # logger.info("Evaluating survey...")
    # results = judge.evaluate(survey, references)
    # logger.info("Survey evaluation completed.")





@hydra.main(config_path="../config", config_name="deep_survey_more_ref", version_base=None)
def main(config):
    logger.info("Starting Deep Survey Pipeline")
    logger.yaml(OmegaConf.to_container(config, resolve=True))

    # initialize the WorkCollector
    logger.info("Initializing Work Collector...")
    work_collector = WorkCollector(config)

    logger.info("Initializing Paper Database...")
    database = Database(config, work_collector)

    # initialize the WorkAnalyzer
    logger.info("Initializing Work Analyzer...")
    work_analyzer = WorkAnalyzer(config, work_collector)

    # initialize the SurveyGenerator
    logger.info("Initializing Survey Generator...")
    survey_generator = SurveyGenerator(config, work_analyzer, database)

    logger.info("Initializing Survey Judge...")
    survey_judge = Judge(config, work_analyzer)

    run_pipeline(config, work_collector, database, work_analyzer, survey_generator, survey_judge)
    # import json
    # with open('./outputs/multi Agent/test_multiAgent.json', 'r') as f:
    #     paper = json.load(f)
    
    # survey_judge.evaluate(paper['survey'], paper['references'])

if __name__ == "__main__":
    main()
