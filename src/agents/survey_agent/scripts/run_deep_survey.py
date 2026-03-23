import hydra
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.rich_logger import get_logger
from omegaconf import OmegaConf
from modules.work_collector import WorkCollector
from modules.database import Database
from modules.work_analyzer import WorkAnalyzer
from modules.survey_generator import SurveyGenerator
from modules.judge import Judge

from utils.file_utils import write_domain_header, write_topic_header, write_result, write_domain_result

logger = get_logger("Deep Survey")


def _merge_global_survey_config(config, survey_cfg):
    api_mapping = {
        "llm_api_key": "APIInfo.llm_api_key",
        "llm_api_base_url": "APIInfo.llm_api_base_url",
        "llm_model_name": "APIInfo.llm_model_name",
        "llm_max_context_length": "APIInfo.llm_max_context_length",
        "batch_chat_agent_worker": "APIInfo.batch_chat_agent_worker",
        "semantic_scholar_api_key": "APIInfo.semantic_scholar_api_key",
        "low_flow_mode": "APIInfo.low_flow_mode",
    }
    if hasattr(survey_cfg, "api"):
        api_cfg = survey_cfg.api
        for key, target in api_mapping.items():
            if hasattr(api_cfg, key):
                OmegaConf.update(config, target, getattr(api_cfg, key), merge=True)

    if hasattr(survey_cfg, "output"):
        output_cfg = survey_cfg.output
        output_mapping = {
            "base_dir": "BasicInfo.base_dir",
            "save_path": "BasicInfo.save_path",
            "save_json_path": "BasicInfo.save_json_path",
            "evaluation_save_path": "BasicInfo.evaluation_save_path",
        }
        for key, target in output_mapping.items():
            if hasattr(output_cfg, key):
                OmegaConf.update(config, target, getattr(output_cfg, key), merge=True)

    if hasattr(survey_cfg, "topic"):
        current_topic = str(OmegaConf.select(config, "BasicInfo.topic", default="") or "").strip()
        if not current_topic:
            OmegaConf.update(config, "BasicInfo.topic", str(survey_cfg.topic), merge=True)

    if hasattr(survey_cfg, "modules"):
        modules_cfg = survey_cfg.modules
        module_container = OmegaConf.to_container(modules_cfg, resolve=True)
        if isinstance(module_container, dict):
            for module_name, module_value in module_container.items():
                OmegaConf.update(
                    config,
                    f"ModuleInfo.{module_name}",
                    module_value,
                    merge=True,
                )

    return config


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
        intra_analysis_results, inter_analysis_results, collected_papers #YZY MODIFY from inter_cluster_analysis
    )
    survey_generator.log_outline(outline)
    logger.info("Survey generation completed.")
    logger.info("Drafting survey content...")
    # generate draft
    draft = survey_generator.draft_survey(
        intra_analysis_results, inter_analysis_results, outline #YZY MODIFY from inter_cluster_analysis
    )
    logger.info("Survey drafting completed.")

    # if config.BasicInfo.debug:
    #     logger.info(f'DRAFT: {draft}')

    logger.info("Reviewing and revising survey draft...")
    draft = survey_generator.review_and_revise_survey(draft, outline)
    logger.info("Reviewing and revising survey completed.")

    # print(draft)
    logger.info("Survey drafting completed.")
    logger.info("Refining survey draft...")
    survey, references = survey_generator.refine_draft(draft)
    survey_generator.save_survey(survey, references)
    logger.info("Survey refinement completed.")

    logger.info("Evaluating survey...")
    result = judge.evaluate(survey, references)
    logger.info("Survey evaluation completed.")

    return result




@hydra.main(config_path="../config", config_name="deep_survey_xiaomi", version_base=None)
def main(config):
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "..", ".."))
        from src.config import load_config, get_survey_config
        load_config()
        survey_cfg = get_survey_config()
        config = _merge_global_survey_config(config, survey_cfg)
    except Exception as e:
        logger.warning(f"Failed to load global config, using local config: {e}")

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

    result = run_pipeline(config, work_collector, database, work_analyzer, survey_generator, survey_judge)
    write_result(config.BasicInfo.evaluation_save_path, config.BasicInfo.topic, result)
    

if __name__ == "__main__":
    main()
