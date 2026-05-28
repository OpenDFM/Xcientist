import sys
import os
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

_SCRIPT_PATH = Path(__file__).resolve()
_REPO_ROOT = _SCRIPT_PATH.parents[4]
for _env_path in (_REPO_ROOT / ".env", _REPO_ROOT / "src" / "config" / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path, override=False)

if not os.environ.get("OPENAI_API_BASE"):
    base_url = str(os.environ.get("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if base_url:
        if base_url.endswith("/chat/completions"):
            os.environ["OPENAI_API_BASE"] = base_url
        else:
            os.environ["OPENAI_API_BASE"] = f"{base_url}/chat/completions"

import hydra
from utils.rich_logger import get_logger
from omegaconf import OmegaConf
from modules.work_collector import WorkCollector
from modules.database import Database
from modules.work_analyzer import WorkAnalyzer
from modules.survey_generator import SurveyGenerator
from modules.judge import Judge
from utils.config_utils import merge_with_default_survey_config
from modules.code_collector import CodeCollector, CodeAnalyzer
from modules.code_report_generator import CodeReportGenerator

from utils.file_utils import write_domain_header, write_topic_header, write_result, write_domain_result, save_analysis_artifacts

logger = get_logger("Deep Survey")


def run_pipeline(config, work_collector, database, work_analyzer, survey_generator, judge, code_collector, code_analyzer, code_report_generator):
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

    logger.info(f"valid_paper_ids: {database.valid_paper_ids}")

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

    code_report = None
    env_report = None
    if config.ModuleInfo.SurveyGenerator.include_code_report:
        logger.info("Building code report among papers...")
        paper_mainfests = code_analyzer.execute(collected_papers)

        env_report = code_report_generator.generate_framework_env_report(paper_mainfests = paper_mainfests, topic = config.BasicInfo.topic)
        code_report = code_report_generator.generate_report(papers = paper_mainfests, topic = config.BasicInfo.topic)
        code_report_generator.save_report(code_report, env_report)

    relation_graph = None
    relation_table = None
    intra_analysis_results = []
    inter_analysis_results = ""

    if config.ModuleInfo.SurveyGenerator.include_relation_graph:
        logger.info("Generating relation graph among papers...")
        relation_graph = work_analyzer.build_relationship_graphs(clustering_result)
        logger.info("Relation graph generation completed.")

    if config.ModuleInfo.SurveyGenerator.include_relation_table:
        logger.info("Generating relation analysis table among papers...")
        relation_table = work_analyzer.generate_cluster_tables(clustering_result)
        logger.info("Relation analysis table generation completed.")
        try:
            logger.info(f"Relation tables:\n{work_analyzer.format_analysis_table(relation_table)}")
        except Exception as e:
            logger.warning(f"Failed to format relation tables: {e}")

    if config.ModuleInfo.SurveyGenerator.include_initial_analysis:
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

    logger.info("Saving analysis artifacts...")
    save_analysis_artifacts(config.BasicInfo.save_path, config.BasicInfo.topic, relation_graph, relation_table, intra_analysis_results, inter_analysis_results, logger)

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
    draft = survey_generator.review_and_revise_survey_in_parts(draft, outline, code_report, env_report)
    logger.info("Reviewing and revising survey completed.")

    # print(draft)
    logger.info("Survey drafting completed.")
    logger.info("Refining survey draft...")
    survey, references = survey_generator.refine_draft(draft, code_report, env_report)
    survey_generator.save_survey(survey, references)
    logger.info("Survey refinement completed.")

    logger.info("Evaluating survey...")
    result, reason = judge.evaluate(survey, references)
    logger.info("Survey evaluation completed.")

    return result, reason




@hydra.main(config_path="../config", config_name="deep_survey_fast", version_base=None)
def main(config):
    config = merge_with_default_survey_config(config)
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

    logger.info("Initializing Code Modules...")
    code_collector = CodeCollector(config)
    code_analyzer = CodeAnalyzer(config, code_collector=code_collector, work_collector=work_collector)
    code_report_generator = CodeReportGenerator(config, work_collector=work_collector ,code_collector=code_collector, code_analyzer=code_analyzer)

    result, reason = run_pipeline(config, work_collector, database, work_analyzer, survey_generator, survey_judge, code_collector, code_analyzer, code_report_generator)
    write_result(config.BasicInfo.evaluation_save_path, config.BasicInfo.topic, result, reason)
    

if __name__ == "__main__":
    main()
