import hydra
from utils.rich_logger import get_logger
from omegaconf import OmegaConf
from modules.work_collector import WorkCollector
from modules.work_analyzer import WorkAnalyzer
from modules.survey_generator import SurveyGenerator

logger = get_logger("Deep Survey")


def run_pipeline(config, work_collector, work_analyzer, survey_generator):
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
        
    # step 2: comprehend papers
    logger.info("Comprehending papers...")

    # deep reading for papers
    collected_papers = seed_paper_ids + expanded_paper_ids
    logger.info(f"Total papers to read: {len(collected_papers)}")
    work_analyzer.read_papers_and_write_keynotes(collected_papers)
    logger.info("Deep reading completed.")

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

    if config.BasicInfo.debug:
        logger.info(f'DRAFT: {draft}')

    # print(draft)
    logger.info("Survey drafting completed.")
    logger.info("Refining survey draft...")
    survey = survey_generator.refine_draft(draft)
    survey_generator.save_survey(survey)
    logger.info("Survey refinement completed.")


@hydra.main(config_path="config", config_name="deep_survey", version_base=None)
def main(config):
    logger.info("Starting Deep Survey Pipeline")
    logger.yaml(OmegaConf.to_container(config, resolve=True))

    # initialize the WorkCollector
    logger.info("Initializing Work Collector...")
    work_collector = WorkCollector(config)

    # initialize the WorkAnalyzer
    logger.info("Initializing Work Analyzer...")
    work_analyzer = WorkAnalyzer(config, work_collector)

    # initialize the SurveyGenerator
    logger.info("Initializing Survey Generator...")
    survey_generator = SurveyGenerator(config, work_analyzer)

    run_pipeline(config, work_collector, work_analyzer, survey_generator)


if __name__ == "__main__":
    main()
