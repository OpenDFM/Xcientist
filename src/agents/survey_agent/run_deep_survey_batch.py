import hydra
from utils.rich_logger import get_logger
from omegaconf import OmegaConf
from modules.work_collector import WorkCollector
from modules.work_analyzer import WorkAnalyzer
from modules.survey_generator import SurveyGenerator

from modules.judge import Judge

logger = get_logger("Deep Survey Batch")

def run_pipeline_batch(config, work_collector, work_analyzer, survey_generator, judge):
    try:
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

        if config.BasicInfo.debug:
            logger.info(f'DRAFT: {draft}')

        # print(draft)
        logger.info("Survey drafting completed.")
        logger.info("Refining survey draft...")
        survey, references = survey_generator.refine_draft(draft)
        survey_generator.save_survey(survey, references)
        logger.info("Survey refinement completed.")

        logger.info("Evaluating survey...")
        judge.evaluate(survey, references)
        logger.info("Survey evaluation completed.")
    except Exception as e:
        logger.error(f"Error occurred during pipeline execution: {e}")
        return False
    return True

@hydra.main(config_path="config", config_name="deep_survey_batch", version_base=None)
def main(config):
    logger.info("Starting Deep Survey Pipeline")
    logger.yaml(OmegaConf.to_container(config, resolve=True))

    topics_file = config.BasicInfo.topic_path
    with open(topics_file, "r", encoding="utf-8") as f:
        topics = [line.strip() for line in f if line.strip()]

    for topic in topics:
        logger.info(f"=== Running pipeline for topic: {topic} ===")

        for attempt in range(config.BasicInfo.topic_max_retry):
            logger.info(f"Attempt {attempt+1} for topic: {topic}")
            cfg_for_topic = OmegaConf.merge(config, {
                                                    "BasicInfo": {
                                                        "topic": topic, 
                                                        "evaluation_save_path": f"{config.BasicInfo.output_base_dir}/{topic.replace(' ', '_')}_evaluation.txt",
                                                        "save_path": f"{config.BasicInfo.output_base_dir}/{topic.replace(' ', '_')}.md",
                                                        "save_json_path": f"{config.BasicInfo.output_base_dir}/{topic.replace(' ', '_')}.json"
                                                        }
                                                    })

            work_collector = WorkCollector(cfg_for_topic)
            work_analyzer  = WorkAnalyzer(cfg_for_topic, work_collector)
            survey_generator = SurveyGenerator(cfg_for_topic, work_analyzer)
            survey_judge = Judge(cfg_for_topic, work_analyzer)

            finished = run_pipeline_batch(cfg_for_topic, work_collector, work_analyzer, survey_generator, survey_judge)
            if finished:
                logger.info(f"Pipeline completed successfully for topic: {topic}")
                break
            else:
                logger.warning(f"Pipeline did not finish successfully for topic: {topic} on attempt {attempt+1}")
                logger.info("Retrying...")


    logger.info("All topics processed.")

if __name__ == "__main__":
    main()
