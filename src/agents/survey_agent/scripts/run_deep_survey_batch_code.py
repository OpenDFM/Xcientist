import hydra
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.rich_logger import get_logger
from omegaconf import OmegaConf
from modules.work_collector import WorkCollector
from modules.database import Database
from modules.work_analyzer import WorkAnalyzer
from modules.code_report_generator import CodeReportGenerator
from modules.survey_generator import SurveyGenerator
from topics.Benchmark_topics import SURVEYGEN_TOPICS, AUTOSURVEY_TOPICS, SUBTEST_TOPICS
from utils.config_utils import merge_with_default_survey_config
from utils.file_utils import write_domain_header, write_topic_header, write_result, write_domain_result, write_header

from modules.judge import Judge
from modules.code_collector import CodeCollector, CodeAnalyzer

logger = get_logger("Deep Survey Batch")

def run_pipeline_batch(config, work_collector, database, work_analyzer, survey_generator, judge, code_collector, code_analyzer, code_report_generator):
    try:
        results = []
        reasons = []
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
        # collected_papers = list(set.union(set(seed_paper_ids),set(expanded_paper_ids)))
        # import json
        # with open("/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/outputs/xiaomi-batch-0307-test-refinement2-whole-review-revise/user_defined/LLMs-based_Agents.json", "r") as f:
        #     survey_dict = json.load(f)
        # collected_papers = survey_dict.get("references")

        logger.info(f"Total papers to read: {len(collected_papers)}")
        err_papers = work_analyzer.read_papers_and_write_keynotes(collected_papers)
        logger.info("Deep reading completed.")
        if err_papers:
            logger.warning(f"Some papers failed to read {len(err_papers)} papers after retries")
            collected_papers = [pid for pid in collected_papers if pid not in err_papers]
            logger.info(f"Proceeding with {len(collected_papers)} successfully read papers.")

        if config.ModuleInfo.SurveyGenerator.include_code_report:
            paper_mainfests = code_analyzer.execute(collected_papers)

            env_report = code_report_generator.generate_framework_env_report(paper_mainfests = paper_mainfests, topic = "LLMs-based Agents")
            code_report = code_report_generator.generate_report(papers = paper_mainfests, topic = config.BasicInfo.topic)
            code_report_generator.save_report(code_report, env_report)
        
        # clustering
        logger.info("Clustering papers...")
        clustering_result = work_analyzer.cluster_papers(collected_papers)
        logger.info(f"Clustering completed. {len(clustering_result)} clusters formed.")
        work_analyzer.log_clusters(clustering_result)

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
        # Pass code_report to draft_survey if available
        draft = survey_generator.draft_survey(
            intra_analysis_results, inter_analysis_results, outline,
            code_report=code_report if config.ModuleInfo.SurveyGenerator.include_code_report else None
        )

        # if config.BasicInfo.debug:
        #     logger.info(f'DRAFT: {draft}')

        logger.info("Reviewing and revising survey draft...")
        draft = survey_generator.review_and_revise_survey_in_parts(draft, outline, code_report)
        logger.info("Reviewing and revising survey completed.")

        # print(draft)
        logger.info("Survey drafting completed.")
        logger.info("Refining survey draft...")
        survey, references = survey_generator.refine_draft(draft, code_report)
        survey_generator.save_survey(survey, references)
        logger.info("Survey refinement completed.")

        logger.info("Evaluating survey...")
        results, reasons = judge.evaluate(survey, references)
        logger.info("Survey evaluation completed.")
    except Exception as e:
        logger.error(f"Error occurred during pipeline execution: {e}")
        return False, None, None
    return True, results, reasons

@hydra.main(config_path="../config", config_name="deep_survey_batch_test", version_base=None)
def main(config):
    config = merge_with_default_survey_config(config)
    logger.info("Starting Deep Survey Pipeline")
    logger.yaml(OmegaConf.to_container(config, resolve=True))

    topics_file = config.BasicInfo.topic_path
    benchmarks = {}

    if config.BasicInfo.user_defiend_benchmarks:
        with open(topics_file, "r", encoding="utf-8") as f:
            benchmarks['user_defined'] = [line.strip() for line in f if line.strip()]

    if config.BasicInfo.AutoSurvey_benchmark:
        for domain, domain_topics in AUTOSURVEY_TOPICS.items():
            benchmarks[domain] = domain_topics
    
    if config.BasicInfo.SurveyGen_benchmark:
        for domain, domain_topics in SURVEYGEN_TOPICS.items():
            benchmarks[domain] = domain_topics

    if config.BasicInfo.sub_benchmark_test:
        for domain, domain_topics in SUBTEST_TOPICS.items():
            benchmarks[domain] = domain_topics
    
    write_header(config.BasicInfo.evaluation_save_path)
    for domain, topics in benchmarks.items():
        results = []
        logger.info(f"=== Starting domain: {domain} with {len(topics)} topics ===")
        write_domain_header(config.BasicInfo.evaluation_save_path, domain)

        for topic in topics:
            logger.info(f"=== Running pipeline for topic: {topic} ===")

            for attempt in range(config.BasicInfo.topic_max_retry):
                logger.info(f"Attempt {attempt+1} for topic: {topic}")
                cfg_for_topic = OmegaConf.merge(config, {
                                                        "BasicInfo": {
                                                                "topic": "LLMs-based_Agents", 
                                                                "save_path": f"{config.BasicInfo.output_base_dir}/{domain}",
                                                            }
                                                        })

                work_collector = WorkCollector(cfg_for_topic)
                database = Database(cfg_for_topic, work_collector)
                work_analyzer  = WorkAnalyzer(cfg_for_topic, work_collector)
                survey_generator = SurveyGenerator(cfg_for_topic, work_analyzer, database)
                survey_judge = Judge(cfg_for_topic, work_analyzer)
                code_collector = CodeCollector(config)
                code_analyzer = CodeAnalyzer(config, code_collector=code_collector, work_collector=work_collector)
                code_report_generator = CodeReportGenerator(config, work_collector=work_collector ,code_collector=code_collector, code_analyzer=code_analyzer)

                finished, eval_results, eval_reasons = run_pipeline_batch(cfg_for_topic, work_collector, database, work_analyzer, survey_generator, survey_judge, code_collector, code_analyzer, code_report_generator)
                if finished:
                    logger.info(f"Pipeline completed successfully for topic: {topic}")
                    write_result(config.BasicInfo.evaluation_save_path, topic, eval_results, eval_reasons)
                    results.append(eval_results)
                    break
                else:
                    logger.warning(f"Pipeline did not finish successfully for topic: {topic} on attempt {attempt+1}")
                    logger.info("Retrying...")
        
        write_domain_result(config.BasicInfo.evaluation_save_path, domain, results)
        logger.info(f"=== Completed domain: {domain} ===")


    logger.info("All topics processed.")

if __name__ == "__main__":
    main()
