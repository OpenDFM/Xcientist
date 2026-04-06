#!/usr/bin/env python3
"""
Deep Survey Adapter

Usage:
    python run_deep_survey_adapter.py --workspace /workspace --config /workspace/config/runtime.yaml

Output:
    /workspace/survey/output/survey.md
    /workspace/survey/output/survey.json
    /workspace/survey/output/evaluation.txt
    /workspace/logs/relation_graph.json
    /workspace/logs/relation_table.json
    /workspace/logs/clustering_result.json
    /workspace/logs/draft.json
    /workspace/logs/deep_survey.log
"""

import argparse
import sys
import os
import json
import logging

# Setup path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Clean proxy environment variables to ensure direct connection to internal API
# This is critical when the adapter is called via SSH or subprocess with inherited env
for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(key, None)
os.environ["no_proxy"] = "58.210.177.113,localhost,127.0.0.1"
os.environ["NO_PROXY"] = "58.210.177.113,localhost,127.0.0.1"

import yaml
from omegaconf import OmegaConf
from utils.rich_logger import get_logger
from modules.work_collector import WorkCollector
from modules.database import Database
from modules.work_analyzer import WorkAnalyzer
from modules.survey_generator import SurveyGenerator
from modules.judge import Judge
from utils.file_utils import write_result


def setup_file_logging(log_dir: str, log_filename: str = "deep_survey.log"):
    """
    Configure file logging for all existing loggers.
    This will capture all log messages to a file in addition to console output.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Create a file handler
    file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Define formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger and all existing loggers
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Also add to all loggers that might have been created before this call
    for logger_name in ['Deep Survey', 'WorkAnalyzer', 'SurveyGenerator', 'Judge', 
                        'WorkCollector', 'Database', 'rich_logger', 'app']:
        try:
            logger = logging.getLogger(logger_name)
            # Avoid duplicate handlers
            if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
                logger.addHandler(file_handler)
        except Exception:
            pass
    
    return log_filepath


def save_json(data, filename: str, output_dir: str, logger_instance=None):
    """
    Generic function to save data as JSON file.
    
    Args:
        data: The data to save (will be serialized to JSON)
        filename: The output filename (e.g., "clustering_result.json")
        output_dir: The output directory
        logger_instance: Optional logger instance for logging
    
    Returns:
        The filepath if successful, None otherwise
    """
    if data is None:
        return None
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    if logger_instance:
        logger_instance.info(f"{filename} saved to: {filepath}")
    return filepath


def save_relation_graph(relation_graph, output_dir: str, logger_instance=None):
    """
    Save relation graph to JSON file.
    relation_graph is a dict: {cluster_name: nx.DiGraph}
    """
    if relation_graph is None:
        return None
    
    filepath = os.path.join(output_dir, "relation_graph.json")
    
    # Convert networkx graphs to serializable dict format
    import networkx as nx
    
    serializable_graph = {}
    for cluster_name, g in relation_graph.items():
        if isinstance(g, nx.DiGraph):
            serializable_graph[cluster_name] = {
                'nodes': list(g.nodes()),
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        'type': data.get('type', 'unspecified'),
                        'analysis': data.get('analysis', ''),
                        'raw': data.get('raw', '')
                    }
                    for u, v, data in g.edges(data=True)
                ]
            }
        else:
            # Already in dict format or other type
            serializable_graph[cluster_name] = g
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_graph, f, indent=2, ensure_ascii=False)
    
    if logger_instance:
        logger_instance.info(f"Relation graph saved to: {filepath}")
    return filepath


def save_draft(draft, output_dir: str, logger_instance=None):
    """
    Save draft to JSON file.
    draft structure:
        {
            "section_drafts": section_drafts,
            "full_draft": outcome_draft,
            "title": outline.get("title", ...),
            "outline": outline
        }
    """
    if draft is None:
        return None
    
    filepath = os.path.join(output_dir, "draft.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(draft, f, indent=2, ensure_ascii=False)
    
    if logger_instance:
        logger_instance.info(f"Draft saved to: {filepath}")
    return filepath


def run_pipeline_with_saving(config, work_collector, database, work_analyzer, survey_generator, judge, artifacts_dir, logger_instance):
    """
    Run the deep survey pipeline with immediate artifact saving.
    This is a modified version of run_pipeline that saves intermediate results immediately.
    """
    # step 1: related work collection
    logger_instance.info("Collecting related work...")

    # collect seed papers
    seed_paper_ids = work_collector.collect_seed_papers(config.BasicInfo.topic)
    logger_instance.info(f"Collected seed paper IDs: {seed_paper_ids}")

    # expand seed papers by reference and citation
    logger_instance.info("Expanding seed papers by reference and citation...")
    expanded_paper_ids = work_collector.expand_seed_papers_by_reference_and_citation(
        seed_paper_ids
    )

    if config.BasicInfo.debug:
        logger_instance.info(f"Expanded paper IDs: {expanded_paper_ids}")
        
    logger_instance.info("Building paper embedding database...")
    database.build_with_graph()
    logger_instance.info("Paper embedding database built.")

    logger_instance.info(f"valid_paper_ids: {database.valid_paper_ids}")

    # step 2: comprehend papers
    logger_instance.info("Comprehending papers...")

    # deep reading for papers
    collected_papers = seed_paper_ids + expanded_paper_ids
    logger_instance.info(f"Total papers to read: {len(collected_papers)}")
    err_papers = work_analyzer.read_papers_and_write_keynotes(collected_papers)
    logger_instance.info("Deep reading completed.")
    if err_papers:
        logger_instance.warning(f"Some papers failed to read {len(err_papers)} papers after retries")
        collected_papers = [pid for pid in collected_papers if pid not in err_papers]
        logger_instance.info(f"Proceeding with {len(collected_papers)} successfully read papers.")
    

    # clustering
    logger_instance.info("Clustering papers...")
    clustering_result = work_analyzer.cluster_papers(collected_papers)
    logger_instance.info(f"Clustering completed. {len(clustering_result)} clusters formed.")
    work_analyzer.log_clusters(clustering_result)
    
    # Save clustering result immediately after generation
    if artifacts_dir:
        save_json(clustering_result, "clustering_result.json", artifacts_dir, logger_instance)

    relation_graph = None
    relation_table = None
    intra_analysis_results = []
    inter_analysis_results = ""

    if config.ModuleInfo.SurveyGenerator.include_relation_graph:
        logger_instance.info("Generating relation graph among papers...")
        relation_graph = work_analyzer.build_relationship_graphs(clustering_result)
        logger_instance.info("Relation graph generation completed.")
        
        # Save relation graph immediately after generation
        if artifacts_dir:
            save_relation_graph(relation_graph, artifacts_dir, logger_instance)

    if config.ModuleInfo.SurveyGenerator.include_relation_table:
        logger_instance.info("Generating relation analysis table among papers...")
        relation_table = work_analyzer.generate_cluster_tables(clustering_result)
        logger_instance.info("Relation analysis table generation completed.")
        
        # Save relation table immediately after generation
        if artifacts_dir:
            save_json(relation_table, "relation_table.json", artifacts_dir, logger_instance)
        
        try:
            logger_instance.info(f"Relation tables:\n{work_analyzer.format_analysis_table(relation_table)}")
        except Exception as e:
            logger_instance.warning(f"Failed to format relation tables: {e}")

    if config.ModuleInfo.SurveyGenerator.include_initial_analysis:
        # intra-cluster analysis
        logger_instance.info(f"Starting intra-cluster analysis...")
        intra_analysis_results = work_analyzer.intra_cluster_analysis(clustering_result)
        work_analyzer.log_intra_cluster_analysis(intra_analysis_results)
        logger_instance.info("Intra-cluster analysis completed.")

        # inter-cluster analysis
        logger_instance.info(f"Starting inter-cluster analysis...")
        inter_analysis_results = work_analyzer.inter_cluster_analysis(
            intra_analysis_results
        )
        work_analyzer.log_inter_cluster_analysis(inter_analysis_results)
        logger_instance.info("Inter-cluster analysis completed.")

    # survey generation
    logger_instance.info("Generating survey...")
    # generate outline
    outline = survey_generator.generate_outline(
        intra_analysis_results, inter_analysis_results, collected_papers
    )
    survey_generator.log_outline(outline)
    logger_instance.info("Survey generation completed.")
    logger_instance.info("Drafting survey content...")
    
    # generate draft
    draft = survey_generator.draft_survey(
        intra_analysis_results, inter_analysis_results, outline
    )
    logger_instance.info("Survey drafting completed.")
    
    # Save draft immediately after generation
    if artifacts_dir:
        save_draft(draft, artifacts_dir, logger_instance)

    logger_instance.info("Reviewing and revising survey draft...")
    draft = survey_generator.review_and_revise_survey_in_parts(draft, outline)
    logger_instance.info("Reviewing and revising survey completed.")

    logger_instance.info("Survey drafting completed.")
    logger_instance.info("Refining survey draft...")
    survey, references = survey_generator.refine_draft(draft)
    survey_generator.save_survey(survey, references)
    logger_instance.info("Survey refinement completed.")

    logger_instance.info("Evaluating survey...")
    result, reason = judge.evaluate(survey, references)
    logger_instance.info("Survey evaluation completed.")

    return result, reason


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.isfile(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Create output directories
    output_dir = os.path.join(args.workspace, 'survey', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logs directory for artifacts and logs
    logs_dir = os.path.join(args.workspace, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup file logging
    log_filepath = setup_file_logging(logs_dir, "deep_survey.log")
    
    # Read and modify config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Override output paths
    config_dict.setdefault('BasicInfo', {})
    config_dict['BasicInfo']['save_path'] = f'{output_dir}/survey.md'
    config_dict['BasicInfo']['save_json_path'] = f'{output_dir}/survey.json'
    config_dict['BasicInfo']['evaluation_save_path'] = f'{output_dir}/evaluation.txt'
    config_dict['BasicInfo']['base_dir'] = PROJECT_ROOT
    
    # Convert to OmegaConf
    config = OmegaConf.create(config_dict)
    
    logger = get_logger("Deep Survey")
    logger.info("Starting Deep Survey Pipeline")
    logger.info(f"Log file: {log_filepath}")
    logger.info(f"Artifacts directory: {logs_dir}")
    logger.yaml(OmegaConf.to_container(config, resolve=True))
    
    try:
        # initialize modules
        logger.info("Initializing Work Collector...")
        work_collector = WorkCollector(config)
        
        logger.info("Initializing Paper Database...")
        database = Database(config, work_collector)
        
        logger.info("Initializing Work Analyzer...")
        work_analyzer = WorkAnalyzer(config, work_collector)
        
        logger.info("Initializing Survey Generator...")
        survey_generator = SurveyGenerator(config, work_analyzer, database)
        
        logger.info("Initializing Survey Judge...")
        survey_judge = Judge(config, work_analyzer)
        
        # Run pipeline with immediate artifact saving
        result, reason = run_pipeline_with_saving(
            config, work_collector, database, work_analyzer, survey_generator, survey_judge,
            artifacts_dir=logs_dir,
            logger_instance=logger
        )
        
        # Write result
        write_result(config.BasicInfo.evaluation_save_path, config.BasicInfo.topic, result, reason)
        
        logger.info("Pipeline completed successfully.")
        logger.info(f"All logs saved to: {log_filepath}")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Full traceback saved to: {log_filepath}")
        sys.exit(1)


if __name__ == "__main__":
    main()
