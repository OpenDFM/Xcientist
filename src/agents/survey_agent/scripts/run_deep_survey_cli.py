#!/usr/bin/env python3
"""
Deep Survey CLI Wrapper

This script provides a command-line interface for running the Deep Survey pipeline
with parameters specified directly on the command line, overriding the YAML config.

Usage:
    python run_deep_survey_cli.py --topic "Your Topic" --output_path "./outputs/result"
    
    # Or with additional overrides
    python run_deep_survey_cli.py --topic "LLM Agent" --output_path "./outputs/test" \
        --api_key "your-api-key" --api_base_url "https://api.example.com" \
        --max_seed_paper_num 10 --debug

For bash usage, see run_deep_survey_cli.sh
"""

import argparse
import sys
import os
import subprocess

# Get the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Deep Survey with custom parameters from command line',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with topic and output path
    python run_deep_survey_cli.py --topic "LLM Agent Memory System" --output_path "./outputs/my-survey"
    
    # With API configuration
    python run_deep_survey_cli.py --topic "Vision Transformers" \\
        --api_key "your-key" --api_base_url "https://api.example.com" \\
        --output_path "./outputs/vit"
    
    # With processing parameters
    python run_deep_survey_cli.py --topic "Graph Neural Networks" \\
        --max_seed_paper_num 20 --reference_graph_depth 2 \\
        --output_path "./outputs/gnn"
    
    # Enable debug mode
    python run_deep_survey_cli.py --topic "My Topic" --debug True --output_path "./outputs/debug"
        """
    )
    
    # Required parameters
    parser.add_argument('--topic', type=str, required=True,
                        help='Survey topic (required)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output path for the survey results')
    
    # API Configuration (BasicInfo + APIInfo)
    parser.add_argument('--api_key', type=str, default=None,
                        help='LLM API key (overrides config)')
    parser.add_argument('--api_base_url', type=str, default=None,
                        help='LLM API base URL (overrides config)')
    parser.add_argument('--model', type=str, default=None,
                        help='LLM model name (overrides config)')
    parser.add_argument('--semantic_scholar_api_key', type=str, default=None,
                        help='Semantic Scholar API key (overrides config)')
    
    # BasicInfo parameters
    parser.add_argument('--base_dir', type=str, default=None,
                        help='Base directory for the project')
    parser.add_argument('--cache_path', type=str, default=None,
                        help='Cache path for database')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Full save path for the survey markdown')
    parser.add_argument('--save_json_path', type=str, default=None,
                        help='Full save path for the survey JSON')
    parser.add_argument('--evaluation_save_path', type=str, default=None,
                        help='Path to save evaluation results')
    parser.add_argument('--debug', type=str, default=None,
                        help='Debug mode (True/False)')
    parser.add_argument('--error_conservatism_mode', type=str, default=None,
                        help='Error conservatism mode (True/False)')
    
    # WorkCollector parameters
    parser.add_argument('--max_seed_paper_num', type=int, default=None,
                        help='Maximum number of seed papers')
    parser.add_argument('--reference_graph_depth', type=int, default=None,
                        help='Reference graph depth for paper expansion')
    parser.add_argument('--related_work_top_k', type=int, default=None,
                        help='Top K related works to retrieve')
    parser.add_argument('--use_seed_filter_llm', type=str, default=None,
                        help='Use LLM to filter seed papers (True/False)')
    parser.add_argument('--llm_seed_threshold', type=int, default=None,
                        help='LLM seed filter threshold')
    
    # Database parameters
    parser.add_argument('--default_top_k', type=int, default=None,
                        help='Default top K for database retrieval')
    
    # WorkAnalyzer parameters
    parser.add_argument('--abstract_only_mode', type=str, default=None,
                        help='Use abstract only mode (True/False)')
    parser.add_argument('--clustering_temperature', type=float, default=None,
                        help='Clustering temperature')
    
    # SurveyGenerator parameters
    parser.add_argument('--include_initial_analysis', type=str, default=None,
                        help='Include initial analysis in survey (True/False)')
    parser.add_argument('--include_relation_graph', type=str, default=None,
                        help='Include relation graph (True/False)')
    parser.add_argument('--include_relation_table', type=str, default=None,
                        help='Include relation table (True/False)')
    
    # Judge parameters
    parser.add_argument('--judge_api_key', type=str, default=None,
                        help='Judge LLM API key')
    parser.add_argument('--judge_api_base_url', type=str, default=None,
                        help='Judge LLM API base URL')
    parser.add_argument('--judge_model', type=str, default=None,
                        help='Judge model name')
    
    # Config file option
    parser.add_argument('--config', type=str, default='deep_survey_fast',
                        help='Base config file name (without .yaml extension)')
    
    return parser.parse_args()


def convert_to_hydra_override(key, value):
    """Convert argument key-value to Hydra override format."""
    # Map argument names to config keys
    key_mapping = {
        # BasicInfo
        'topic': 'BasicInfo.topic',
        'base_dir': 'BasicInfo.base_dir',
        'cache_path': 'BasicInfo.cache_path',
        'save_path': 'BasicInfo.save_path',
        'save_json_path': 'BasicInfo.save_json_path',
        'evaluation_save_path': 'BasicInfo.evaluation_save_path',
        'debug': 'BasicInfo.debug',
        'error_conservatism_mode': 'BasicInfo.error_conservatism_mode',
        
        # APIInfo
        'api_key': 'APIInfo.llm_api_key',
        'api_base_url': 'APIInfo.llm_api_base_url',
        'model': 'APIInfo.llm_model_name',
        'semantic_scholar_api_key': 'APIInfo.semantic_scholar_api_key',
        
        # WorkCollector
        'max_seed_paper_num': 'ModuleInfo.WorkCollector.max_seed_paper_num',
        'reference_graph_depth': 'ModuleInfo.WorkCollector.reference_graph_depth',
        'related_work_top_k': 'ModuleInfo.WorkCollector.related_work_top_k',
        'use_seed_filter_llm': 'ModuleInfo.WorkCollector.use_seed_filter_LLM',
        'llm_seed_threshold': 'ModuleInfo.WorkCollector.LLM_seed_threshold',
        
        # Database
        'default_top_k': 'ModuleInfo.Database.default_top_k',
        
        # WorkAnalyzer
        'abstract_only_mode': 'ModuleInfo.WorkAnalyzer.abstract_only_mode',
        'clustering_temperature': 'ModuleInfo.WorkAnalyzer.clustering_temperature',
        
        # SurveyGenerator
        'include_initial_analysis': 'ModuleInfo.SurveyGenerator.include_initial_analysis',
        'include_relation_graph': 'ModuleInfo.SurveyGenerator.include_relation_graph',
        'include_relation_table': 'ModuleInfo.SurveyGenerator.include_relation_table',
        
        # Judge
        'judge_api_key': 'ModuleInfo.Judge.judge_llm_api_key',
        'judge_api_base_url': 'ModuleInfo.Judge.judge_llm_api_base_url',
        'judge_model': 'ModuleInfo.Judge.model',
    }
    
    config_key = key_mapping.get(key, key)
    
    # Handle boolean and numeric values
    if isinstance(value, bool):
        return f"{config_key}={str(value).lower()}"
    elif isinstance(value, str):
        if value.lower() in ('true', 'false'):
            return f"{config_key}={value.lower()}"
        elif value.startswith('./') or value.startswith('/') or '=' in value:
            return f'{config_key}="{value}"'
        else:
            return f'{config_key}="{value}"'
    else:
        return f"{config_key}={value}"


def main():
    """Main function to build and run the command."""
    args = parse_args()
    
    # Build Hydra overrides
    overrides = []
    
    # Process all arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            override = convert_to_hydra_override(key, value)
            overrides.append(override)
    
    # Set default output_path if not provided
    if args.output_path and 'BasicInfo.save_path' not in str(overrides):
        # Create default save paths based on output_path
        safe_topic = args.topic.replace(' ', '_').replace('/', '_')
        default_save_path = os.path.join(args.output_path, f"{safe_topic}.md")
        default_save_json_path = os.path.join(args.output_path, f"{safe_topic}.json")
        default_eval_path = os.path.join(args.output_path, f"{safe_topic}_eval.txt")
        
        overrides.append(f'BasicInfo.save_path="{default_save_path}"')
        overrides.append(f'BasicInfo.save_json_path="{default_save_json_path}"')
        overrides.append(f'BasicInfo.evaluation_save_path="{default_eval_path}"')
        overrides.append(f'BasicInfo.output_base_dir="{args.output_path}"')
    
    # Set default base_dir if not provided
    if not any('BasicInfo.base_dir' in o for o in overrides):
        overrides.append(f'BasicInfo.base_dir="{PROJECT_ROOT}"')
    
    # Build the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        os.path.join(SCRIPT_DIR, 'run_deep_survey.py'),
        f'--config-name={args.config}',
    ]
    cmd.extend(overrides)
    
    # Print information
    print("=" * 60)
    print("Deep Survey CLI")
    print("=" * 60)
    print(f"Topic: {args.topic}")
    print(f"Config: {args.config}.yaml")
    print("\nCommand-line overrides:")
    for override in overrides:
        # Mask sensitive information
        if 'api_key' in override.lower():
            print(f"  {override.split('=')[0]}=[HIDDEN]")
        else:
            print(f"  {override}")
    print("=" * 60)
    print()
    
    # Run the command
    print(f"Running: {' '.join(cmd[:3])} ...")
    print()
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
