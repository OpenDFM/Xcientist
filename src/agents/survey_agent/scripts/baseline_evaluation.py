import hydra
import sys
import os
import re
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
from modules.data_manager import DataManager

import json
from pathlib import Path
from typing import List, Dict, Tuple

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

def get_SurveyForge_cfg(config, domain, topic, save_path, paper_path):
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

def get_Lira_cfg(config, domain, topic, save_path, paper_path):
    file_path = Path(paper_path) / f"{topic}.txt"
    ref_path = Path(paper_path) / f"{topic}_ref.json"
    
    logger.info(f"Loading human survey from {file_path}")
    
    if not file_path.exists():
        logger.error(f"Survey file not found: {file_path}")
        logger.error(f"Topic: '{topic}'")
        logger.error(f"Paper path directory: {Path(paper_path)}")
        if Path(paper_path).exists():
            logger.error("Available files:")
            for f in sorted(Path(paper_path).glob("*")):
                if topic.split()[0].lower() in f.name.lower():
                    logger.error(f"  - {f.name}")
        raise FileNotFoundError(f"Survey file not found: {file_path}")
    
    if not ref_path.exists():
        logger.error(f"Reference file not found: {ref_path}")
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        survey = f.read()
    with open(ref_path, 'r', encoding="utf-8") as f:
        ref_dict = json.load(f)
    
    # 转换paper titles到paper_ids
    # 收集所有唯一的titles
    titles = list(ref_dict.values())
    logger.info(f"Converting {len(titles)} paper titles to paper_ids...")
    
    # 批量查询paper信息
    data_manager = DataManager(config)
    title_to_paper = data_manager.get_paper_with_title_batch(titles)
    
    # 构建references列表，只包含paper_id
    references = []
    not_found_titles = []
    for title in titles:
        if title in title_to_paper and title_to_paper[title] is not None:
            paper_info = title_to_paper[title]
            if 'paperId' in paper_info:
                references.append(paper_info['paperId'])
            else:
                not_found_titles.append(title)
        else:
            not_found_titles.append(title)
    
    if not_found_titles:
        logger.warning(f"Could not find {len(not_found_titles)} papers in database out of {len(titles)} total references")
    
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


def extract_refs_from_bib(bib_content: str) -> Dict[str, Dict]:
    """
    从BibTeX格式的参考文献中提取所有引用信息
    
    Args:
        bib_content: BibTeX格式的参考文献内容
        
    Returns:
        Dict: label -> {"title": ..., "paper_id": None} 的映射
    """
    refs_dict = {}
    
    # 改进的正则表达式 - 匹配 @type{content} 直到整个条目结束
    # 支持多行title，条目格式：@article{label,\ntitle={...}\n}
    article_pattern = r'@(\w+)\{([^@]+)\}\s*(?=\n@|\Z)'
    matches = re.findall(article_pattern, bib_content, re.DOTALL)
    
    for entry_type, content in matches:
        # 提取label - 取第一个逗号前的部分
        label_pattern = r'^([^,\s]+)'
        label_match = re.search(label_pattern, content.strip())
        if not label_match:
            continue
        label = label_match.group(1).strip()
        
        # 提取title - 支持多行和嵌套括号
        title_pattern = r'title\s*=\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
        title_match = re.search(title_pattern, content, re.DOTALL)
        if title_match:
            title = ' '.join(title_match.group(1).split())
            refs_dict[label] = {
                "title": title,
                "paper_id": None
            }
    
    return refs_dict


def process_survey_tex(survey: str, label_to_paper: Dict[str, str]) -> Tuple[str, List]:
    """
    处理survey.tex，将\cite{label}替换为[idx]，并返回按引用顺序排列的references
    
    Args:
        survey: survey.tex内容
        label_to_paper: label -> paper_id的映射
        
    Returns:
        (处理后的survey, references_list): references_list中paper_id按首次引用顺序排列
    """
    # 找出所有出现的cite标签及其顺序
    cite_pattern = r'\\cite\{([^}]+)\}'
    all_cites = re.findall(cite_pattern, survey)
    
    # 处理可能的多个引用: \cite{a,b,c}
    cite_order = []
    for cite_group in all_cites:
        labels = [l.strip() for l in cite_group.split(',')]
        for label in labels:
            if label not in cite_order:
                cite_order.append(label)
    
    # 只对在label_to_paper中存在的标签分配索引
    # 这样可以确保references_list的长度和最大索引匹配
    # 注意：索引从1开始，因为judge.py使用 references[index-1] 来访问
    label_to_idx = {}
    references_list = []
    unmatched_labels = []
    
    for label in cite_order:
        if label in label_to_paper and label not in label_to_idx:
            label_to_idx[label] = len(references_list) + 1  # 索引从1开始
            references_list.append(label_to_paper[label])
        elif label not in label_to_paper and label not in unmatched_labels:
            unmatched_labels.append(label)
    
    if unmatched_labels:
        logger.warning(f"Found {len(unmatched_labels)} citations with labels not found in database: {unmatched_labels[:10]}{'...' if len(unmatched_labels) > 10 else ''}")
    
    # 替换survey中的\cite{label}为[idx]，只替换在label_to_idx中的标签
    def replace_cite(match):
        cite_group = match.group(1)
        labels = [l.strip() for l in cite_group.split(',')]
        new_labels = [f"[{label_to_idx[l]}]" if l in label_to_idx else l for l in labels]
        return ','.join(new_labels)
    
    processed_survey = re.sub(cite_pattern, replace_cite, survey)
    
    return processed_survey, references_list


def get_SurveyX_cfg(config, domain, topic, save_path, paper_path):
    """
    加载SurveyX生成的survey
    
    SurveyX的论文存储在 {paper_path}/{topic}/latex/survey.tex
    引用存储在 {paper_path}/{topic}/latex/references.bib (BibTeX格式)
    需要解析BibTeX提取title，然后使用DataManager.get_paper_with_title_batch获取paper_info
    将\cite{label}替换为[idx]格式
    """
    file_path = Path(paper_path) / topic / "latex" / "survey.tex"
    bib_path = Path(paper_path) / topic / "latex" / "references.bib"
    
    logger.info(f"Loading SurveyX survey from {file_path}")
    logger.info(f"Loading SurveyX references from {bib_path}")
    
    # 读取survey正文
    with open(file_path, "r", encoding="utf-8") as f:
        survey = f.read()
    
    # 读取并解析BibTeX参考文献
    with open(bib_path, "r", encoding="utf-8") as f:
        bib_content = f.read()
    
    # 解析BibTeX获取label->title映射
    refs_dict = extract_refs_from_bib(bib_content)
    logger.info(f"Extracted {len(refs_dict)} references from BibTeX")
    
    # 批量查询paper信息
    titles = [ref["title"] for ref in refs_dict.values()]
    data_manager = DataManager(config)
    paper_results = data_manager.get_paper_with_title_batch(titles)
    
    # 构建label -> paper_id的映射
    label_to_paper = {}
    not_found_titles = []
    for label, ref_info in refs_dict.items():
        title = ref_info["title"]
        if title in paper_results and paper_results[title] is not None:
            # 只提取paper_id，而不是整个字典
            paper_info = paper_results[title]
            if 'paperId' in paper_info:
                label_to_paper[label] = paper_info['paperId']
            else:
                # 如果没有paperId字段，记录警告并跳过
                logger.warning(f"Paper found but no paperId field: {title[:50]}...")
                not_found_titles.append(title)
        else:
            not_found_titles.append(title)
    
    if not_found_titles:
        logger.warning(f"Could not find {len(not_found_titles)} papers in database out of {len(refs_dict)} total references")
    
    # 处理survey.tex，将\cite{label}替换为[idx]
    processed_survey, references_list = process_survey_tex(survey, label_to_paper)
    
    logger.info(f"Processed survey: {len(references_list)} references successfully mapped out of {len(refs_dict)} total BibTeX entries")
    
    cfg_for_topic = OmegaConf.merge(config, {
                                    "BasicInfo": {
                                        "topic": topic, 
                                        "evaluation_save_path": save_path
                                        }
                                    })
    return cfg_for_topic, processed_survey, references_list
                            

@hydra.main(config_path="../config/personal", config_name="evaluate_baseline", version_base=None)
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
                    elif "surveyforge" in baseline.lower():
                        cfg_for_topic, survey, references_list = get_SurveyForge_cfg(config, domain, topic, save_path, paper_path)
                    elif "deepsurvey" in baseline.lower():
                        cfg_for_topic, survey, references_list = get_DeepSurvey_cfg(config, domain, topic, save_path, paper_path)
                    elif "surveyx" in baseline.lower():
                        cfg_for_topic, survey, references_list = get_SurveyX_cfg(config, domain, topic, save_path, paper_path)
                    elif "human" in baseline.lower():
                        cfg_for_topic, survey, references_list = get_Human_cfg(config, domain, topic, save_path, paper_path)
                    elif "lira" in baseline.lower():
                        cfg_for_topic, survey, references_list = get_Lira_cfg(config, domain, topic, save_path, paper_path)
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
            domain_result = write_domain_result(save_path, domain, domain_results)
            for key, value in domain_result.items():
                logger.info(f"{key}: {value}\n")



    

if __name__ == "__main__":
    main()
